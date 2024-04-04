import logging

import os
import json
import pandas as pd
import numpy as np
from typing import List

import requests
import pickle
import torch

from cellwhisperer.utils.inference import (
    score_transcriptomes_vs_texts,
    rank_terms_by_score,
    prepare_terms,
    gene_score_contributions,
)
import torch
from cellwhisperer.utils.model_io import load_cellwhisperer_model
from . import llava_utils, llava_conversation

default_conversation = llava_conversation.conv_mistral_instruct

logger = logging.getLogger(__name__)


class CellWhispererWrapper:
    def __init__(self, model_path_or_url: str):
        """
        Load the model from the given path or use it via the given URL
        """
        if os.path.exists(model_path_or_url):
            logging.info("Loading LLM embedding model...")
            self.pl_model, self.tokenizer, self.transcriptome_processor = load_cellwhisperer_model(
                model_path_or_url, cache=True
            )
            logging.info("Loading done")
            self.logit_scale = self.pl_model.model.discriminator.temperature.exp()
        else:
            self.pl_model = None
            self.api_url = model_path_or_url
            # load logit_scale via API
            response = requests.get(self.api_url + "/logit_scale")
            self.logit_scale = float(response.content)

    def preprocess_data(self, adaptor):
        """
        Preprocess data for LLM embeddings, making sure that subsequent API requests run fast.
        If things are cached already (through frozenmodel and/or the adaptor) this will be fast

        adaptor: Access to the adata object
        """
        logging.info("Preprocessing data for LLM embeddings, making sure it's fast")
        return  # just for testing

        # Make sure that all the zero-shot class terms are embedded
        mask = np.zeros(adaptor.data.shape[0], dtype=bool)
        mask[0] = True  # Generate mask with single element
        self.llm_obs_to_text(adaptor, mask=mask)

        # Embed all cells
        self.llm_text_to_annotations(adaptor, text="test")

        response = requests.post(self.api_url + "/store_cache")

    def llm_obs_to_text(self, adaptor, mask):
        """
        Embed the given cells into the LLM space and return their average similarity to different keywords as formatted text.
        Keyword types used for comparison are: (i) selected enrichR terms (see cellwhisperer.validation.zero_shot.functions.write_enrichr_terms_to_json) \
        and (ii) cell type annotations (currently all values in adata.obs.columns). For more info, see cellwhisperer.validation.zero_shot.functions.
        :param adaptor: DataAdaptor instance
        :param mask:
        :return:  dictionary  {text: }
        """
        var_index_col_name = adaptor.get_schema()["annotations"]["var"]["index"]
        obs_index_col_name = adaptor.get_schema()["annotations"]["obs"]["index"]

        if "transcriptome_embeds" in adaptor.data.obsm:
            transcriptome_embeds = torch.from_numpy(adaptor.data.obsm["transcriptome_embeds"][mask])
            # transcriptomes = transcriptomes.to(self.pl_model.model.device)
        else:
            # Provide raw read counts, which will be processed by the model
            try:
                transcriptomes = adaptor.data[mask].to_memory(copy=True)
            except MemoryError:
                raise

            transcriptomes.var.index = adaptor.data.var[var_index_col_name].astype(str)
            transcriptomes.obs.index = adaptor.data.obs.loc[mask, obs_index_col_name].astype(str)
            transcriptome_embeds = self.pl_model.embed_transcriptomes(transcriptomes)

        # Get all categorical columns (too extensive and doesn't make sense)
        # obs_cols = [c for c, t in adaptor.data.obs.dtypes.items() if isinstance(t, CategoricalDtype)]
        # additional_text_dict = {
        #     obs_col: adaptor.data.obs[obs_col].astype(str).unique().tolist() for obs_col in obs_cols
        # }
        terms = adaptor.data.uns["terms"]

        terms_df = prepare_terms(terms)  # additional_text_dict
        text_embeds = self._embed_texts(terms_df["term"].to_list())

        scores, _ = score_transcriptomes_vs_texts(
            transcriptome_input=transcriptome_embeds,
            text_list_or_text_embeds=text_embeds,
            logit_scale=self.logit_scale,
            average_mode="embeddings",
            score_norm_method="zscore",
        )  # n_text * 1

        similarity_scores_df = rank_terms_by_score(scores, terms_df)

        top_5_entries = (
            similarity_scores_df.query("logits > 0.0")  # drop negatives
            .groupby("library")
            .apply(lambda x: x.nlargest(5, "logits"))
            .reset_index(drop=True)
        )

        # Combine the term names with the scores (logits)
        top_5_entries["labels"] = top_5_entries["term"] + " (" + top_5_entries["logits"].astype(str) + ")"

        # Combine the term names with the scores (logits)
        top_5_entries["labels"] = top_5_entries["term"] + " (" + top_5_entries["logits"].astype(str) + ")"

        # Group by 'library' and create a list of 'labels'
        grouped = top_5_entries.groupby("library")

        # Find the maximum logits value for each group
        max_logits_per_group = grouped["logits"].max()

        # Sort the groups by the maximum logits value in descending order
        sorted_groups = max_logits_per_group.sort_values(ascending=False)

        # Generate the final object to return, sorted by strongest hits on the library-level
        structured_text = [
            {
                "library": library,
                "keywords": grouped.get_group(library)["labels"].tolist(),
            }
            for library in sorted_groups.index
        ]
        return structured_text

    def llm_text_to_annotations(self, adaptor, text) -> pd.Series:
        """
        Embed the given text into the LLM space and return the similarity of each cell to the text. The similarity will be used as new cell-level annotation
        """
        # Converts an obs index of "0", "1", ... to "TTTGCATGAGAGGC-1", ...
        obs_index_col_name = adaptor.get_schema()["annotations"]["obs"]["index"]
        var_index_col_name = adaptor.get_schema()["annotations"]["var"]["index"]

        if "transcriptome_embeds" in adaptor.data.obsm:
            transcriptome_embeds = torch.from_numpy(adaptor.data.obsm["transcriptome_embeds"])
            # transcriptomes = transcriptomes.to(self.pl_model.model.device)
        else:
            assert self.pl_model is not None, "Model is not loaded, so embeddings need to be preprocessed in advance"
            # Provide raw read counts, which will be processed by the model
            transcriptomes = adaptor.data.to_memory(copy=True)  # NOTE copy is slow!
            transcriptomes.var.index = adaptor.data.var[var_index_col_name]
            transcriptomes.obs.index = adaptor.data.obs[obs_index_col_name].astype(str)
            transcriptome_embeds = self.pl_model.embed_transcriptomes(transcriptomes)

        texts = text.split("MINUS")
        assert len(texts) in [1, 2], "At max. one MINUS sign allowed"
        text_embeds = self._embed_texts(texts)

        scores, _ = score_transcriptomes_vs_texts(
            transcriptome_input=transcriptome_embeds,
            text_list_or_text_embeds=text_embeds,
            logit_scale=self.logit_scale,
            average_mode=None,
            batch_size=64,
            score_norm_method=None,
            grouping_keys=adaptor.data.obs[obs_index_col_name].astype(str).values,
        )

        if len(text_embeds) == 2:
            scores = scores[0] - scores[1]
        else:
            scores = scores[0]

        return pd.Series(scores.cpu().detach())

    def _embed_texts(self, texts: List[str]):
        if self.pl_model is None:
            # Serialize your input data
            # Send the POST request with the json-list of texts
            response = requests.post(self.api_url + "/text_embedding", json=texts)

            # Check if the request was successful
            if response.status_code == 200:
                # Deserialize the response data
                text_embeds = torch.from_numpy(pickle.loads(response.content))
            else:
                logging.warning(f"Request failed with status code {response.status_code}, {response.content}")
                raise RuntimeError(f"Request to model API failed: {response.status_code}")
        else:
            assert self.pl_model is not None, "Model is not loaded, but querying API for text embedding failed as well"
            text_embeds = self.pl_model.embed_texts(texts)

        return text_embeds

    def llm_chat(self, adaptor, messages, mask):
        # Extract necessary information from the request
        transcriptome_embeds = adaptor.data.obsm["transcriptome_embeds"][mask].mean(axis=0).tolist()

        transcriptomes = adaptor.data.X[mask]
        if transcriptomes.shape[0] > 10000:
            logging.warning("Too many cells to process, sampling 10k cells")
            transcriptomes = transcriptomes[np.random.choice(transcriptomes.shape[0], 10000, replace=False)]
        mean_transcriptome = transcriptomes.mean(axis=0).A1

        # Compute top genes
        try:
            mean_normalized_transcriptome = (
                mean_transcriptome - adaptor.data.var["log1p_normalizer"].to_numpy()
            )  # normalize in logspace via difference
        except KeyError:
            logging.warning("No log1p_normalizer found in var. Using unnormalized log1ps to compute top genes")
            mean_normalized_transcriptome = mean_transcriptome

        n_top_genes = 20  # TODO needs to become config
        top_genes = (
            pd.Series(data=mean_normalized_transcriptome, index=adaptor.data.var.gene_name)
            .nlargest(n_top_genes)
            .index.tolist()
        )

        # Initialize the conversation
        state = default_conversation.copy()

        # TODO consider including both normalized and unnormalized genes. Why? A reviewer might check whether the genes are amongst the top expressed ones
        state.messages = [
            [
                "USER",
                f"Help me analyzing this sample of cells. Always respond in proper english sentences and in a tone of uncertainty. Start by listing the top {n_top_genes} genes.",
            ],
            [
                "ASSISTANT",
                f"Sure, It looks like the top normalized genes are {', '.join(top_genes)}.",
            ],
        ]
        state.offset = 2

        # TODO the transcriptome is added too late. consider changing

        for i, message in enumerate(messages):
            if i == 0:
                assert message["from"] == "human"
                llava_utils.add_text(state, message["value"], transcriptome_embeds, "Transcriptome")
            else:
                role = {"human": state.roles[0], "gpt": state.roles[1]}[message["from"]]
                state.append_message(role, message["value"])
        state.append_message(state.roles[1], None)

        # TODO need to make CONTROLLER_URL flexible in there
        for chunk in llava_utils.http_bot(
            state, "Mistral-7B-Instruct-v0.2__03jujd8s", temperature=0.2, top_p=0.7, max_new_tokens=512
        ):
            yield json.dumps({"text": chunk}).encode() + b"\x00"

    def gene_score_contributions(self, adaptor, prompt, mask) -> pd.Series:
        """
        Which genes increase or decrease the prompt-similiarity in the selected cells?
        """

        # TODO need to calculate it I believe
        transcriptome_embeds = adaptor.data.obsm["transcriptome_embeds"][mask].mean(axis=0).tolist()

        text_embeds = self._embed_texts([prompt])

        gene_contribs: pd.Series = gene_score_contributions(
            transcriptome_input=transcriptome_embeds,
            text_list_or_text_embeds=text_embeds,
            logit_scale=self.logit_scale,
            average_mode=None,
            score_norm_method=None,
        ).sort_values()

        top_bottom: pd.Series = pd.concat([gene_contribs.iloc[:10], gene_contribs.iloc[-10:]])  # type: ignore
        return top_bottom

    def manual_chat_request():
        """
        unused in favor of function borrowed from llava
        DEPRECATED/DELETE

        """
        # Construct the payload for the worker
        pload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "images": [transcriptome_embeds],
        }

        # Get the worker address from the controller
        worker_addr_response = requests.post(f"{controller_url}/get_worker_address", json={"model": model})
        worker_addr = worker_addr_response.json()["address"]
        print(worker_addr)

        # Stream the response
        with requests.post(
            f"{worker_addr}/worker_generate_stream", headers={"User-Agent": "LLaVA Client"}, json=pload, stream=True
        ) as r:
            for chunk in r.iter_lines(delimiter=b"\x00"):
                if chunk:
                    yield chunk + b"\x00"
