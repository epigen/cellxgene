import logging
import math
import os
import json
import pandas as pd
import numpy as np
from typing import List, Union

import requests
import pickle
import torch

from cellwhisperer.utils.inference import (
    score_transcriptomes_vs_texts,
    rank_terms_by_score,
    prepare_terms,
    # gene_score_contributions,
)
import torch
from cellwhisperer.utils.model_io import load_cellwhisperer_model
from . import llava_utils, llava_conversation

default_conversation = llava_conversation.conv_mistral_instruct

logger = logging.getLogger(__name__)

MODEL_NAME = "Mistral-7B-Instruct-v0.2__cellwhisperer_clip_v1"


class CellWhispererWrapper:
    def __init__(self, model_path_or_url: str):
        """
        Load the model from the given path or use it via the given URL
        """
        if os.path.exists(model_path_or_url):
            logger.info("Loading LLM embedding model...")
            self.pl_model, self.tokenizer, self.transcriptome_processor = load_cellwhisperer_model(
                model_path_or_url, cache=True
            )
            logger.info("Loading done")
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

        NOTE: In a deployment setting with many services, this function might block/lock your server and lead to timeouts. Use with care.
        """
        logger.info("Preprocessing data for LLM embeddings, making sure it's fast")

        # Make sure that all the zero-shot class terms are embedded
        mask = np.zeros(adaptor.data.shape[0], dtype=bool)
        mask[0] = True  # Generate mask with single element
        self.llm_obs_to_text(adaptor, mask=mask)

        # Embed all cells
        self.llm_text_to_annotations(adaptor, text="test")

        response = requests.post(self.api_url + "/store_cache")

    def llm_obs_to_text(self, adaptor, mask):
        """
        Currently unused, in favor of the more advanced chat functionality, but still functional

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
        logger.info(f"Search request: {text}")
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
                logger.warning(f"Request failed with status code {response.status_code}, {response.content}")
                raise RuntimeError(f"Request to model API failed: {response.status_code}")
        else:
            assert self.pl_model is not None, "Model is not loaded, but querying API for text embedding failed as well"
            text_embeds = self.pl_model.model.embed_texts(texts)

        return text_embeds

    def _prepare_messages(self, adaptor, messages, mask):
        # Extract necessary information from the request
        transcriptome_embeds = adaptor.data.obsm["transcriptome_embeds"][mask].mean(axis=0).tolist()

        # Heuristic to get the top genes efficiently (computing it is infeasible, due to the (recommended) use of CSC matrices)
        # Equivalent to slower `pd.Series(df.values.ravel()).value_counts()`
        top_genes_df = adaptor.data.obsm["top_genes"][mask]
        codes = np.concatenate(
            [
                np.repeat(
                    top_genes_df[col].cat.codes.values, math.ceil(math.log2(100 - i))
                )  # give top1 genes more weight than top 100 genes
                for i, col in enumerate(top_genes_df.columns)
            ]
        )
        counts = np.bincount(codes, minlength=len(top_genes_df["Top_1"].cat.categories))
        category_counts = pd.Series(counts, index=top_genes_df[top_genes_df.columns[0]].cat.categories)
        n_top_genes = 50  # NOTE number of top genes to list should be configurable
        top_genes = category_counts.sort_values(ascending=False).index[:n_top_genes].to_list()

        # Initialize the conversation
        state = default_conversation.copy()

        # TODO consider including both normalized and unnormalized genes. Why? A reviewer might check whether the genes are amongst the top expressed ones
        state.messages = [
            [
                "USER",
                f"Help me analyzing this sample of cells. Respond in proper english in a tone of uncertainty and focus on the biology of the sample rather than any potential donor or patient information (e.g. do not mention age and sex). Start by listing the top {n_top_genes} genes.",
            ],
            [
                "ASSISTANT",
                f"Sure. I'll respond as you requested, focusing on the sample of cells and avoiding any personal information. It looks like the 20 top normalized genes are {', '.join(top_genes[:20])}.\nStill remarkably strong expressed seem the genes {', '.join(top_genes[20:n_top_genes])}. Note that there are even more strongly expressed genes beyond the ones I just listed.",
            ],
        ]
        state.offset = 2

        # NOTE: the transcriptome is added too late. consider changing

        for i, message in enumerate(messages):
            if i == 0:
                assert message["from"] == "human"
                llava_utils.add_text(state, message["value"], transcriptome_embeds, "Transcriptome")
            else:
                role = {"human": state.roles[0], "gpt": state.roles[1]}[message["from"]]
                state.append_message(role, message["value"])

        return state

    def llm_feedback(self, adaptor, messages, mask, thumbDirection):
        """
        Log the values of the conversation
        """
        state = self._prepare_messages(adaptor, messages, mask)

        llava_utils.log_state(state, thumbDirection, MODEL_NAME)

    def llm_chat(self, adaptor, messages, mask, temperature):
        state = self._prepare_messages(adaptor, messages, mask)

        state.append_message(state.roles[1], None)

        for chunk in llava_utils.http_bot(state, MODEL_NAME, temperature, top_p=0.7, max_new_tokens=512, log=True):
            yield json.dumps({"text": chunk}).encode() + b"\x00"

    def gene_score_contributions(self, adaptor, prompt, mask) -> pd.Series:
        """
        Which genes increase or decrease the prompt-similiarity in the selected cells?
        """
        raise NotImplementedError("Analysis showed that this is not working as expected")

        var_index_col_name = adaptor.get_schema()["annotations"]["var"]["index"]
        obs_index_col_name = adaptor.get_schema()["annotations"]["obs"]["index"]
        try:
            transcriptomes = adaptor.data[mask].to_memory(copy=True)
        except MemoryError:
            raise

        transcriptomes.var.index = adaptor.data.var[var_index_col_name].astype(str)
        transcriptomes.obs.index = adaptor.data.obs.loc[mask, obs_index_col_name].astype(str)

        text_embeds = self._embed_texts([prompt])

        gene_contribs: pd.Series = gene_score_contributions(  # NOTE: note implemented
            transcriptome_input=transcriptomes,
            text_list_or_text_embeds=text_embeds,
            logit_scale=self.logit_scale,
            score_norm_method=None,
        ).sort_values()

        top_bottom: pd.Series = pd.concat([gene_contribs.iloc[:10], gene_contribs.iloc[-10:]])  # type: ignore
        return top_bottom
