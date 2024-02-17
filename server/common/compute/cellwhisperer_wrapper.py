from pathlib import Path
import logging

import re
import os
import pandas as pd
import numpy as np

import requests
import pickle
import torch

from cellwhisperer.jointemb.processing import TranscriptomeTextDualEncoderProcessor
from cellwhisperer.utils.inference import score_transcriptomes_vs_texts, rank_terms_by_score, prepare_terms
from cellwhisperer.jointemb.single_cellm_lightning import TranscriptomeTextDualEncoderLightning
import torch
from cellwhisperer.config import get_path, model_path_from_name
from cellwhisperer.utils.model_io import load_cellwhisperer_model


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
        else:
            self.pl_model = None
            self.api_url = model_path_or_url

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
            transcriptomes = torch.from_numpy(adaptor.data.obsm["transcriptome_embeds"][mask])
            # transcriptomes = transcriptomes.to(self.pl_model.model.device)
        else:
            # Provide raw read counts, which will be processed by the model
            try:
                transcriptomes = adaptor.data[mask].to_memory(copy=True)
            except MemoryError:
                raise

            transcriptomes.var.index = adaptor.data.var[var_index_col_name].astype(str)
            transcriptomes.obs.index = adaptor.data.obs.loc[mask, obs_index_col_name].astype(str)

        # Get all categorical columns (too extensive and doesn't make sense)
        # obs_cols = [c for c, t in adaptor.data.obs.dtypes.items() if isinstance(t, CategoricalDtype)]
        # additional_text_dict = {
        #     obs_col: adaptor.data.obs[obs_col].astype(str).unique().tolist() for obs_col in obs_cols
        # }
        terms = adaptor.data.uns["terms"]

        terms_df = prepare_terms(terms)  # additional_text_dict

        if self.pl_model is None:
            # Call the model via API
            data = pickle.dumps((transcriptomes, terms_df["term"].to_list(), "embeddings", None, "zscore"))

            # Send the POST request with the serialized data
            response = requests.post(self.api_url + "/score_transcriptomes_vs_texts", data=data)

            # Check if the request was successful
            if response.status_code == 200:
                # Deserialize the response data
                scores = pickle.loads(response.content)
            else:
                logging.warning(f"Request failed with status code {response.status_code} and error {response.content}")
                raise RuntimeError(f"Request to model API failed: {response.status_code}")

        else:
            scores, _ = score_transcriptomes_vs_texts(
                model=self.pl_model.model,
                transcriptome_input=transcriptomes,
                text_list_or_text_embeds=terms_df["term"].to_list(),
                average_mode="embeddings",
                text_tokenizer=self.tokenizer,
                transcriptome_processor=self.transcriptome_processor,
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
            transcriptomes = torch.from_numpy(adaptor.data.obsm["transcriptome_embeds"])
            # transcriptomes = transcriptomes.to(self.pl_model.model.device)
        else:
            assert self.pl_model is not None, "Model is not loaded, so embeddings need to be preprocessed in advance"
            # Provide raw read counts, which will be processed by the model
            transcriptomes = adaptor.data.to_memory(copy=True)  # NOTE copy is slow!
            transcriptomes.var.index = adaptor.data.var[var_index_col_name]
            transcriptomes.obs.index = adaptor.data.obs[obs_index_col_name].astype(str)

        texts = text.split("MINUS")
        assert len(texts) in [1, 2], "At max. one MINUS sign allowed"

        if self.pl_model is None:
            # Serialize your input data
            data = pickle.dumps(
                (transcriptomes, texts, None, adaptor.data.obs[obs_index_col_name].astype(str).values, None)
            )

            # Send the POST request with the serialized data
            response = requests.post(self.api_url + "/score_transcriptomes_vs_texts", data=data)

            # Check if the request was successful
            if response.status_code == 200:
                # Deserialize the response data
                scores = pickle.loads(response.content)
            else:
                logging.warning(f"Request failed with status code {response.status_code}, {response.content}")
                raise RuntimeError(f"Request to model API failed: {response.status_code}")
        else:
            scores, _ = score_transcriptomes_vs_texts(
                model=self.pl_model.model,
                transcriptome_input=transcriptomes,
                text_list_or_text_embeds=texts,
                transcriptome_processor=self.transcriptome_processor,
                text_tokenizer=self.tokenizer,
                average_mode=None,
                batch_size=64,
                score_norm_method=None,
                grouping_keys=adaptor.data.obs[obs_index_col_name].astype(str).values,
            )

        if len(texts) == 2:
            scores = scores[0] - scores[1]
        else:
            scores = scores[0]

        return pd.Series(scores.cpu().detach())
