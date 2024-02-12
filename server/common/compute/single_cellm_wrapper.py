from pathlib import Path
import logging

import re
import pandas as pd
import numpy as np
from pandas.core.dtypes.dtypes import CategoricalDtype
from single_cellm.jointemb.config import TranscriptomeTextDualEncoderConfig
from single_cellm.jointemb.processing import TranscriptomeTextDualEncoderProcessor
import torch
from single_cellm.config import get_path, model_path_from_name

from single_cellm.jointemb.single_cellm_lightning import TranscriptomeTextDualEncoderLightning
from single_cellm.jointemb.geneformer_model import GeneformerTranscriptomeProcessor
from transformers import AutoTokenizer
from single_cellm.validation.zero_shot.functions import (
    anndata_to_scored_keywords,
    formatted_text_from_df,
    score_text_vs_transcriptome_many_vs_many,
)

import subprocess
import yaml

logger = logging.getLogger(__name__)


class SingleCeLLMWrapper:
    def __init__(self, model_path):
        logging.info("Loading LLM embedding model...")
        # The model is the best-performing run from the `second` sweep

        # TODO load model with function from src/single_cellm/utils/model_io.py
        self.model_path = Path(model_path).expanduser()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pl_model = TranscriptomeTextDualEncoderLightning.load_from_checkpoint(self.model_path)
        self.pl_model.eval().to(self.device)
        self.pl_model.model.prepare_models(
            self.pl_model.model.transcriptome_model, self.pl_model.model.text_model, force_freeze=True
        )
        self.pl_model.freeze()

        # TODO transcriptome_processor_kwargs might be missing
        self.processor = TranscriptomeTextDualEncoderProcessor(
            self.pl_model.model.transcriptome_model.config.model_type,
            model_path_from_name(self.pl_model.model.text_model.config.model_type),
        )

        self.tokenizer = self.processor.tokenizer
        self.transcriptome_processor = self.processor.transcriptome_processor

        logging.info("Loading done")

    def preprocess_data(self, adaptor):
        """
        Preprocess data for LLM embeddings, making sure that subsequent API requests run fast.
        If things are cached already (through frozenmodel and/or the adaptor) this will be fast

        adaptor: Access to the adata object
        """
        logging.info("Preprocessing data for LLM embeddings, making sure it's fast")

        # Make sure that all the zero-shot class terms are embedded
        mask = np.zeros(adaptor.data.shape[0], dtype=bool)
        mask[0] = True  # Generate mask with single element
        self.llm_obs_to_text(adaptor, mask=mask)

        # Embed all cells
        self.llm_text_to_annotations(adaptor, text="test")

        # Store
        self.pl_model.model.store_cache()

    def llm_obs_to_text(self, adaptor, mask):
        """
        Embed the given cells into the LLM space and return their average similarity to different keywords as formatted text.
        Keyword types used for comparison are: (i) selected enrichR terms (see single_cellm.validation.zero_shot.functions.write_enrichr_terms_to_json) \
        and (ii) cell type annotations (currently all values in adata.obs.columns). For more info, see single_cellm.validation.zero_shot.functions.
        :param adaptor: DataAdaptor instance
        :param mask:
        :return:  dictionary  {text: }
        """
        var_index_col_name = adaptor.get_schema()["annotations"]["var"]["index"]
        obs_index_col_name = adaptor.get_schema()["annotations"]["obs"]["index"]

        if "transcriptome_embeds" in adaptor.data.obsm:
            transcriptomes = torch.from_numpy(adaptor.data.obsm["transcriptome_embeds"][mask])
            transcriptomes = transcriptomes.to(self.pl_model.model.device)
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

        similarity_scores_df = anndata_to_scored_keywords(
            transcriptome_input=transcriptomes,
            model=self.pl_model.model,
            terms=terms,
            transcriptome_processor=self.transcriptome_processor,
            text_tokenizer=self.tokenizer,
            average_mode="embeddings",
            batch_size=64,
            score_norm_method="zscore",
        )

        top_5_entries = (
            similarity_scores_df.query("logits > 0.0")  # drop negatives
            .groupby("library")
            .apply(lambda x: x.nlargest(5, "logits"))
            .reset_index(drop=True)
        )

        # Combine the term names with the scores (logits)
        top_5_entries["labels"] = (
            top_5_entries["term_without_prefix"] + " (" + top_5_entries["logits"].astype(str) + ")"
        )

        # Combine the term names with the scores (logits)
        top_5_entries["labels"] = (
            top_5_entries["term_without_prefix"] + " (" + top_5_entries["logits"].astype(str) + ")"
        )

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
            transcriptomes = transcriptomes.to(self.pl_model.model.device)
        else:
            # Provide raw read counts, which will be processed by the model
            transcriptomes = adaptor.data.to_memory(copy=True)  # NOTE copy is slow!
            transcriptomes.var.index = adaptor.data.var[var_index_col_name]
            transcriptomes.obs.index = adaptor.data.obs[obs_index_col_name].astype(str)

        texts = text.split("MINUS")
        assert len(texts) in [1, 2], "At max. one MINUS sign allowed"

        scores, _ = score_text_vs_transcriptome_many_vs_many(
            model=self.pl_model.model,
            transcriptome_input=transcriptomes,
            text_list_or_text_embeds=texts,
            transcriptome_processor=self.transcriptome_processor,
            text_tokenizer=self.tokenizer,
            average_mode=None,
            batch_size=64,
            score_norm_method=None,
            transcriptome_annotations=adaptor.data.obs[obs_index_col_name].astype(str).values,
        )
        if len(texts) == 2:
            scores = scores[0] - scores[1]
        else:
            scores = scores[0]

        return pd.Series(scores.cpu().detach())
