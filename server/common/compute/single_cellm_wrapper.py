from pathlib import Path
import logging

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
    def __init__(self):
        logging.info("Loading LLM embedding model...")
        # The model is the best-performing run from the `second` sweep
        # TODO should go to /home/moritz/Projects/single-cellm/modules/cellxgene/server/default_config.py and server/common/config respectively
        self.model_path = Path(
            "~/single-cellm/results/wandb_logging/JointEmbed_Training/f6fjywkb/checkpoints/last.ckpt"
        ).expanduser()
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

        self.terms_json_path = get_path(["paths", "enrichr_terms_json"])
        assert (
            self.terms_json_path.exists()
        ), "EnrichR terms json file does not exist. Please run single_cellm/src/validation/zero_shot/write_enrichr_terms.py"

        logging.info("Loading done")

    def preprocess_data(self, adaptor):
        """
        Preprocess data for LLM embeddings, making sure that subsequent API requests run fast
        """
        logging.info("Preprocessing data for LLM embeddings")

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

        expression = adaptor.data.copy()
        var_index_col_name = adaptor.get_schema()["annotations"]["var"]["index"]
        expression.var.index = adaptor.data.var[var_index_col_name].astype(str)
        obs_index_col_name = adaptor.get_schema()["annotations"]["obs"]["index"]
        expression.obs.index = adaptor.data.obs[obs_index_col_name].astype(str)

        obs_cols = [c for c, t in adaptor.data.obs.dtypes.items() if isinstance(t, CategoricalDtype)]

        similarity_scores_df = anndata_to_scored_keywords(
            transcriptome_input=expression[mask],
            model=self.pl_model.model,
            terms_json_path=self.terms_json_path,
            transcriptome_processor=self.transcriptome_processor,
            text_tokenizer=self.tokenizer,
            average_mode="embeddings",
            batch_size=64,
            obs_cols=obs_cols,
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
        structured_text = top_5_entries.groupby("library")["labels"].apply(list).to_dict()

        return {"text": structured_text}

    def llm_text_to_annotations(self, adaptor, text) -> pd.Series:
        """
        Embed the given text into the LLM space and return the similarity of each cell to the text. The similarity will be used as new cell-level annotation
        """
        expression = adaptor.data.copy()
        var_index_col_name = adaptor.get_schema()["annotations"]["var"]["index"]
        expression.var.index = adaptor.data.var[var_index_col_name]
        # Converts an obs index of "0", "1", ... to "TTTGCATGAGAGGC-1", ...
        obs_index_col_name = adaptor.get_schema()["annotations"]["obs"]["index"]
        expression.obs.index = adaptor.data.obs[obs_index_col_name].astype(str)

        # TODO similarities is [nan, ...]
        similarities, unnormalized_similarities = score_text_vs_transcriptome_many_vs_many(
            model=self.pl_model.model,
            transcriptome_input=expression,
            text_list_or_text_embeds=[text],
            transcriptome_processor=self.transcriptome_processor,
            text_tokenizer=self.tokenizer,
            average_mode=None,
            batch_size=64,
            score_norm_method="zscore",
            transcriptome_annotations=expression.obs.index,
        )

        return pd.Series(unnormalized_similarities.squeeze(0).cpu().detach())
