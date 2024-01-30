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

        self.terms_json_path = get_path(["paths", "enrichr_terms_json"])
        assert (
            self.terms_json_path.exists()
        ), "EnrichR terms json file does not exist. Please run single_cellm/src/validation/zero_shot/write_enrichr_terms.py"

        logging.info("Loading done")

        self.model_processed_data = None

    def preprocess_data(self, adaptor, dataset_path):
        """
        Preprocess data for LLM embeddings, making sure that subsequent API requests run fast
        TODO: simply add the transcriptome_embeds to the adata object (as a layer or obsm or so). then the `dataset_path` is not necessary anymore and we can call this from the constructor right away

        adaptor: Access to the adata object
        dataset_path: Just to extract the dataset_name (might be better to directly pass the dataset_name)
        """
        logging.info("Preprocessing data for LLM embeddings")

        try:
            model_id = adaptor.data.uns["model"]
        except KeyError:
            model_id = Path(self.model_path).stem
            # check that model_id is alphanumerical
            if len(model_id) != 8 or not model_id.isalnum():
                logging.warning(f"Model id {model_id} does not seem to be valid. Aborting preprocessing.")
                model_id = None

        try:
            dataset = adaptor.data.uns["dataset"]
        except KeyError:
            try:
                dataset = re.search(r"results/([^/]+)/.+\.h5ad", dataset_path).group(1)
            except AttributeError:
                logging.warning(f"Could not infer dataset name from path {dataset_path}. Aborting preprocessing.")
                dataset = None

        model_processed_data_path = get_path(["paths", "model_processed_dataset"], dataset=dataset, model=model_id)
        if model_processed_data_path.exists():
            logging.info("Loading precomputed data")
            self.model_processed_data = np.load(model_processed_data_path, allow_pickle=True)
            # only retain the embeddings (to save RAM)
            self.model_processed_data = {k: self.model_processed_data[k] for k in ["orig_ids", "transcriptome_embeds"]}
        else:
            self.model_processed_data = None
            logging.info("Precomputed data file does not exist. Precompute using FrozenCachedModel")
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

        if self.model_processed_data is not None:
            selected_obs_ids = set(adaptor.data.obs[obs_index_col_name].astype(str)[mask])
            selector = [str(orig_id) in selected_obs_ids for orig_id in self.model_processed_data["orig_ids"]]
            transcriptomes = torch.from_numpy(self.model_processed_data["transcriptome_embeds"][selector])
            transcriptomes = transcriptomes.to(self.pl_model.model.device)

            if sum(selector) != len(selected_obs_ids):
                logging.warning(
                    f"Precomputed embeddings seem to lack entries: {sum(selector)} of {len(selected_obs_ids)} found"
                )
        else:
            # Provide raw read counts, which will be processed by the model
            transcriptomes = adaptor.data[mask].copy()
            transcriptomes.var.index = adaptor.data.var[var_index_col_name].astype(str)
            transcriptomes.obs.index = adaptor.data.obs.loc[mask, obs_index_col_name].astype(str)

        # Get all categorical columns (too extensive and doesn't make sense)
        # obs_cols = [c for c, t in adaptor.data.obs.dtypes.items() if isinstance(t, CategoricalDtype)]
        # additional_text_dict = {
        #     obs_col: adaptor.data.obs[obs_col].astype(str).unique().tolist() for obs_col in obs_cols
        # }

        similarity_scores_df = anndata_to_scored_keywords(
            transcriptome_input=transcriptomes,
            model=self.pl_model.model,
            terms_json_path=self.terms_json_path,
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

        obs_ids = adaptor.data.obs[obs_index_col_name].astype(str).values
        obs_ids_lookup = {v: k for k, v in enumerate(obs_ids)}
        if self.model_processed_data is not None:
            # Do fast embeddings lookup (order matters, so we select by indices rather than by mask)
            selector = [obs_ids_lookup[str(orig_id)] for orig_id in self.model_processed_data["orig_ids"]]
            transcriptomes = torch.from_numpy(self.model_processed_data["transcriptome_embeds"][selector])
            transcriptomes = transcriptomes.to(self.pl_model.model.device)
        else:
            # Provide raw read counts, which will be processed by the model
            transcriptomes = adaptor.data.copy()  # NOTE copy is slow!
            transcriptomes.var.index = adaptor.data.var[var_index_col_name]
            transcriptomes.obs.index = adaptor.data.obs[obs_index_col_name].astype(str)

        scores, _ = score_text_vs_transcriptome_many_vs_many(
            model=self.pl_model.model,
            transcriptome_input=transcriptomes,
            text_list_or_text_embeds=[text],
            transcriptome_processor=self.transcriptome_processor,
            text_tokenizer=self.tokenizer,
            average_mode=None,
            batch_size=64,
            score_norm_method=None,
            transcriptome_annotations=obs_ids,
        )

        return pd.Series(scores.squeeze(0).cpu().detach())
