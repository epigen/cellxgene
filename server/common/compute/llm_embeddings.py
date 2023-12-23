from pathlib import Path
import logging

import pandas as pd
from pandas.core.dtypes.dtypes import CategoricalDtype
from single_cellm.jointemb.config import TranscriptomeTextDualEncoderConfig
from single_cellm.jointemb.processing import TranscriptomeTextDualEncoderProcessor
import torch
from single_cellm.config import get_path

from single_cellm.jointemb.lightning import TranscriptomeTextDualEncoderLightning
from single_cellm.jointemb.geneformer_model import GeneformerTranscriptomeProcessor
from transformers import AutoTokenizer
from single_cellm.validation.zero_shot.functions import (
    anndata_to_scored_keywords,
    formatted_text_from_df,
)

logger = logging.getLogger(__name__)
import subprocess
import yaml

# use subprocess to get git root
parent_repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).strip().decode("utf-8")).parents[1]
with open(parent_repo / "config.yaml") as f:
    parent_config = yaml.safe_load(f)

logging.info("Loading LLM embedding model...")
# Model loading

# TODO new model
model_path = Path(
    "~/single-cellm/results/wandb_logging/JointEmbed_Training/5e0elqs5/checkpoints/epoch=1-val_loss=3.46.ckpt"
).expanduser()

# model = TranscriptomeTextDualEncoderModel.from_pretrained(geneformer_biogpt_model_path).to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config_transcriptome = {"model_type": "geneformer"}
# config_text = {"model_type": "biogpt"}

# model_config = TranscriptomeTextDualEncoderConfig(
#     projection_dim=512,
#     logit_scale_init_value=2.6592,
#     transcriptome_config=config_transcriptome,
#     text_config=config_text,
# )


# pl_model = TranscriptomeTextDualEncoderLightning(
#     model_config
# )
pl_model = TranscriptomeTextDualEncoderLightning.load_from_checkpoint(model_path)
pl_model.eval().to(device)

processor = TranscriptomeTextDualEncoderProcessor(
    pl_model.model.transcriptome_model.config.model_type, pl_model.model.text_model.config.model_type
)

tokenizer = processor.get_tokenizer()
transcriptome_processor = processor.get_transcriptome_processor()

terms_json_path = get_path(["paths", "enrichr_terms_json"])
assert (
    terms_json_path.exists()
), "EnrichR terms json file does not exist. Please run single_cellm/src/validation/zero_shot/write_enrichr_terms.py"

logging.info("Loading done")


def llm_obs_to_text(adaptor, mask):
    """
    Embed the given cells into the LLM space and return their average similarity to different keywords as formatted text.
    Keyword types used for comparison are: (i) selected enrichR terms (see single_cellm.validation.zero_shot.functions.write_enrichr_terms_to_json) \
    and (ii) cell type annotations (currently all values in adata.obs.columns). For more info, see single_cellm.validation.zero_shot.functions.
    :param adaptor: DataAdaptor instance
    :param mask:
    :return:  dictionary  {text: }
    """

    # create anndata object from adaptor
    expression = adaptor.data.copy()
    # workaround (https://github.com/scverse/scanpy/issues/747#issuecomment-1242183366)
    var_index_col_name = adaptor.get_schema()["annotations"]["var"]["index"]
    expression.var.index = adaptor.data.var[var_index_col_name].astype(str)
    obs_index_col_name = adaptor.get_schema()["annotations"]["obs"]["index"]
    expression.obs.index = adaptor.data.obs[obs_index_col_name].astype(str)

    obs_cols = [c for c, t in adaptor.data.obs.dtypes.items() if isinstance(t, CategoricalDtype)]

    # get top n keywords by similarity
    similarity_scores_df = anndata_to_scored_keywords(
        transcriptome_input=expression[mask],
        model=pl_model.model,
        terms_json_path=terms_json_path,
        transcriptome_processor=transcriptome_processor,
        text_tokenizer=tokenizer,
        average_mode="cells",  # TODO test which method is better, "cells" or "embeddings"
        batch_size=64,
        obs_cols=obs_cols,  # Note that this might be confusing to users and we need to see whather this stays shall remain enabled
        score_norm_method="zscore",  # TODO test which method is better
    )

    similarity_formatted_text = formatted_text_from_df(similarity_scores_df, n_top_per_term=5)

    # TODO Implement a caching mechanism (e.g. could be deployed along with the model)

    return {"text": similarity_formatted_text}


def llm_text_to_annotations(adaptor, text):
    # text embedding
    text_tokens = tokenizer(text, return_tensors="pt", padding=True)
    # make sure text_tokens are on GPU
    for k, v in text_tokens.items():
        text_tokens[k] = v.to(device)
    _, text_embeds = pl_model.model.get_text_features(**text_tokens)

    # transcriptome embedding
    expression = adaptor.data.copy()
    var_index_col_name = adaptor.get_schema()["annotations"]["var"]["index"]
    expression.var.index = adaptor.data.var[var_index_col_name]

    # TODO this caching mechanism should be called upon cellxgene dataset loading and in the background
    try:
        llm_text_to_annotations.transcriptome_embeds
    except AttributeError:
        transcriptome_tokens = transcriptome_processor(expression, return_tensors="pt", padding=True)
        # make sure transcriptome_tokens are on GPU
        for k, v in transcriptome_tokens.items():
            transcriptome_tokens[k] = v.to(device)
        _, transcriptome_embeds = pl_model.model.get_transcriptome_features(**transcriptome_tokens)
        # normalized features
        transcriptome_embeds = transcriptome_embeds / transcriptome_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        llm_text_to_annotations.transcriptome_embeds = transcriptome_embeds

    # cosine similarity as logits  # TODO check CLIP paper to see whether we want logits here (it's a linear scaling so it does not matter actually)
    logit_scale = pl_model.model.logit_scale.exp()
    logits_per_text = torch.matmul(text_embeds, llm_text_to_annotations.transcriptome_embeds.t()) * logit_scale
    # logits_per_transcriptome = logits_per_text.T
    return pd.Series(logits_per_text.squeeze(0).cpu().detach())
