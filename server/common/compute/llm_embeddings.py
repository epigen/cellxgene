from os import popen
from pathlib import Path
import logging

import numpy as np
from scipy import sparse, stats
import pandas as pd
import torch

from single_cellm.jointemb.model import TranscriptomeTextDualEncoderModel, TranscriptomeTextDualEncoderLightning
from single_cellm.jointemb.geneformer_model import GeneformerTranscriptomeProcessor
from transformers import AutoTokenizer
from single_cellm.jointemb.datautils_JointEmbed import JointEmbedDataset
from single_cellm.validation.zero_shot.transcriptomes_to_scored_keywords import \
    write_enrichr_terms_to_json, anndata_to_scored_keywords

logger = logging.getLogger(__name__)
import subprocess
import yaml

# use subprocess to get git root
parent_repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).strip().decode("utf-8")).parents[1]
with open(parent_repo / "config.yaml") as f:
    parent_config = yaml.safe_load(f)


logging.info("Loading LLM embedding model...")
# Model loading

model_path = Path(
    "~/single-cellm/results/jointEmbed/lightning_logging/JointEmbed_Training/9dwkh6rl/checkpoints/epoch=5-val_loss=2.71.ckpt"
).expanduser()

# model = TranscriptomeTextDualEncoderModel.from_pretrained(geneformer_biogpt_model_path).to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pl_model = TranscriptomeTextDualEncoderLightning.load_from_checkpoint(model_path)
pl_model.eval().to(device)

tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
# image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
transcriptome_processor = GeneformerTranscriptomeProcessor(nproc=4, emb_label=parent_config["anndata_label_name"])

logging.info("Preparing EnrichR terms...")
terms_json_path = Path("~/projects/single-cellm/resources/enrichr_terms/terms.json").expanduser()
if not terms_json_path.exists():
    write_enrichr_terms_to_json(terms_json_path=terms_json_path)
else:
    logging.info("EnrichR terms already exist - skipping")

logging.info("Preparing done")

logging.info("Loading done")
# processor = TranscriptomeTextDualEncoderProcessor(transcriptome_processor, tokenizer)


def llm_obs_to_text(adaptor, mask):
    """
    Embed the given cells into the LLM space and return their average similarity to different keywords as formatted text.
    Keyword types used for comparison are: (i) selected enrichR terms (see single_cellm.validation.zero_shot.transcriptomes_to_scored_keywords.write_enrichr_terms_to_json) \
    and (ii) cell type annotations (currently all values in adata.obs.columns). For more info, see single_cellm.validation.zero_shot.transcriptomes_to_scored_keywords.
    :param adaptor: DataAdaptor instance
    :param mask: 
    :return:  dictionary  {text: }
    """

    # create anndata object from adaptor
    data = adaptor.get_X_array(mask, None)
    expression = data.copy()
    var_index_col_name = adaptor.get_schema()["annotations"]["var"]["index"] 
    expression.var.index = adaptor.data.var[var_index_col_name]

    obs_cols = expression.obs.columns # TODO is this the right way to get the obs columns?

    # get top n keywords by similarity
    top_n_text = anndata_to_scored_keywords(adata=expression,
                                       model=pl_model,
                                       terms_json_path=terms_json_path,
                                       transcriptome_processor=transcriptome_processor,
                                       text_tokenizer=tokenizer,
                                       device=device,
                                       average_mode="cells", # TODO test which method is better, "cells" or "embeddings"
                                       chunk_size_text_emb_and_scoring=64,
                                       n_top_per_term=5,
                                       obs_cols=["cell type", "cell type rough"],
                                       score_norm_method="zscore", # TODO test which method is better
                                       return_mode="text")
    
    # TODO Do we need a caching mechanism here?
    
    return {"text": top_n_text}


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
