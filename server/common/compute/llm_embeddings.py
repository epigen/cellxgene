from os import popen
from pathlib import Path
import logging

import numpy as np
from scipy import sparse, stats
import pandas as pd
import torch

from server.common.constants import XApproximateDistribution
from single_cellm.jointemb.model import TranscriptomeTextDualEncoderModel
from single_cellm.jointemb.geneformer_model import GeneformerTranscriptomeProcessor
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


logging.info("Loading LLM embedding model...")
# Model loading
geneformer_biogpt_model_path = Path("~/projects/single-cellm/results/models/geneformer-biogpt").expanduser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TranscriptomeTextDualEncoderModel.from_pretrained(geneformer_biogpt_model_path).to(device)

tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
# image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
transcriptome_processor = GeneformerTranscriptomeProcessor(
    nproc=4, emb_label=model.transcriptome_model.config.emb_label
)
logging.info("Loading done")
# processor = TranscriptomeTextDualEncoderProcessor(transcriptome_processor, tokenizer)


def llm_obs_to_text(adaptor, mask):
    """
    Embed the given cells into the LLM space and return the text representation of the cells.

    Either take the mean of the observations (cells) before going into the shared embedding space. Or, embed all observations and take the mean of the embeddings

    :param adaptor: DataAdaptor instance
    :return:  dictionary  {text: }
    """

    X_approximate_distribution = adaptor.get_X_approximate_distribution()
    data = adaptor.get_X_array(mask, None)

    # TODO call LLM etc.

    return {"text": " ".join(["foo", "bar"])}


def llm_text_to_annotations(adaptor, text):
    # text embedding
    text_tokens = tokenizer(text, return_tensors="pt", padding=True)
    # make sure text_tokens are on GPU
    for k, v in text_tokens.items():
        text_tokens[k] = v.to(device)
    _, text_embeds = model.get_text_features(**text_tokens)

    # transcriptome embedding
    expression = adaptor.data.copy()
    var_index_col_name = adaptor.get_schema()["annotations"]["var"]["index"]
    expression.var.index = adaptor.data.var[var_index_col_name]

    transcriptome_tokens = transcriptome_processor(expression, return_tensors="pt", padding=True)
    # make sure transcriptome_tokens are on GPU
    for k, v in transcriptome_tokens.items():
        transcriptome_tokens[k] = v.to(device)

    # TODO this caching mechanism should be called upon cellxgene dataset loading and in the background
    try:
        llm_text_to_annotations.transcriptome_embeds
    except AttributeError:
        pad_token_id = transcriptome_processor.tokenizer.gene_token_dict.get("<pad>")
        _, transcriptome_embeds = model.get_transcriptome_features(**transcriptome_tokens, pad_token_id=pad_token_id)
        # normalized features
        transcriptome_embeds = transcriptome_embeds / transcriptome_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        llm_text_to_annotations.transcriptome_embeds = transcriptome_embeds

    # cosine similarity as logits  # TODO check CLIP paper to see whether we want logits here (it's a linear scaling so it does not matter actually)
    logit_scale = model.logit_scale.exp()
    logits_per_text = torch.matmul(text_embeds, llm_text_to_annotations.transcriptome_embeds.t()) * logit_scale
    # logits_per_transcriptome = logits_per_text.T
    return pd.Series(logits_per_text.squeeze(0).cpu().detach())
