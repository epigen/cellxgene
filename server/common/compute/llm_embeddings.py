import numpy as np
from scipy import sparse, stats
import pandas as pd

from server.common.constants import XApproximateDistribution


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
    # TODO call LLM etc.
    return pd.Series(np.random.randn(len(adaptor.data.obs)))
