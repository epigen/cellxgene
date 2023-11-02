import unittest

import numpy as np

from server.common.compute import llm_embeddings
from server.data_common.matrix_loader import MatrixDataLoader
from test.unit import app_config
from test import PROJECT_ROOT


class LLMEmbsTest(unittest.TestCase):
    """Tests the llmembs returns the expected results for one test case, using the h5ad
    adaptor types and different algorithms."""

    def load_dataset(self, path, extra_server_config={}, extra_dataset_config={}):
        config = app_config(path, extra_server_config=extra_server_config, extra_dataset_config=extra_dataset_config)
        loader = MatrixDataLoader(path)
        adaptor = loader.open(config)
        return adaptor

    def get_mask(self, adaptor, start, stride):
        """Simple function to return a mask or rows"""
        rows = adaptor.get_shape()[0]
        sel = list(range(start, rows, stride))
        mask = np.zeros(rows, dtype=bool)
        mask[sel] = True
        return mask

    def compare_llmembs_results(self, results, expects):
        self.assertEqual(len(results), len(expects))
        for result, expect in zip(results, expects):
            self.assertEqual(result[0], expect[0])
            self.assertTrue(np.isclose(result[1], expect[1], 1e-6, 1e-4))
            self.assertTrue(np.isclose(result[2], expect[2], 1e-6, 1e-4))
            self.assertTrue(np.isclose(result[3], expect[3], 1e-6, 1e-4))

    def check_results(self, results):
        """Checks the results for a specific set of rows selections"""

        self.assertIn("text", results)
        self.assertIsInstance(results["text"], str)
        # expects = []

        # self.compare_llmembs_results(results, expects)

    def test_anndata_default(self):
        """Test an anndata adaptor with its default llmembs algorithm (llmembs_generic)"""
        adaptor = self.load_dataset(f"{PROJECT_ROOT}/example-dataset/pbmc3k.h5ad")
        mask = self.get_mask(adaptor, 1, 10)
        results = adaptor.compute_llmembs_obs_to_text(mask)
        self.check_results(results)


def test_h5ad_default(self):
    """Test a h5ad adaptor with its default llmembs algorithm (llmembs_cxg)"""
    adaptor = self.load_dataset(f"{PROJECT_ROOT}/example-dataset/pbmc3k.h5ad")
    mask = self.get_mask(adaptor, 1, 10)

    # run it through the adaptor
    results = adaptor.compute_llmembs_obs_to_text(mask)
    self.check_results(results)

    # run it directly
    results = llm_embeddings.llm_obs_to_text(adaptor, mask)
    self.check_results(results)


# TODO test also compute_llmembs_text_to_annotations
