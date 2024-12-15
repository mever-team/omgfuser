"""
Created by Dimitrios Karageorgiou, email: dkarageo@iti.gr

Originally distributed under: https://github.com/mever-team/omgfuser

Copyright 2024 Media Analysis, Verification and Retrieval Group -
Information Technologies Institute - Centre for Research and Technology Hellas, Greece

This piece of code is licensed under the Apache License, Version 2.0.
A copy of the license can be found in the LICENSE file distributed together
with this file, as well as under https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under this repository is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the license for the specific language governing permissions and
limitations under the License.
"""

import unittest
from typing import Union

import numpy as np
import torch

from omgfuser import metrics


class TestMetrics(unittest.TestCase):

    def test_metrics_multidimensional_arrays(self) -> None:
        m: metrics.Metrics = metrics.Metrics()

        # Create tensors that resemble 2D outputs with batch size 8 and 6.
        predicted_1: torch.Tensor = torch.randn(8, 224, 224)
        actual_1: torch.Tensor = torch.randint(2, (8, 224, 224))
        predicted_2: torch.Tensor = torch.ones(6, 224, 224)
        actual_2: torch.Tensor = torch.ones(6, 224, 224)

        # Update the state of the metrics.
        results_1: dict[str, Union[np.ndarray, float]] = m.update(predicted_1, actual_1)
        results_2: dict[str, Union[np.ndarray, float]] = m.update(predicted_2, actual_2)

        # Compute the metrics over all the data.
        results: dict[str, Union[np.ndarray, float]] = m.compute()

        # Check that all expected metrics are included in intermediate and final results.
        for r in [results_1, results_2, results]:
            self.assertIn("auc", r)
            self.assertIn("iou", r)
            self.assertIn("precision", r)
            self.assertIn("recall", r)
            self.assertIn("f1", r)
            self.assertIn("accuracy", r)

        self.assertEqual(tuple(), results["auc"].shape)
        self.assertEqual(tuple(), results["accuracy"].shape)
        self.assertEqual((2,), results["iou"].shape)
        self.assertEqual((2,), results["precision"].shape)
        self.assertEqual((2,), results["recall"].shape)
        self.assertEqual((2,), results["f1"].shape)

    def test_metrics_multidimensional_arrays_perfect_match(self) -> None:
        m: metrics.Metrics = metrics.Metrics()

        # Create tensors that resemble 2D outputs with batch size 8 and 6.
        actual_1: torch.Tensor = torch.randint(2, (8, 1, 224, 224))
        actual_2: torch.Tensor = torch.randint(2, (6, 1, 224, 224))

        # Update the state of the metrics.
        results_1: dict[str, np.ndarray] = m.update(actual_1.float(), actual_1)
        results_2: dict[str, np.ndarray] = m.update(actual_2.float(), actual_2)

        # Compute the metrics over all the data.
        results: dict[str, np.ndarray] = m.compute()

        for r in [results_1, results_2, results]:
            self.assertAlmostEqual(float(r["auc"]), 1.0)
            self.assertAlmostEqual(float(r["accuracy"]), 1.0)
            for c in [0, 1]:
                self.assertAlmostEqual(float(r["iou"][c]), 1.0)
                self.assertAlmostEqual(float(r["precision"][c]), 1.0)
                self.assertAlmostEqual(float(r["recall"][c]), 1.0)
                self.assertAlmostEqual(float(r["f1"][c]), 1.0)

    def test_metrics_multidimensional_arrays_perfect_match_above_threshold(self) -> None:
        m: metrics.Metrics = metrics.Metrics()

        # Create tensors that resemble 2D outputs with batch size 8 and 6.
        actual_1: torch.Tensor = torch.randint(2, (8, 1, 224, 224))
        actual_2: torch.Tensor = torch.randint(2, (6, 1, 224, 224))

        # Update the state of the metrics.
        results_1: dict[str, np.ndarray] = m.update(
            torch.clamp(actual_1.float(), min=.0, max=0.65), actual_1)
        results_2: dict[str, np.ndarray] = m.update(
            torch.clamp(actual_2.float(), min=.0, max=0.65), actual_2)

        # Compute the metrics over all the data.
        results: dict[str, np.ndarray] = m.compute()

        for r in [results_1, results_2, results]:
            self.assertAlmostEqual(float(r["auc"]), 1.0)
            self.assertAlmostEqual(float(r["accuracy"]), 1.0)
            for c in [0, 1]:
                self.assertAlmostEqual(float(r["iou"][c]), 1.0)
                self.assertAlmostEqual(float(r["precision"][c]), 1.0)
                self.assertAlmostEqual(float(r["recall"][c]), 1.0)
                self.assertAlmostEqual(float(r["f1"][c]), 1.0)

    def test_metrics_multidimensional_arrays_worst_match(self) -> None:
        m: metrics.Metrics = metrics.Metrics()

        # Create tensors that resemble 2D outputs with batch size 8 and 6.
        actual_1: torch.Tensor = torch.randint(2, (8, 1, 224, 224))
        actual_2: torch.Tensor = torch.randint(2, (6, 1, 224, 224))

        # Update the state of the metrics.
        results_1: dict[str, np.ndarray] = m.update(1-actual_1.float(), actual_1)
        results_2: dict[str, np.ndarray] = m.update(1-actual_2.float(), actual_2)

        # Compute the metrics over all the data.
        results: dict[str, np.ndarray] = m.compute()

        for r in [results_1, results_2, results]:
            self.assertAlmostEqual(float(r["auc"]), .0)
            self.assertAlmostEqual(float(r["accuracy"]), .0)
            for c in [0, 1]:
                self.assertAlmostEqual(float(r["iou"][c]), .0)
                self.assertAlmostEqual(float(r["precision"][c]), .0)
                self.assertAlmostEqual(float(r["recall"][c]), .0)
                self.assertTrue(np.isnan(r["f1"][c]))

    # def test_metrics_multidimensional_arrays_worst_match_below_threshold(self) -> None:
    #     m: metrics.Metrics = metrics.Metrics()
    #
    #     # Create tensors that resemble 2D outputs with batch size 8 and 6.
    #     actual_1: torch.Tensor = torch.zeros((8, 1, 224, 224))
    #     actual_2: torch.Tensor = torch.ones((8, 1, 224, 224))
    #
    #     # Update the state of the metrics.
    #     results_1: dict[str, np.ndarray] = m.update(
    #         torch.clamp(actual_1.float(), min=0.0, max=0.45), actual_1)
    #     results_2: dict[str, np.ndarray] = m.update(
    #         torch.clamp(actual_2.float(), min=0.0, max=0.45), actual_2)
    #
    #     # Compute the metrics over all the data.
    #     results: dict[str, np.ndarray] = m.compute()
    #
    #     for r in [results_1, results_2, results]:
    #         self.assertAlmostEqual(float(r["auc"]), 1.0)
    #         self.assertAlmostEqual(float(r["accuracy"]), .0)
    #         for c in [0, 1]:
    #             self.assertAlmostEqual(float(r["iou"][c]), .0)
    #             self.assertAlmostEqual(float(r["precision"][c]), .0)
    #             self.assertAlmostEqual(float(r["recall"][c]), .0)
    #             self.assertTrue(np.isnan(r["f1"][c]))

    def test_metrics_on_vector_arrays(self) -> None:
        m: metrics.Metrics = metrics.Metrics()

        # Create tensors that resemble 1D outputs with batch size 8 and 6.
        predicted_1: torch.Tensor = torch.zeros(8, 2)
        actual_1: torch.Tensor = torch.zeros(8, 2)
        predicted_2: torch.Tensor = torch.ones(6, 2)
        actual_2: torch.Tensor = torch.zeros(6, 2)

        # Update the state of the metrics.
        results_1: dict[str, Union[np.ndarray, float]] = m.update(predicted_1, actual_1)
        results_2: dict[str, Union[np.ndarray, float]] = m.update(predicted_2, actual_2)

        # Compute the metrics over all the data.
        results: dict[str, Union[np.ndarray, float]] = m.compute()

        # Check that all expected metrics are included in intermediate and final results.
        for r in [results_1, results_2, results]:
            self.assertIn("auc", r)
            self.assertIn("iou", r)
            self.assertIn("precision", r)
            self.assertIn("recall", r)
            self.assertIn("f1", r)
            self.assertIn("accuracy", r)

        self.assertEqual(tuple(), results["auc"].shape)
        self.assertEqual(tuple(), results["accuracy"].shape)
        self.assertEqual((2,), results["iou"].shape)
        self.assertEqual((2,), results["precision"].shape)
        self.assertEqual((2,), results["recall"].shape)
        self.assertEqual((2,), results["f1"].shape)
