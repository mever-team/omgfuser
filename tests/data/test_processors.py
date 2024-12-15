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

import pathlib
import unittest
from typing import Any

import torch

from omgfuser.data import processors


parent_dir: pathlib.Path = pathlib.Path(__file__).parent


class TestNoiseprintProcessor(unittest.TestCase):

    def test_preprocess_with_padding(self) -> None:
        batch_sizes: list[int] = [1, 3, 6]

        processor = processors.NoiseprintProcessor(
            pretrained_weights=parent_dir / "test_data/noiseprintplusplus.pth"
        )

        for b in batch_sizes:
            batch_data: dict[str, Any] = {
                "image": torch.randn((b, 3, 224, 224)),
                "original_image": [torch.randn(3, 448, 896) for _ in range(b)],
                "unpadded_size": (
                    torch.Tensor([112 for _ in range(b)]), torch.Tensor([224 for _ in range(b)]))
            }

            out_b: dict[str, Any] = processor.preprocess(batch_data)

            self.assertIn("input", out_b)
            self.assertEqual(torch.Size((b, 4, 224, 224)), out_b["input"].size())


class TestDCTProcessor(unittest.TestCase):

    def test_preprocess_with_padding(self) -> None:
        batch_sizes: list[int] = [1, 3, 6]

        processor = processors.DCTProcessor(
            pretrained_weights=parent_dir.parent.parent / "checkpoints/DCT_djpeg.pth.tar"
        )

        for b in batch_sizes:
            batch_data: dict[str, Any] = {
                "image": torch.randn((b, 3, 224, 224)),
                "original_image": [torch.randn(3, 448, 896) for _ in range(b)],
                "unpadded_size": (
                    torch.Tensor([112 for _ in range(b)]), torch.Tensor([224 for _ in range(b)])),
                "dct_vol": [torch.randint(0, 1, (21, 448, 896)).float() for _ in range(b)],
                "qtable": torch.randint(0, 255, (b, 1, 8, 8)).float()
            }

            out_b: dict[str, Any] = processor.preprocess(batch_data)

            self.assertIn("input", out_b)
            self.assertEqual(torch.Size((b, 3+672, 224, 224)), out_b["input"].size())


class TestCombinedProcessor(unittest.TestCase):

    def test_preprocess_with_padding(self) -> None:
        batch_sizes: list[int] = [1, 3, 6]

        processor = processors.CombinedProcessor([
            processors.NoiseprintProcessor(
                pretrained_weights=parent_dir / "test_data/noiseprintplusplus.pth"
            ),
            processors.DCTProcessor(
                pretrained_weights=parent_dir.parent.parent / "checkpoints/DCT_djpeg.pth.tar"
            )
        ])

        for b in batch_sizes:
            batch_data: dict[str, Any] = {
                "image": torch.randn((b, 3, 224, 224)),
                "original_image": [torch.randn(3, 448, 896) for _ in range(b)],
                "unpadded_size": (
                    torch.Tensor([112 for _ in range(b)]), torch.Tensor([224 for _ in range(b)])),
                "dct_vol": [torch.randint(0, 1, (21, 448, 896)).float() for _ in range(b)],
                "qtable": torch.randint(0, 255, (b, 1, 8, 8)).float()
            }

            out_b: dict[str, Any] = processor.preprocess(batch_data)

            self.assertIn("input", out_b)
            self.assertEqual(torch.Size((b, 3+1+672, 224, 224)), out_b["input"].size())
