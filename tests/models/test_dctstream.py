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

import argparse
import pathlib
import unittest

import torch

from omgfuser.models import dctstream


class TestDCTStream(unittest.TestCase):

    def test_dct_stream(self) -> None:
        image_sizes: list[tuple[int, int]] = [
            (768, 512),
            (512, 768),
            (1536, 1024),
            (1536, 1536)
        ]
        batch_sizes: list[int] = [1, 3]

        current_dir: pathlib.Path = pathlib.Path(__file__).parent
        pretrained_weights: pathlib.Path = (
            current_dir.parent.parent / "checkpoints/DCT_djpeg.pth.tar"
        )
        dct_model: dctstream.DCTStream = dctstream.DCTStream()
        dct_model.init_weights(str(pretrained_weights))

        for img_sz in image_sizes:
            for bs in batch_sizes:
                dct_vol: torch.Tensor = torch.randint(0, 1, (bs, 21, img_sz[0], img_sz[1])).float()
                qtable: torch.Tensor = torch.randint(0, 255, (bs, 1, 8, 8)).float()

                dct_out: torch.Tensor = dct_model(dct_vol, qtable)

                self.assertEqual(torch.Size([bs, 672, img_sz[0]//8, img_sz[1]//8]), dct_out.size())

    def test_dct_stream_with_final_layers(self) -> None:
        image_sizes: list[tuple[int, int]] = [
            (768, 512),
            (512, 768),
            (1536, 1024),
            (1536, 1536)
        ]
        batch_sizes: list[int] = [1, 3]

        current_dir: pathlib.Path = pathlib.Path(__file__).parent
        pretrained_weights: pathlib.Path = (
                current_dir.parent.parent / "checkpoints/DCT_djpeg.pth.tar"
        )
        dct_model: dctstream.DCTStream = dctstream.DCTStream(include_final_layers=True)
        dct_model.init_weights(str(pretrained_weights))

        for img_sz in image_sizes:
            for bs in batch_sizes:
                dct_vol: torch.Tensor = torch.randint(0, 1, (bs, 21, img_sz[0], img_sz[1])).float()
                qtable: torch.Tensor = torch.randint(0, 255, (bs, 1, 8, 8)).float()

                dct_out: torch.Tensor = dct_model(dct_vol, qtable)

                self.assertEqual(torch.Size([bs, 2, img_sz[0] // 8, img_sz[1] // 8]),
                                 dct_out.size())
