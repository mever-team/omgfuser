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

import einops
import numpy as np
import torch


from omgfuser import data_utils


class TestConvert2DMapToBinaryOneHot(unittest.TestCase):

    def test_batch_1(self) -> None:
        t: torch.Tensor = torch.from_numpy(np.array(
            [
                [.0, .0, .00001],
                [.1, .99, .0],
                [1., .0, 0.66]
            ]
        )).unsqueeze(dim=0).unsqueeze(dim=0)
        expected: torch.Tensor = torch.from_numpy(np.array(
            [
                [[1, 0], [1, 0], [1, 0]],
                [[1, 0], [0, 1], [1, 0]],
                [[0, 1], [1, 0], [0, 1]]
            ]
        )).unsqueeze(dim=0)
        expected = einops.rearrange(expected, "b h w c -> b c h w")

        out: torch.Tensor = data_utils.convert_2d_map_to_binary_one_hot(t)

        self.assertEqual(torch.Size([1, 2, 3, 3]), out.size())
        self.assertTrue(torch.all(out == expected))

    def test_batch_8(self) -> None:
        t: torch.Tensor = torch.randn((8, 1, 224, 224))
        out: torch.Tensor = data_utils.convert_2d_map_to_binary_one_hot(t)

        self.assertEqual(torch.Size([8, 2, 224, 224]), out.size())
        self.assertTrue(torch.sum(t > 0.5) == torch.sum(out[:, 1, :, :]))
