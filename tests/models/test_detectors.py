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

import torch

from omgfuser.models import detectors


class TestTransformerDetector(unittest.TestCase):

    def test_softmax_head(self) -> None:
        batch_sizes: list[int] = [1, 4, 8]

        for b in batch_sizes:
            m: detectors.TransformerDetector = detectors.TransformerDetector(
                layers=6,
                dim=384,
                out_dim=2,
                heads=12,
                dim_head=384//12,
                mlp_dim=384*4,
                classification_hidden_dim=384*4,
            )
            t: torch.Tensor = torch.randn((b, 14*14, 384))

            o: torch.Tensor = m(t)

            self.assertEqual(torch.Size((b, 2)), o.size())
            self.assertTrue(torch.all(o >= .0))
            self.assertTrue(torch.all(o <= 1.))
            self.assertTrue(torch.all(torch.abs(torch.sum(o, dim=1)-1.0) < 1e-4))

    def test_sigmoid_head_two_outputs(self) -> None:
        batch_sizes: list[int] = [1, 4, 8]

        for b in batch_sizes:
            m: detectors.TransformerDetector = detectors.TransformerDetector(
                layers=6,
                dim=384,
                out_dim=2,
                heads=12,
                dim_head=384//12,
                mlp_dim=384*4,
                classification_hidden_dim=384*4,
                last_layer_activation_type=detectors.ActivationType.SIGMOID
            )
            t: torch.Tensor = torch.randn((b, 14*14, 384))

            o: torch.Tensor = m(t)

            self.assertEqual(torch.Size((b, 2)), o.size())
            self.assertTrue(torch.all(o >= .0))
            self.assertTrue(torch.all(o <= 1.))

    def test_sigmoid_head_one_output(self) -> None:
        batch_sizes: list[int] = [1, 4, 8]

        for b in batch_sizes:
            m: detectors.TransformerDetector = detectors.TransformerDetector(
                layers=6,
                dim=384,
                out_dim=1,
                heads=12,
                dim_head=384 // 12,
                mlp_dim=384 * 4,
                classification_hidden_dim=384 * 4,
                last_layer_activation_type=detectors.ActivationType.SIGMOID
            )
            t: torch.Tensor = torch.randn((b, 14 * 14, 384))

            o: torch.Tensor = m(t)

            self.assertEqual(torch.Size((b, 1)), o.size())
            self.assertTrue(torch.all(o >= .0))
            self.assertTrue(torch.all(o <= 1.))

    def test_no_last_layer_activation(self) -> None:
        batch_sizes: list[int] = [1, 4, 8]

        for b in batch_sizes:
            m: detectors.TransformerDetector = detectors.TransformerDetector(
                layers=6,
                dim=384,
                out_dim=2,
                heads=12,
                dim_head=384//12,
                mlp_dim=384*4,
                classification_hidden_dim=384*4,
                last_layer_activation_type=None
            )
            t: torch.Tensor = torch.randn((b, 14*14, 384))

            o: torch.Tensor = m(t)

            self.assertEqual(torch.Size((b, 2)), o.size())

    def test_head_with_single_layer_mlp(self) -> None:
        batch_sizes: list[int] = [1, 4, 8]
        out_dims: list[int] = [1, 2]

        for b in batch_sizes:
            for o_dim in out_dims:
                m: detectors.TransformerDetector = detectors.TransformerDetector(
                    layers=4,
                    dim=384,
                    out_dim=o_dim,
                    heads=12,
                    dim_head=384//12,
                    mlp_dim=384*4,
                    classification_hidden_dim=0,
                    last_layer_activation_type=None
                )
                t: torch.Tensor = torch.randn((b, 14 * 14, 384))

                o: torch.Tensor = m(t)

                self.assertEqual(torch.Size((b, o_dim)), o.size())
