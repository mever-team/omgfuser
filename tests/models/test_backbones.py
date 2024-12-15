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

from omgfuser.models import backbones


class TestConvolutionalBackbone(unittest.TestCase):

    def test_with_single_channel_tensor(self) -> None:
        m: backbones.ConvolutionalBackbone = backbones.ConvolutionalBackbone(1)
        t: torch.Tensor = torch.rand(1, 1, 224, 224)
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 384, 14, 14)))

    def test_with_three_channel_tensor(self) -> None:
        m: backbones.ConvolutionalBackbone = backbones.ConvolutionalBackbone(3)
        t: torch.Tensor = torch.rand(1, 3, 224, 224)
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 384, 14, 14)))

    def test_with_single_channel_tensor_group_norm(self) -> None:
        m: backbones.ConvolutionalBackbone = backbones.ConvolutionalBackbone(1, group_norm=True)
        t: torch.Tensor = torch.rand(1, 1, 224, 224)
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 384, 14, 14)))

    def test_with_three_channel_tensor_group_norm(self) -> None:
        m: backbones.ConvolutionalBackbone = backbones.ConvolutionalBackbone(3, group_norm=True)
        t: torch.Tensor = torch.rand(1, 3, 224, 224)
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 384, 14, 14)))


class TestDINOBackbone(unittest.TestCase):

    def test_feature_extraction(self) -> None:
        batch_sizes: list[int] = [1, 4, 8]

        m: backbones.DINOBackbone = backbones.DINOBackbone()

        for b in batch_sizes:
            t: torch.Tensor = torch.rand(b, 3, 224, 224)
            o: torch.Tensor = m(t)
            self.assertEqual(torch.Size((b, 196, 384)), o.size())


class TestDINOv2Backbone(unittest.TestCase):

    def test_feature_extraction(self) -> None:
        batch_sizes: list[int] = [1, 4, 8]

        m: backbones.DINOv2Backbone = backbones.DINOv2Backbone()

        for b in batch_sizes:
            t: torch.Tensor = torch.rand(b, 3, 224, 224)
            o: torch.Tensor = m(t)
            self.assertEqual(torch.Size((b, 196, 384)), o.size())

    def test_frozen_model(self) -> None:
        batch_sizes: list[int] = [1, 4, 8]

        m: backbones.DINOv2Backbone = backbones.DINOv2Backbone(frozen=True)

        for b in batch_sizes:
            t: torch.Tensor = torch.rand(b, 3, 224, 224)
            o: torch.Tensor = m(t)
            self.assertEqual(torch.Size((b, 196, 384)), o.size())

    def test_patch_embed_frozen_model(self) -> None:
        batch_sizes: list[int] = [1, 4, 8]

        m: backbones.DINOv2Backbone = backbones.DINOv2Backbone(
            frozen=True, freeze_only_patch_embed=True
        )

        for b in batch_sizes:
            t: torch.Tensor = torch.rand(b, 3, 224, 224)
            o: torch.Tensor = m(t)
            self.assertEqual(torch.Size((b, 196, 384)), o.size())

    def test_feature_extraction_from_different_layers(self) -> None:
        batch_sizes: list[int] = [1, 4, 8]
        layer_nums: list[int] = [1, 2, 3]

        for ln in layer_nums:
            m: backbones.DINOv2Backbone = backbones.DINOv2Backbone(features_layer=ln)

            for b in batch_sizes:
                t: torch.Tensor = torch.rand(b, 3, 224, 224)
                o: torch.Tensor = m(t)
                self.assertEqual(torch.Size((b, 196, 384)), o.size())

    def test_features_interpolation(self) -> None:
        interpolation_type: list[backbones.DINOv2FeaturesInterpolationType] = [
            backbones.DINOv2FeaturesInterpolationType.CONVOLUTIONAL,
            backbones.DINOv2FeaturesInterpolationType.BILINEAR
        ]
        batch_sizes: list[int] = [1, 4, 8]

        for it in interpolation_type:
            m: backbones.DINOv2Backbone = backbones.DINOv2Backbone(
                features_interpolation=True,
                features_interpolation_type=it
            )

            for b in batch_sizes:
                t: torch.Tensor = torch.rand(b, 3, 224, 224)
                o: torch.Tensor = m(t)
                self.assertEqual(torch.Size((b, 196, 384)), o.size())

    def test_extraction_of_multiscale_features(self) -> None:
        batch_sizes: list[int] = [1, 4, 8]
        features_interpolation: list[bool] = [True, False]
        features_layer: list[int] = [2, 5, 8, 11]

        for fi in features_interpolation:
            m: backbones.DINOv2Backbone = backbones.DINOv2Backbone(features_interpolation=fi,
                                                                   features_layer=features_layer)

            for b in batch_sizes:
                t: torch.Tensor = torch.rand(b, 3, 224, 224)
                o: list[torch.Tensor] = m(t)
                self.assertEqual(len(features_layer), len(o))
                for f in o:
                    self.assertEqual(torch.Size((b, 196, 384)), f.size())
