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

import numpy as np
import torch

from omgfuser.models import upscalers


class TestConvolutionalUpscaler(unittest.TestCase):

    def test_upscaler_batch_one(self) -> None:
        m: upscalers.ConvolutionalUpscaler = upscalers.ConvolutionalUpscaler(c_in=384, c_out=1)
        t: torch.Tensor = torch.randn((1, 384, 14, 14))
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 1, 224, 224)))
        self.assertTrue(np.all(out.detach().numpy() >= 0))
        self.assertTrue(np.all(out.detach().numpy() <= 1))

    def test_upscaler_batch_eight(self) -> None:
        m: upscalers.ConvolutionalUpscaler = upscalers.ConvolutionalUpscaler(c_in=384, c_out=1)
        t: torch.Tensor = torch.randn((8, 384, 14, 14))
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((8, 1, 224, 224)))
        self.assertTrue(np.all(out.detach().numpy() >= 0))
        self.assertTrue(np.all(out.detach().numpy() <= 1))

    def test_upscaler_batch_one_group_norm(self) -> None:
        m: upscalers.ConvolutionalUpscaler = upscalers.ConvolutionalUpscaler(
            c_in=384, c_out=1, group_norm=True)
        t: torch.Tensor = torch.randn((1, 384, 14, 14))
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 1, 224, 224)))
        self.assertTrue(np.all(out.detach().numpy() >= 0))
        self.assertTrue(np.all(out.detach().numpy() <= 1))

    def test_upscaler_batch_eight_group_norm(self) -> None:
        m: upscalers.ConvolutionalUpscaler = upscalers.ConvolutionalUpscaler(
            c_in=384, c_out=1, group_norm=True)
        t: torch.Tensor = torch.randn((8, 384, 14, 14))
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((8, 1, 224, 224)))
        self.assertTrue(np.all(out.detach().numpy() >= 0))
        self.assertTrue(np.all(out.detach().numpy() <= 1))

    def test_upscaler_batch_one_without_sigmoid(self) -> None:
        m: upscalers.ConvolutionalUpscaler = upscalers.ConvolutionalUpscaler(
            c_in=384, c_out=1, add_final_sigmoid_layer=False
        )
        t: torch.Tensor = torch.randn((1, 384, 14, 14))
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 1, 224, 224)))
        self.assertTrue(np.any(out.detach().numpy() > 1.))
        self.assertTrue(np.any(out.detach().numpy() < .0))

    def test_upscaler_batch_eight_without_sigmoid(self) -> None:
        m: upscalers.ConvolutionalUpscaler = upscalers.ConvolutionalUpscaler(
            c_in=384, c_out=1, add_final_sigmoid_layer=False
        )
        t: torch.Tensor = torch.randn((8, 384, 14, 14))
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((8, 1, 224, 224)))
        self.assertTrue(np.any(out.detach().numpy() > 1.))
        self.assertTrue(np.any(out.detach().numpy() < .0))


class TestDoubleConvolutionalUpscaler(unittest.TestCase):

    def test_upscaler_batch_one(self) -> None:
        m: upscalers.DoubleConvolutionalUpscaler = upscalers.DoubleConvolutionalUpscaler(
            c_in=384, c_out=1
        )
        t: torch.Tensor = torch.randn((1, 384, 14, 14))
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 1, 224, 224)))
        self.assertTrue(np.all(out.detach().numpy() >= 0))
        self.assertTrue(np.all(out.detach().numpy() <= 1))

    def test_upscaler_batch_eight(self) -> None:
        m: upscalers.DoubleConvolutionalUpscaler = upscalers.DoubleConvolutionalUpscaler(
            c_in=384, c_out=1
        )
        t: torch.Tensor = torch.randn((8, 384, 14, 14))
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((8, 1, 224, 224)))
        self.assertTrue(np.all(out.detach().numpy() >= 0))
        self.assertTrue(np.all(out.detach().numpy() <= 1))

    def test_upscaler_batch_one_group_norm(self) -> None:
        m: upscalers.DoubleConvolutionalUpscaler = upscalers.DoubleConvolutionalUpscaler(
            c_in=384, c_out=1, group_norm=True)
        t: torch.Tensor = torch.randn((1, 384, 14, 14))
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 1, 224, 224)))
        self.assertTrue(np.all(out.detach().numpy() >= 0))
        self.assertTrue(np.all(out.detach().numpy() <= 1))

    def test_upscaler_batch_eight_group_norm(self) -> None:
        m: upscalers.DoubleConvolutionalUpscaler = upscalers.DoubleConvolutionalUpscaler(
            c_in=384, c_out=1, group_norm=True)
        t: torch.Tensor = torch.randn((8, 384, 14, 14))
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((8, 1, 224, 224)))
        self.assertTrue(np.all(out.detach().numpy() >= 0))
        self.assertTrue(np.all(out.detach().numpy() <= 1))

    def test_upscaler_batch_one_without_sigmoid(self) -> None:
        m: upscalers.DoubleConvolutionalUpscaler = upscalers.DoubleConvolutionalUpscaler(
            c_in=384, c_out=1, add_final_sigmoid_layer=False
        )
        t: torch.Tensor = torch.randn((1, 384, 14, 14))
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 1, 224, 224)))
        self.assertTrue(np.any(out.detach().numpy() > 1.))
        self.assertTrue(np.any(out.detach().numpy() < .0))

    def test_upscaler_batch_eight_without_sigmoid(self) -> None:
        m: upscalers.DoubleConvolutionalUpscaler = upscalers.DoubleConvolutionalUpscaler(
            c_in=384, c_out=1, add_final_sigmoid_layer=False
        )
        t: torch.Tensor = torch.randn((8, 384, 14, 14))
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((8, 1, 224, 224)))
        self.assertTrue(np.any(out.detach().numpy() > 1.))
        self.assertTrue(np.any(out.detach().numpy() < .0))

    def test_upscaler_batch_one_softmax(self) -> None:
        m: upscalers.DoubleConvolutionalUpscaler = upscalers.DoubleConvolutionalUpscaler(
            c_in=384,
            c_out=2,
            add_final_sigmoid_layer=False,
            use_softmax_activation=True
        )
        t: torch.Tensor = torch.randn((1, 384, 14, 14))
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 2, 224, 224)))
        self.assertTrue(np.all(out.detach().numpy() >= 0))
        self.assertTrue(np.all(out.detach().numpy() <= 1))

    def test_upscaler_batch_eight_softmax(self) -> None:
        m: upscalers.DoubleConvolutionalUpscaler = upscalers.DoubleConvolutionalUpscaler(
            c_in=384,
            c_out=2,
            add_final_sigmoid_layer=False,
            use_softmax_activation=True
        )
        t: torch.Tensor = torch.randn((8, 384, 14, 14))
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((8, 2, 224, 224)))
        self.assertTrue(np.all(out.detach().numpy() >= 0))
        self.assertTrue(np.all(out.detach().numpy() <= 1))


class TestResidualUpscaler(unittest.TestCase):

    def test_upscaler_batch_one(self) -> None:
        m: upscalers.ResidualUpscaler = upscalers.ResidualUpscaler(c_in=384, c_out=1)
        t: torch.Tensor = torch.randn((1, 384, 14, 14))
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 1, 224, 224)))
        self.assertTrue(np.all(out.detach().numpy() >= 0))
        self.assertTrue(np.all(out.detach().numpy() <= 1))

    def test_upscaler_batch_eight(self) -> None:
        m: upscalers.ResidualUpscaler = upscalers.ResidualUpscaler(c_in=384, c_out=1)
        t: torch.Tensor = torch.randn((8, 384, 14, 14))
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((8, 1, 224, 224)))
        self.assertTrue(np.all(out.detach().numpy() >= 0))
        self.assertTrue(np.all(out.detach().numpy() <= 1))

    def test_upscaler_batch_one_without_sigmoid(self) -> None:
        m: upscalers.ResidualUpscaler = upscalers.ResidualUpscaler(
            c_in=384, c_out=1, add_final_sigmoid_layer=False
        )
        t: torch.Tensor = torch.randn((1, 384, 14, 14))
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 1, 224, 224)))
        self.assertTrue(np.any(out.detach().numpy() > 1.))
        self.assertTrue(np.any(out.detach().numpy() < .0))

    def test_upscaler_batch_eight_without_sigmoid(self) -> None:
        m: upscalers.ResidualUpscaler = upscalers.ResidualUpscaler(
            c_in=384, c_out=1, add_final_sigmoid_layer=False
        )
        t: torch.Tensor = torch.randn((8, 384, 14, 14))
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((8, 1, 224, 224)))
        self.assertTrue(np.any(out.detach().numpy() > 1.))
        self.assertTrue(np.any(out.detach().numpy() < .0))
