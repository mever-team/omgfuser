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

from omgfuser.models import fusion


class TestFusionModel(unittest.TestCase):

    def test_with_three_inputs_batch_one(self) -> None:
        m: fusion.FusionModel = fusion.FusionModel(inputs_num=3,
                                                   first_stage_depth=8,
                                                   first_stage_heads=8,
                                                   second_stage_depth=8,
                                                   second_stage_heads=8,
                                                   height=224,
                                                   width=224)
        t: torch.Tensor = torch.rand((1, 12, 224, 224))
        out: dict[str, torch.Tensor] = m(t)

        # Check that output contains both localization and detection tensors.
        self.assertIn("localization", out)
        self.assertIn("detection", out)

        # Validate localization map.
        localization_out: torch.Tensor = out["localization"].detach().cpu()
        self.assertEqual(torch.Size((1, 1, 224, 224)), localization_out.size())
        # Localization map should have values in [0, 1].
        self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

        # Validate detection output.
        detection_out: torch.Tensor = out["detection"].detach().cpu()
        self.assertEqual(torch.Size((1, 2)), detection_out.size())
        self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_three_inputs_batch_eight(self) -> None:
        m: fusion.FusionModel = fusion.FusionModel(inputs_num=3,
                                                   first_stage_depth=8,
                                                   first_stage_heads=8,
                                                   second_stage_depth=8,
                                                   second_stage_heads=8,
                                                   height=224,
                                                   width=224)
        t: torch.Tensor = torch.rand((8, 12, 224, 224))
        out: dict[str, torch.Tensor] = m(t)

        self.assertIn("localization", out)
        self.assertIn("detection", out)

        # Validate localization map.
        localization_out: torch.Tensor = out["localization"].detach().cpu()
        self.assertEqual(torch.Size((8, 1, 224, 224)), localization_out.size())
        # Localization map should have values in [0, 1].
        self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

        # Validate detection output.
        detection_out: torch.Tensor = out["detection"].detach().cpu()
        self.assertEqual(torch.Size((8, 2)), detection_out.size())
        self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)


class TestFusionModelWithImage(unittest.TestCase):

    def test_with_three_inputs_batch_one(self) -> None:
        m: fusion.FusionModelWithImage = fusion.FusionModelWithImage(inputs_num=3,
                                                                     first_stage_depth=8,
                                                                     first_stage_heads=8,
                                                                     second_stage_depth=8,
                                                                     second_stage_heads=8,
                                                                     height=224,
                                                                     width=224)
        t: torch.Tensor = torch.rand((1, 9, 224, 224))
        out: dict[str, torch.Tensor] = m(t)

        # Check that output contains both localization and detection tensors.
        self.assertIn("localization", out)
        self.assertIn("detection", out)

        # Validate localization map.
        localization_out: torch.Tensor = out["localization"].detach().cpu()
        self.assertEqual(torch.Size((1, 1, 224, 224)), localization_out.size())
        # Localization map should have values in [0, 1].
        self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

        # Validate detection output.
        detection_out: torch.Tensor = out["detection"].detach().cpu()
        self.assertEqual(torch.Size((1, 2)), detection_out.size())
        self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_three_inputs_batch_eight(self) -> None:
        m: fusion.FusionModelWithImage = fusion.FusionModelWithImage(inputs_num=3,
                                                                     first_stage_depth=8,
                                                                     first_stage_heads=8,
                                                                     second_stage_depth=8,
                                                                     second_stage_heads=8,
                                                                     height=224,
                                                                     width=224)
        t: torch.Tensor = torch.rand((8, 9, 224, 224))
        out: dict[str, torch.Tensor] = m(t)

        self.assertIn("localization", out)
        self.assertIn("detection", out)

        # Validate localization map.
        localization_out: torch.Tensor = out["localization"].detach().cpu()
        self.assertEqual(torch.Size((8, 1, 224, 224)), localization_out.size())
        # Localization map should have values in [0, 1].
        self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

        # Validate detection output.
        detection_out: torch.Tensor = out["detection"].detach().cpu()
        self.assertEqual(torch.Size((8, 2)), detection_out.size())
        self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_three_inputs_batch_one_single_feature_extractors(self) -> None:
        m: fusion.FusionModelWithImage = fusion.FusionModelWithImage(
            inputs_num=3,
            first_stage_depth=8,
            first_stage_heads=8,
            second_stage_depth=8,
            second_stage_heads=8,
            height=224,
            width=224,
            single_image_feature_extractor=True,
            single_segmentation_map_feature_extractor=True
        )
        t: torch.Tensor = torch.rand((1, 9, 224, 224))
        out: dict[str, torch.Tensor] = m(t)

        # Check that output contains both localization and detection tensors.
        self.assertIn("localization", out)
        self.assertIn("detection", out)

        # Validate localization map.
        localization_out: torch.Tensor = out["localization"].detach().cpu()
        self.assertEqual(torch.Size((1, 1, 224, 224)), localization_out.size())
        # Localization map should have values in [0, 1].
        self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

        # Validate detection output.
        detection_out: torch.Tensor = out["detection"].detach().cpu()
        self.assertEqual(torch.Size((1, 2)), detection_out.size())
        self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_three_inputs_batch_eight_single_feature_extractors(self) -> None:
        m: fusion.FusionModelWithImage = fusion.FusionModelWithImage(
            inputs_num=3,
            first_stage_depth=8,
            first_stage_heads=8,
            second_stage_depth=8,
            second_stage_heads=8,
            height=224,
            width=224,
            single_segmentation_map_feature_extractor=True,
            single_image_feature_extractor=True
        )
        t: torch.Tensor = torch.rand((8, 9, 224, 224))
        out: dict[str, torch.Tensor] = m(t)

        self.assertIn("localization", out)
        self.assertIn("detection", out)

        # Validate localization map.
        localization_out: torch.Tensor = out["localization"].detach().cpu()
        self.assertEqual(torch.Size((8, 1, 224, 224)), localization_out.size())
        # Localization map should have values in [0, 1].
        self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

        # Validate detection output.
        detection_out: torch.Tensor = out["detection"].detach().cpu()
        self.assertEqual(torch.Size((8, 2)), detection_out.size())
        self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_three_inputs_batch_one_residual_upscaler(self) -> None:
        m: fusion.FusionModelWithImage = fusion.FusionModelWithImage(
            inputs_num=3,
            first_stage_depth=8,
            first_stage_heads=8,
            second_stage_depth=8,
            second_stage_heads=8,
            height=224,
            width=224,
            upscaler_type=fusion.UpscalerType.RESIDUAL_UPSCALER
        )
        t: torch.Tensor = torch.rand((1, 9, 224, 224))
        out: dict[str, torch.Tensor] = m(t)

        # Check that output contains both localization and detection tensors.
        self.assertIn("localization", out)
        self.assertIn("detection", out)

        # Validate localization map.
        localization_out: torch.Tensor = out["localization"].detach().cpu()
        self.assertEqual(torch.Size((1, 1, 224, 224)), localization_out.size())
        # Localization map should have values in [0, 1].
        self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

        # Validate detection output.
        detection_out: torch.Tensor = out["detection"].detach().cpu()
        self.assertEqual(torch.Size((1, 2)), detection_out.size())
        self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_three_inputs_batch_eight_residual_upscaler(self) -> None:
        m: fusion.FusionModelWithImage = fusion.FusionModelWithImage(
            inputs_num=3,
            first_stage_depth=8,
            first_stage_heads=8,
            second_stage_depth=8,
            second_stage_heads=8,
            height=224,
            width=224,
            upscaler_type=fusion.UpscalerType.RESIDUAL_UPSCALER
        )
        t: torch.Tensor = torch.rand((8, 9, 224, 224))
        out: dict[str, torch.Tensor] = m(t)

        self.assertIn("localization", out)
        self.assertIn("detection", out)

        # Validate localization map.
        localization_out: torch.Tensor = out["localization"].detach().cpu()
        self.assertEqual(torch.Size((8, 1, 224, 224)), localization_out.size())
        # Localization map should have values in [0, 1].
        self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

        # Validate detection output.
        detection_out: torch.Tensor = out["detection"].detach().cpu()
        self.assertEqual(torch.Size((8, 2)), detection_out.size())
        self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_three_inputs_batch_one_without_logits_layer(self) -> None:
        m: fusion.FusionModelWithImage = fusion.FusionModelWithImage(inputs_num=3,
                                                                     first_stage_depth=8,
                                                                     first_stage_heads=8,
                                                                     second_stage_depth=8,
                                                                     second_stage_heads=8,
                                                                     height=224,
                                                                     width=224,
                                                                     add_final_logit_layers=False)
        t: torch.Tensor = torch.rand((1, 9, 224, 224))
        out: dict[str, torch.Tensor] = m(t)

        # Check that output contains both localization and detection tensors.
        self.assertIn("localization", out)
        self.assertIn("detection", out)

        # Validate localization map.
        localization_out: torch.Tensor = out["localization"].detach().cpu()
        self.assertEqual(torch.Size((1, 1, 224, 224)), localization_out.size())
        # Localization map without logits should not have values in [0, 1].
        self.assertGreaterEqual(float(np.max(localization_out.numpy())), 1.0)
        self.assertLessEqual(float(np.min(localization_out.numpy())), 0.0)

        # Validate detection output.
        detection_out: torch.Tensor = out["detection"].detach().cpu()
        self.assertEqual(torch.Size((1, 2)), detection_out.size())

    def test_with_three_inputs_batch_eight_without_logits_layer(self) -> None:
        m: fusion.FusionModelWithImage = fusion.FusionModelWithImage(inputs_num=3,
                                                                     first_stage_depth=8,
                                                                     first_stage_heads=8,
                                                                     second_stage_depth=8,
                                                                     second_stage_heads=8,
                                                                     height=224,
                                                                     width=224,
                                                                     add_final_logit_layers=False)
        t: torch.Tensor = torch.rand((8, 9, 224, 224))
        out: dict[str, torch.Tensor] = m(t)

        self.assertIn("localization", out)
        self.assertIn("detection", out)

        # Validate localization map.
        localization_out: torch.Tensor = out["localization"].detach().cpu()
        self.assertEqual(torch.Size((8, 1, 224, 224)), localization_out.size())
        # Localization map without logits should not have values in [0, 1].
        self.assertGreaterEqual(float(np.max(localization_out.numpy())), 1.0)
        self.assertLessEqual(float(np.min(localization_out.numpy())), 0.0)

        # Validate detection output.
        detection_out: torch.Tensor = out["detection"].detach().cpu()
        self.assertEqual(torch.Size((8, 2)), detection_out.size())

    def test_with_three_inputs_batch_one_single_value_detection(self) -> None:
        m: fusion.FusionModelWithImage = fusion.FusionModelWithImage(
            inputs_num=3,
            first_stage_depth=8,
            first_stage_heads=8,
            second_stage_depth=8,
            second_stage_heads=8,
            height=224,
            width=224,
            detection_head_type=fusion.DetectionHeadType.SINGLE_OUT
        )
        t: torch.Tensor = torch.rand((1, 9, 224, 224))
        out: dict[str, torch.Tensor] = m(t)

        # Check that output contains both localization and detection tensors.
        self.assertIn("localization", out)
        self.assertIn("detection", out)

        # Validate localization map.
        localization_out: torch.Tensor = out["localization"].detach().cpu()
        self.assertEqual(torch.Size((1, 1, 224, 224)), localization_out.size())
        # Localization map should have values in [0, 1].
        self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

        # Validate detection output.
        detection_out: torch.Tensor = out["detection"].detach().cpu()
        self.assertEqual(torch.Size((1, 1)), detection_out.size())
        self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_three_inputs_batch_eight_single_value_detection(self) -> None:
        m: fusion.FusionModelWithImage = fusion.FusionModelWithImage(
            inputs_num=3,
            first_stage_depth=8,
            first_stage_heads=8,
            second_stage_depth=8,
            second_stage_heads=8,
            height=224,
            width=224,
            detection_head_type=fusion.DetectionHeadType.SINGLE_OUT
        )
        t: torch.Tensor = torch.rand((8, 9, 224, 224))
        out: dict[str, torch.Tensor] = m(t)

        self.assertIn("localization", out)
        self.assertIn("detection", out)

        # Validate localization map.
        localization_out: torch.Tensor = out["localization"].detach().cpu()
        self.assertEqual(torch.Size((8, 1, 224, 224)), localization_out.size())
        # Localization map should have values in [0, 1].
        self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

        # Validate detection output.
        detection_out: torch.Tensor = out["detection"].detach().cpu()
        self.assertEqual(torch.Size((8, 1)), detection_out.size())
        self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_three_inputs_batch_one_single_value_detection_without_logits_layer_res_upscale(
        self
    ) -> None:
        m: fusion.FusionModelWithImage = fusion.FusionModelWithImage(
            inputs_num=3,
            first_stage_depth=8,
            first_stage_heads=8,
            second_stage_depth=8,
            second_stage_heads=8,
            height=224,
            width=224,
            detection_head_type=fusion.DetectionHeadType.SINGLE_OUT,
            add_final_logit_layers=False,
            upscaler_type=fusion.UpscalerType.RESIDUAL_UPSCALER
        )
        t: torch.Tensor = torch.rand((1, 9, 224, 224))
        out: dict[str, torch.Tensor] = m(t)

        # Check that output contains both localization and detection tensors.
        self.assertIn("localization", out)
        self.assertIn("detection", out)

        # Validate localization map.
        localization_out: torch.Tensor = out["localization"].detach().cpu()
        self.assertEqual(torch.Size((1, 1, 224, 224)), localization_out.size())
        # Localization map without logits should not have values in [0, 1].
        self.assertGreaterEqual(float(np.max(localization_out.numpy())), 1.0)
        self.assertLessEqual(float(np.min(localization_out.numpy())), 0.0)

        # Validate detection output.
        detection_out: torch.Tensor = out["detection"].detach().cpu()
        self.assertEqual(torch.Size((1, 1)), detection_out.size())

    def test_with_three_inputs_batch_eight_single_value_detection_without_logits_layer_res_upscale(
        self
    ) -> None:
        m: fusion.FusionModelWithImage = fusion.FusionModelWithImage(
            inputs_num=3,
            first_stage_depth=8,
            first_stage_heads=8,
            second_stage_depth=8,
            second_stage_heads=8,
            height=224,
            width=224,
            detection_head_type=fusion.DetectionHeadType.SINGLE_OUT,
            add_final_logit_layers=False,
            upscaler_type=fusion.UpscalerType.RESIDUAL_UPSCALER
        )
        t: torch.Tensor = torch.rand((8, 9, 224, 224))
        out: dict[str, torch.Tensor] = m(t)

        self.assertIn("localization", out)
        self.assertIn("detection", out)

        # Validate localization map.
        localization_out: torch.Tensor = out["localization"].detach().cpu()
        self.assertEqual(torch.Size((8, 1, 224, 224)), localization_out.size())
        # Localization map without logits should not have values in [0, 1].
        self.assertGreaterEqual(float(np.max(localization_out.numpy())), 1.0)
        self.assertLessEqual(float(np.min(localization_out.numpy())), 0.0)

        # Validate detection output.
        detection_out: torch.Tensor = out["detection"].detach().cpu()
        self.assertEqual(torch.Size((8, 1)), detection_out.size())


class TestMaskedAttentionFusionModelWithImage(unittest.TestCase):

    def test_with_three_inputs_batch_one(self) -> None:
        target_sizes: list[tuple[int, int]] = [
            (224, 224),
            (448, 448)
        ]

        for ts in target_sizes:
            m: fusion.MaskedAttentionFusionModelWithImage = \
                fusion.MaskedAttentionFusionModelWithImage(
                    inputs_num=3,
                    first_stage_depth=8,
                    first_stage_heads=8,
                    second_stage_depth=8,
                    second_stage_heads=8,
                    height=ts[0],
                    width=ts[1]
                )
            t: torch.Tensor = torch.rand((1, 6, ts[0], ts[1]))
            attention_mask: torch.Tensor = torch.randint(
                0, 2, (1, ts[0]**2//16**2, ts[1]**2//16**2)
            ).bool()
            out: dict[str, torch.Tensor] = m(t, attention_mask)

            # Check that output contains both localization and detection tensors.
            self.assertIn("localization", out)
            self.assertIn("detection", out)

            # Validate localization map.
            localization_out: torch.Tensor = out["localization"].detach().cpu()
            self.assertEqual(torch.Size((1, 1, ts[0], ts[1])), localization_out.size())
            # Localization map should have values in [0, 1].
            self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
            self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

            # Validate detection output.
            detection_out: torch.Tensor = out["detection"].detach().cpu()
            self.assertEqual(torch.Size((1, 2)), detection_out.size())
            self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
            self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_three_inputs_batch_eight(self) -> None:
        target_sizes: list[tuple[int, int]] = [
            (224, 224),
            (448, 448)
        ]
        for ts in target_sizes:
            m: fusion.MaskedAttentionFusionModelWithImage = fusion.MaskedAttentionFusionModelWithImage(
                inputs_num=3,
                first_stage_depth=8,
                first_stage_heads=8,
                second_stage_depth=8,
                second_stage_heads=8,
                height=ts[0],
                width=ts[1]
            )
            t: torch.Tensor = torch.rand((8, 6, ts[0], ts[1]))
            attention_mask: torch.Tensor = torch.randint(
                0, 2, (8, ts[0]**2//16**2, ts[1]**2//16**2)
            ).bool()
            out: dict[str, torch.Tensor] = m(t, attention_mask)

            # Check that output contains both localization and detection tensors.
            self.assertIn("localization", out)
            self.assertIn("detection", out)

            # Validate localization map.
            localization_out: torch.Tensor = out["localization"].detach().cpu()
            self.assertEqual(torch.Size((8, 1, ts[0], ts[1])), localization_out.size())
            # Localization map should have values in [0, 1].
            self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
            self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

            # Validate detection output.
            detection_out: torch.Tensor = out["detection"].detach().cpu()
            self.assertEqual(torch.Size((8, 2)), detection_out.size())
            self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
            self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_three_inputs_batch_one_double_conv_upscaler(self) -> None:
        m: fusion.MaskedAttentionFusionModelWithImage = fusion.MaskedAttentionFusionModelWithImage(
            inputs_num=3,
            first_stage_depth=8,
            first_stage_heads=8,
            second_stage_depth=8,
            second_stage_heads=8,
            height=224,
            width=224,
            upscaler_type=fusion.UpscalerType.DOUBLE_CONV_UPSCALER
        )
        t: torch.Tensor = torch.rand((1, 6, 224, 224))
        attention_mask: torch.Tensor = torch.randint(0, 2, (1, 224**2//16**2, 224**2//16**2)).bool()
        out: dict[str, torch.Tensor] = m(t, attention_mask)

        # Check that output contains both localization and detection tensors.
        self.assertIn("localization", out)
        self.assertIn("detection", out)

        # Validate localization map.
        localization_out: torch.Tensor = out["localization"].detach().cpu()
        self.assertEqual(torch.Size((1, 1, 224, 224)), localization_out.size())
        # Localization map should have values in [0, 1].
        self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

        # Validate detection output.
        detection_out: torch.Tensor = out["detection"].detach().cpu()
        self.assertEqual(torch.Size((1, 2)), detection_out.size())
        self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_three_inputs_batch_eight_double_conv_upscaler(self) -> None:
        m: fusion.MaskedAttentionFusionModelWithImage = fusion.MaskedAttentionFusionModelWithImage(
            inputs_num=3,
            first_stage_depth=8,
            first_stage_heads=8,
            second_stage_depth=8,
            second_stage_heads=8,
            height=224,
            width=224,
            upscaler_type=fusion.UpscalerType.DOUBLE_CONV_UPSCALER
        )
        t: torch.Tensor = torch.rand((8, 6, 224, 224))
        attention_mask: torch.Tensor = torch.randint(0, 2, (8, 224**2//16**2, 224**2//16**2)).bool()
        out: dict[str, torch.Tensor] = m(t, attention_mask)

        # Check that output contains both localization and detection tensors.
        self.assertIn("localization", out)
        self.assertIn("detection", out)

        # Validate localization map.
        localization_out: torch.Tensor = out["localization"].detach().cpu()
        self.assertEqual(torch.Size((8, 1, 224, 224)), localization_out.size())
        # Localization map should have values in [0, 1].
        self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

        # Validate detection output.
        detection_out: torch.Tensor = out["detection"].detach().cpu()
        self.assertEqual(torch.Size((8, 2)), detection_out.size())
        self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_three_inputs_positional_fusion(self) -> None:
        fusion_types: list[fusion.FusionType] = [
            fusion.FusionType.POSITIONAL_FUSION,
            fusion.FusionType.POSITIONAL_FUSION_NO_TFT,
            fusion.FusionType.POSITIONAL_FUSION_NO_LDT
        ]

        batch_sizes: list[int] = [1, 5, 8]
        drop_stream_probabilities: list[float] = [0.0, 0.15, 1.0]
        drop_path_probabilities: list[float] = [0.0, 0.15, 1.0]
        for ft in fusion_types:
            for ds in drop_stream_probabilities:
                for dp in drop_path_probabilities:
                    for b in batch_sizes:
                        m = fusion.MaskedAttentionFusionModelWithImage(
                            inputs_num=3,
                            first_stage_depth=8,
                            first_stage_heads=8,
                            second_stage_depth=8,
                            second_stage_heads=8,
                            height=224,
                            width=224,
                            second_stage_fusion_type=ft,
                            drop_stream_probability=ds,
                            drop_path_probability=dp
                        )
                        t: torch.Tensor = torch.rand((b, 6, 224, 224))
                        attention_mask: torch.Tensor = torch.randint(
                            0, 2, (b, 224**2//16**2, 224**2//16**2)).bool()
                        out: dict[str, torch.Tensor] = m(t, attention_mask)

                        # Check that output contains both localization and detection tensors.
                        self.assertIn("localization", out)
                        self.assertIn("detection", out)

                        # Validate localization map.
                        localization_out: torch.Tensor = out["localization"].detach().cpu()
                        self.assertEqual(torch.Size((b, 1, 224, 224)), localization_out.size())
                        # Localization map should have values in [0, 1].
                        self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
                        self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

                        # Validate detection output.
                        detection_out: torch.Tensor = out["detection"].detach().cpu()
                        self.assertEqual(torch.Size((b, 2)), detection_out.size())
                        self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
                        self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_three_inputs_dino_backbone(self) -> None:
        dino_backbones: list[fusion.BackboneType] = [
            fusion.BackboneType.NONE,
            fusion.BackboneType.DINO,
            fusion.BackboneType.DINOv2,
            fusion.BackboneType.DINOv2_FROZEN,
            fusion.BackboneType.DINOv2_FROZEN_FEATURE_INTERPOLATION,
            fusion.BackboneType.DINOv2_FROZEN_PENULTIMATE_LAYER,
            fusion.BackboneType.DINOv2_FROZEN_PENULTIMATE_LAYER_FEATURE_INTERPOLATION,
            fusion.BackboneType.DINOv2_FROZEN_MULTISCALE,
            fusion.BackboneType.DINOv2_FROZEN_MULTISCALE_FEATURE_INTERPOLATION,
            fusion.BackboneType.DINOv2_FROZEN_MULTISCALE_BILINEAR_FEAT_INT
        ]
        batch_sizes: list[int] = [1, 5, 8]
        target_sizes: list[tuple[int, int]] = [
            (224, 224)
        ]
        drop_stream_probabilities: list[float] = [0.0, 0.15, 1.0]
        drop_path_probabilities: list[float] = [0.0, 0.15, 1.0]

        for ts in target_sizes:
            for ds in drop_stream_probabilities:
                for dp in drop_path_probabilities:
                    for backbone_type in dino_backbones:
                        for b in batch_sizes:
                            if ts[0] > 224 and b > 4:
                                continue

                            m = fusion.MaskedAttentionFusionModelWithImage(
                                inputs_num=3,
                                first_stage_depth=8,
                                first_stage_heads=8,
                                second_stage_depth=8,
                                second_stage_heads=8,
                                height=ts[0],
                                width=ts[1],
                                image_backbone_type=backbone_type,
                                drop_stream_probability=ds,
                                drop_path_probability=dp
                            )
                            t: torch.Tensor = torch.rand((b, 6, ts[0], ts[1]))
                            attention_mask: torch.Tensor = torch.randint(
                                0, 2, (b, ts[0]**2//16**2, ts[1]**2//16**2)).bool()
                            out: dict[str, torch.Tensor] = m(t, attention_mask)

                            # Check that output contains both localization and detection tensors.
                            self.assertIn("localization", out)
                            self.assertIn("detection", out)

                            # Validate localization map.
                            localization_out: torch.Tensor = out["localization"].detach().cpu()
                            self.assertEqual(torch.Size((b, 1, ts[0], ts[1])), localization_out.size())
                            # Localization map should have values in [0, 1].
                            self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
                            self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

                            # Validate detection output.
                            detection_out: torch.Tensor = out["detection"].detach().cpu()
                            self.assertEqual(torch.Size((b, 2)), detection_out.size())
                            self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
                            self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_three_inputs_dino_backbone_448_input(self) -> None:
        dino_backbones: list[fusion.BackboneType] = [
            # fusion.BackboneType.DINO,
            fusion.BackboneType.DINOv2,
            fusion.BackboneType.DINOv2_FROZEN,
            fusion.BackboneType.DINOv2_FROZEN_FEATURE_INTERPOLATION,
            fusion.BackboneType.DINOv2_FROZEN_PENULTIMATE_LAYER,
            fusion.BackboneType.DINOv2_FROZEN_PENULTIMATE_LAYER_FEATURE_INTERPOLATION,
            fusion.BackboneType.DINOv2_FROZEN_MULTISCALE,
            fusion.BackboneType.DINOv2_FROZEN_MULTISCALE_FEATURE_INTERPOLATION,
            fusion.BackboneType.DINOv2_FROZEN_MULTISCALE_BILINEAR_FEAT_INT
        ]
        batch_sizes: list[int] = [1, 5, 8]
        target_sizes: list[tuple[int, int]] = [
            (448, 448)
        ]
        drop_stream_probabilities: list[float] = [0.0, 0.15, 1.0]
        drop_path_probabilities: list[float] = [0.0, 0.15, 1.0]

        for ts in target_sizes:
            for ds in drop_stream_probabilities:
                for dp in drop_path_probabilities:
                    for backbone_type in dino_backbones:
                        for b in batch_sizes:
                            if ts[0] > 224 and b > 4:
                                continue

                            m = fusion.MaskedAttentionFusionModelWithImage(
                                inputs_num=3,
                                first_stage_depth=8,
                                first_stage_heads=8,
                                second_stage_depth=8,
                                second_stage_heads=8,
                                height=ts[0],
                                width=ts[1],
                                image_backbone_type=backbone_type,
                                drop_stream_probability=ds,
                                drop_path_probability=dp
                            )
                            t: torch.Tensor = torch.rand((b, 6, ts[0], ts[1]))
                            attention_mask: torch.Tensor = torch.randint(
                                0, 2, (b, ts[0]**2//16**2, ts[1]**2//16**2)).bool()
                            out: dict[str, torch.Tensor] = m(t, attention_mask)

                            # Check that output contains both localization and detection tensors.
                            self.assertIn("localization", out)
                            self.assertIn("detection", out)

                            # Validate localization map.
                            localization_out: torch.Tensor = out["localization"].detach().cpu()
                            self.assertEqual(torch.Size((b, 1, ts[0], ts[1])), localization_out.size())
                            # Localization map should have values in [0, 1].
                            self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
                            self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

                            # Validate detection output.
                            detection_out: torch.Tensor = out["detection"].detach().cpu()
                            self.assertEqual(torch.Size((b, 2)), detection_out.size())
                            self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
                            self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_three_inputs_dino_backbones_with_instance_guided_attention(self) -> None:
        dino_backbones: list[fusion.BackboneType] = [
            fusion.BackboneType.DINO,
            fusion.BackboneType.DINOv2,
            fusion.BackboneType.DINOv2_FROZEN,
            fusion.BackboneType.DINOv2_FROZEN_FEATURE_INTERPOLATION,
            fusion.BackboneType.DINOv2_FROZEN_BILINEAR_FEAT_INT,
            fusion.BackboneType.DINOv2_FROZEN_PENULTIMATE_LAYER,
            fusion.BackboneType.DINOv2_FROZEN_PENULTIMATE_LAYER_FEATURE_INTERPOLATION,
            fusion.BackboneType.DINOv2_FROZEN_MULTISCALE,
            fusion.BackboneType.DINOv2_FROZEN_MULTISCALE_FEATURE_INTERPOLATION,
            fusion.BackboneType.DINOv2_FROZEN_MULTISCALE_BILINEAR_FEAT_INT
        ]
        batch_sizes: list[int] = [1, 5, 8]
        drop_stream_probabilities: list[float] = [0.0, 0.15, 1.0]
        drop_path_probabilities: list[float] = [0.0, 0.15, 1.0]

        for ds in drop_stream_probabilities:
            for dp in drop_path_probabilities:
                for backbone_type in dino_backbones:
                    for b in batch_sizes:
                        m = fusion.MaskedAttentionFusionModelWithImage(
                            inputs_num=3,
                            first_stage_depth=8,
                            first_stage_heads=8,
                            second_stage_depth=8,
                            second_stage_heads=8,
                            height=224,
                            width=224,
                            image_backbone_type=backbone_type,
                            pass_dino_features_through_first_stage=True,
                            drop_stream_probability=ds,
                            drop_path_probability=dp
                        )
                        t: torch.Tensor = torch.rand((b, 6, 224, 224))
                        attention_mask: torch.Tensor = torch.randint(
                            0, 2, (b, 224**2//16**2, 224**2//16**2)).bool()
                        out: dict[str, torch.Tensor] = m(t, attention_mask)

                        # Check that output contains both localization and detection tensors.
                        self.assertIn("localization", out)
                        self.assertIn("detection", out)

                        # Validate localization map.
                        localization_out: torch.Tensor = out["localization"].detach().cpu()
                        self.assertEqual(torch.Size((b, 1, 224, 224)), localization_out.size())
                        # Localization map should have values in [0, 1].
                        self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
                        self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

                        # Validate detection output.
                        detection_out: torch.Tensor = out["detection"].detach().cpu()
                        self.assertEqual(torch.Size((b, 2)), detection_out.size())
                        self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
                        self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_return_similarities(self) -> None:
        target_sizes: list[tuple[int, int]] = [
            (224, 224),
            (448, 448)
        ]
        for ts in target_sizes:
            m: fusion.MaskedAttentionFusionModelWithImage = fusion.MaskedAttentionFusionModelWithImage(
                inputs_num=3,
                first_stage_depth=8,
                first_stage_heads=8,
                second_stage_depth=8,
                second_stage_heads=8,
                height=ts[0],
                width=ts[1],
                second_stage_fusion_type=fusion.FusionType.POSITIONAL_FUSION,
            )
            t: torch.Tensor = torch.rand((8, 6, ts[0], ts[1]))
            attention_mask: torch.Tensor = torch.randint(
                0, 2, (8, ts[0] ** 2 // 16 ** 2, ts[1] ** 2 // 16 ** 2)
            ).bool()
            out: dict[str, torch.Tensor] = m(t, attention_mask, return_similarities=True)

            # Check that output contains both localization and detection tensors.
            self.assertIn("localization", out)
            self.assertIn("detection", out)

            # Validate localization map.
            localization_out: torch.Tensor = out["localization"].detach().cpu()
            self.assertEqual(torch.Size((8, 1, ts[0], ts[1])), localization_out.size())
            # Localization map should have values in [0, 1].
            self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
            self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

            # Validate detection output.
            detection_out: torch.Tensor = out["detection"].detach().cpu()
            self.assertEqual(torch.Size((8, 2)), detection_out.size())
            self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
            self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

            # Validate similarities matrices.
            for s in out["similarities"]:
                self.assertEqual(torch.Size((8, ts[0] // 16, ts[1] // 16)), s.size())

    def test_return_attention_map(self) -> None:
        target_sizes: list[tuple[int, int]] = [
            (224, 224),
            (448, 448)
        ]
        for ts in target_sizes:
            m: fusion.MaskedAttentionFusionModelWithImage = fusion.MaskedAttentionFusionModelWithImage(
                inputs_num=3,
                first_stage_depth=8,
                first_stage_heads=8,
                second_stage_depth=8,
                second_stage_heads=8,
                height=ts[0],
                width=ts[1],
                second_stage_fusion_type=fusion.FusionType.POSITIONAL_FUSION,
            )
            t: torch.Tensor = torch.rand((8, 6, ts[0], ts[1]))
            attention_mask: torch.Tensor = torch.randint(
                0, 2, (8, ts[0] ** 2 // 16 ** 2, ts[1] ** 2 // 16 ** 2)
            ).bool()
            out: dict[str, torch.Tensor] = m(t, attention_mask,
                                             return_similarities=True,
                                             similarities_type="attn")

            # Check that output contains both localization and detection tensors.
            self.assertIn("localization", out)
            self.assertIn("detection", out)

            # Validate localization map.
            localization_out: torch.Tensor = out["localization"].detach().cpu()
            self.assertEqual(torch.Size((8, 1, ts[0], ts[1])), localization_out.size())
            # Localization map should have values in [0, 1].
            self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
            self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

            # Validate detection output.
            detection_out: torch.Tensor = out["detection"].detach().cpu()
            self.assertEqual(torch.Size((8, 2)), detection_out.size())
            self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
            self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

            # Validate similarities matrices.
            self.assertEqual(len(out["similarities"]), 4)
            for s in out["similarities"]:
                self.assertEqual(torch.Size((8, 8, 8, ts[0] // 16, ts[1] // 16)), s.size())


class TestMaskedAttentionFusionModel(unittest.TestCase):
    def test_with_three_inputs_batch_one(self) -> None:
        target_size: list[tuple[int, int]] = [
            (224, 224)
        ]

        for ts in target_size:
            m: fusion.MaskedAttentionFusionModel = fusion.MaskedAttentionFusionModel(
                inputs_num=3,
                first_stage_depth=8,
                first_stage_heads=8,
                second_stage_depth=8,
                second_stage_heads=8,
                height=ts[0],
                width=ts[1]
            )
            t: torch.Tensor = torch.rand((1, 3, ts[0], ts[1]))
            attention_mask: torch.Tensor = torch.randint(
                0, 2, (1, ts[0]**2//16**2, ts[1]**2//16**2)
            ).bool()
            out: dict[str, torch.Tensor] = m(t, attention_mask)

            # Check that output contains both localization and detection tensors.
            self.assertIn("localization", out)
            self.assertIn("detection", out)

            # Validate localization map.
            localization_out: torch.Tensor = out["localization"].detach().cpu()
            self.assertEqual(torch.Size((1, 1, ts[0], ts[1])), localization_out.size())
            # Localization map should have values in [0, 1].
            self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
            self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

            # Validate detection output.
            detection_out: torch.Tensor = out["detection"].detach().cpu()
            self.assertEqual(torch.Size((1, 2)), detection_out.size())
            self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
            self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_three_inputs_batch_eight(self) -> None:
        m: fusion.MaskedAttentionFusionModel = fusion.MaskedAttentionFusionModel(
            inputs_num=3,
            first_stage_depth=8,
            first_stage_heads=8,
            second_stage_depth=8,
            second_stage_heads=8,
            height=224,
            width=224
        )
        t: torch.Tensor = torch.rand((8, 3, 224, 224))
        attention_mask: torch.Tensor = torch.randint(0, 2, (8, 224**2//16**2, 224**2//16**2)).bool()
        out: dict[str, torch.Tensor] = m(t, attention_mask)

        # Check that output contains both localization and detection tensors.
        self.assertIn("localization", out)
        self.assertIn("detection", out)

        # Validate localization map.
        localization_out: torch.Tensor = out["localization"].detach().cpu()
        self.assertEqual(torch.Size((8, 1, 224, 224)), localization_out.size())
        # Localization map should have values in [0, 1].
        self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

        # Validate detection output.
        detection_out: torch.Tensor = out["detection"].detach().cpu()
        self.assertEqual(torch.Size((8, 2)), detection_out.size())
        self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_three_inputs_batch_one_cape_pos_emb(self) -> None:
        m: fusion.MaskedAttentionFusionModel = fusion.MaskedAttentionFusionModel(
            inputs_num=3,
            first_stage_depth=8,
            first_stage_heads=8,
            second_stage_depth=8,
            second_stage_heads=8,
            height=224,
            width=224,
            pos_embeddings_type=fusion.PositionalEmbeddingsType.CAPE
        )
        t: torch.Tensor = torch.rand((1, 3, 224, 224))
        attention_mask: torch.Tensor = torch.randint(0, 2, (1, 224**2//16**2, 224**2//16**2)).bool()
        out: dict[str, torch.Tensor] = m(t, attention_mask)

        # Check that output contains both localization and detection tensors.
        self.assertIn("localization", out)
        self.assertIn("detection", out)

        # Validate localization map.
        localization_out: torch.Tensor = out["localization"].detach().cpu()
        self.assertEqual(torch.Size((1, 1, 224, 224)), localization_out.size())
        # Localization map should have values in [0, 1].
        self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

        # Validate detection output.
        detection_out: torch.Tensor = out["detection"].detach().cpu()
        self.assertEqual(torch.Size((1, 2)), detection_out.size())
        self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_three_inputs_batch_eight_cape_pos_emb(self) -> None:
        m: fusion.MaskedAttentionFusionModel = fusion.MaskedAttentionFusionModel(
            inputs_num=3,
            first_stage_depth=8,
            first_stage_heads=8,
            second_stage_depth=8,
            second_stage_heads=8,
            height=224,
            width=224,
            pos_embeddings_type=fusion.PositionalEmbeddingsType.CAPE
        )
        t: torch.Tensor = torch.rand((8, 3, 224, 224))
        attention_mask: torch.Tensor = torch.randint(0, 2, (8, 224**2//16**2, 224**2//16**2)).bool()
        out: dict[str, torch.Tensor] = m(t, attention_mask)

        # Check that output contains both localization and detection tensors.
        self.assertIn("localization", out)
        self.assertIn("detection", out)

        # Validate localization map.
        localization_out: torch.Tensor = out["localization"].detach().cpu()
        self.assertEqual(torch.Size((8, 1, 224, 224)), localization_out.size())
        # Localization map should have values in [0, 1].
        self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

        # Validate detection output.
        detection_out: torch.Tensor = out["detection"].detach().cpu()
        self.assertEqual(torch.Size((8, 2)), detection_out.size())
        self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_three_inputs_batch_one_positional_wise_fusion(self) -> None:
        m: fusion.MaskedAttentionFusionModel = fusion.MaskedAttentionFusionModel(
            inputs_num=3,
            first_stage_depth=8,
            first_stage_heads=8,
            second_stage_depth=8,
            second_stage_heads=8,
            height=224,
            width=224,
            second_stage_fusion_type=fusion.FusionType.POSITIONAL_FUSION
        )
        t: torch.Tensor = torch.rand((1, 3, 224, 224))
        attention_mask: torch.Tensor = torch.randint(0, 2, (1, 224**2//16**2, 224**2//16**2)).bool()
        out: dict[str, torch.Tensor] = m(t, attention_mask)

        # Check that output contains both localization and detection tensors.
        self.assertIn("localization", out)
        self.assertIn("detection", out)

        # Validate localization map.
        localization_out: torch.Tensor = out["localization"].detach().cpu()
        self.assertEqual(torch.Size((1, 1, 224, 224)), localization_out.size())
        # Localization map should have values in [0, 1].
        self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

        # Validate detection output.
        detection_out: torch.Tensor = out["detection"].detach().cpu()
        self.assertEqual(torch.Size((1, 2)), detection_out.size())
        self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_three_inputs_batch_eight_positional_wise_fusion(self) -> None:
        m: fusion.MaskedAttentionFusionModel = fusion.MaskedAttentionFusionModel(
            inputs_num=3,
            first_stage_depth=8,
            first_stage_heads=8,
            second_stage_depth=8,
            second_stage_heads=8,
            height=224,
            width=224,
            second_stage_fusion_type=fusion.FusionType.POSITIONAL_FUSION
        )
        t: torch.Tensor = torch.rand((8, 3, 224, 224))
        attention_mask: torch.Tensor = torch.randint(0, 2, (8, 224**2//16**2, 224**2//16**2)).bool()
        out: dict[str, torch.Tensor] = m(t, attention_mask)

        # Check that output contains both localization and detection tensors.
        self.assertIn("localization", out)
        self.assertIn("detection", out)

        # Validate localization map.
        localization_out: torch.Tensor = out["localization"].detach().cpu()
        self.assertEqual(torch.Size((8, 1, 224, 224)), localization_out.size())
        # Localization map should have values in [0, 1].
        self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

        # Validate detection output.
        detection_out: torch.Tensor = out["detection"].detach().cpu()
        self.assertEqual(torch.Size((8, 2)), detection_out.size())
        self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_three_inputs_batch_one_double_conv_one_hot_upscaler(self) -> None:
        m: fusion.MaskedAttentionFusionModel = fusion.MaskedAttentionFusionModel(
            inputs_num=3,
            first_stage_depth=8,
            first_stage_heads=8,
            second_stage_depth=8,
            second_stage_heads=8,
            height=224,
            width=224,
            upscaler_type=fusion.UpscalerType.DOUBLE_CONV_UPSCALER_ONE_HOT
        )
        t: torch.Tensor = torch.rand((1, 3, 224, 224))
        attention_mask: torch.Tensor = torch.randint(0, 2, (1, 224**2//16**2, 224**2//16**2)).bool()
        out: dict[str, torch.Tensor] = m(t, attention_mask)

        # Check that output contains both localization and detection tensors.
        self.assertIn("localization", out)
        self.assertIn("detection", out)

        # Validate localization map.
        localization_out: torch.Tensor = out["localization"].detach().cpu()
        self.assertEqual(torch.Size((1, 2, 224, 224)), localization_out.size())
        # Localization map should have values in [0, 1].
        self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

        # Validate detection output.
        detection_out: torch.Tensor = out["detection"].detach().cpu()
        self.assertEqual(torch.Size((1, 2)), detection_out.size())
        self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_three_inputs_batch_eight_double_conv_one_hot_upscaler(self) -> None:
        m: fusion.MaskedAttentionFusionModel = fusion.MaskedAttentionFusionModel(
            inputs_num=3,
            first_stage_depth=8,
            first_stage_heads=8,
            second_stage_depth=8,
            second_stage_heads=8,
            height=224,
            width=224,
            upscaler_type=fusion.UpscalerType.DOUBLE_CONV_UPSCALER_ONE_HOT
        )
        t: torch.Tensor = torch.rand((8, 3, 224, 224))
        attention_mask: torch.Tensor = torch.randint(0, 2, (8, 224**2//16**2, 224**2//16**2)).bool()
        out: dict[str, torch.Tensor] = m(t, attention_mask)

        # Check that output contains both localization and detection tensors.
        self.assertIn("localization", out)
        self.assertIn("detection", out)

        # Validate localization map.
        localization_out: torch.Tensor = out["localization"].detach().cpu()
        self.assertEqual(torch.Size((8, 2, 224, 224)), localization_out.size())
        # Localization map should have values in [0, 1].
        self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

        # Validate detection output.
        detection_out: torch.Tensor = out["detection"].detach().cpu()
        self.assertEqual(torch.Size((8, 2)), detection_out.size())
        self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)

    def test_with_softmax_transformer_detector(self) -> None:
        batch_sizes: list[int] = [1, 4, 8]

        for b in batch_sizes:
            m: fusion.MaskedAttentionFusionModel = fusion.MaskedAttentionFusionModel(
                inputs_num=3,
                first_stage_depth=6,
                first_stage_heads=12,
                second_stage_depth=6,
                second_stage_heads=12,
                height=224,
                width=224,
                detection_head_type=fusion.DetectionHeadType.TRANSFORMER_TWO_OUT_SOFTMAX
            )
            t: torch.Tensor = torch.rand((b, 3, 224, 224))
            attention_mask: torch.Tensor = torch.randint(
                0, 2, (b, 224**2//16**2, 224**2//16**2)).bool()
            out: dict[str, torch.Tensor] = m(t, attention_mask)

            # Check that output contains both localization and detection tensors.
            self.assertIn("localization", out)
            self.assertIn("detection", out)

            # Validate localization map.
            localization_out: torch.Tensor = out["localization"].detach().cpu()
            self.assertEqual(torch.Size((b, 1, 224, 224)), localization_out.size())
            # Localization map should have values in [0, 1].
            self.assertLessEqual(float(np.max(localization_out.numpy())), 1.0)
            self.assertGreaterEqual(float(np.min(localization_out.numpy())), 0.0)

            # Validate detection output.
            detection_out: torch.Tensor = out["detection"].detach().cpu()
            self.assertEqual(torch.Size((b, 2)), detection_out.size())
            self.assertLessEqual(float(np.max(detection_out.numpy())), 1.0)
            self.assertGreaterEqual(float(np.min(detection_out.numpy())), 0.0)
            self.assertTrue(torch.all(torch.abs(torch.sum(detection_out, dim=1) - 1.0) < 1e-4))


class TestBottleneckFusionTwoToOne(unittest.TestCase):

    def test_one_deep_with_bottleneck_smaller_than_input(self) -> None:
        m: fusion.BottleneckFusionTwoToOne = fusion.BottleneckFusionTwoToOne(
            dim=384, depth=1, bottleneck_units=8, heads=12, dim_head=384//12, mlp_dim=384*4)
        t: torch.Tensor = torch.randn(1, 2, 196, 384)
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 8, 384)))

    def test_one_deep_with_bottleneck_equal_to_input(self) -> None:
        m: fusion.BottleneckFusionTwoToOne = fusion.BottleneckFusionTwoToOne(
            dim=384, depth=1, bottleneck_units=196, heads=12, dim_head=384//12, mlp_dim=384*4)
        t: torch.Tensor = torch.randn(1, 2, 196, 384)
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 196, 384)))

    def test_many_deep_with_bottleneck_smaller_than_input(self) -> None:
        m: fusion.BottleneckFusionTwoToOne = fusion.BottleneckFusionTwoToOne(
            dim=384, depth=8, bottleneck_units=8, heads=12, dim_head=384//12, mlp_dim=384*4)
        t: torch.Tensor = torch.randn(1, 2, 196, 384)
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 8, 384)))

    def test_many_deep_with_bottleneck_equal_to_input(self) -> None:
        m: fusion.BottleneckFusionTwoToOne = fusion.BottleneckFusionTwoToOne(
            dim=384, depth=8, bottleneck_units=196, heads=12, dim_head=384//12, mlp_dim=384*4)
        t: torch.Tensor = torch.randn(1, 2, 196, 384)
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 196, 384)))


class TestBottleneckFusionManyToOne(unittest.TestCase):
    def test_two_signals_one_deep_with_bottleneck_smaller_than_input(self) -> None:
        m: fusion.BottleneckFusionManyToOne = fusion.BottleneckFusionManyToOne(
            signals=2, dim=384, depth=1, bottleneck_units=8,
            heads=12, dim_head=384//12, mlp_dim=384*4)
        t: torch.Tensor = torch.randn(1, 2, 196, 384)
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 8, 384)))

    def test_two_signals_one_deep_with_bottleneck_equal_to_input(self) -> None:
        m: fusion.BottleneckFusionManyToOne = fusion.BottleneckFusionManyToOne(
            signals=2, dim=384, depth=1, bottleneck_units=196,
            heads=12, dim_head=384//12, mlp_dim=384*4)
        t: torch.Tensor = torch.randn(1, 2, 196, 384)
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 196, 384)))

    def test_two_signals_many_deep_with_bottleneck_smaller_than_input(self) -> None:
        m: fusion.BottleneckFusionManyToOne = fusion.BottleneckFusionManyToOne(
            signals=2, dim=384, depth=8, bottleneck_units=8,
            heads=12, dim_head=384//12, mlp_dim=384*4)
        t: torch.Tensor = torch.randn(1, 2, 196, 384)
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 8, 384)))

    def test_two_signals_many_deep_with_bottleneck_equal_to_input(self) -> None:
        m: fusion.BottleneckFusionManyToOne = fusion.BottleneckFusionManyToOne(
            signals=2, dim=384, depth=8, bottleneck_units=196,
            heads=12, dim_head=384//12, mlp_dim=384*4)
        t: torch.Tensor = torch.randn(1, 2, 196, 384)
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 196, 384)))

    def test_five_signals_one_deep_with_bottleneck_smaller_than_input(self) -> None:
        m: fusion.BottleneckFusionManyToOne = fusion.BottleneckFusionManyToOne(
            signals=5, dim=384, depth=1, bottleneck_units=8,
            heads=12, dim_head=384//12, mlp_dim=384*4)
        t: torch.Tensor = torch.randn(1, 5, 196, 384)
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 8, 384)))

    def test_five_signals_one_deep_with_bottleneck_equal_to_input(self) -> None:
        m: fusion.BottleneckFusionManyToOne = fusion.BottleneckFusionManyToOne(
            signals=5, dim=384, depth=1, bottleneck_units=196,
            heads=12, dim_head=384//12, mlp_dim=384*4)
        t: torch.Tensor = torch.randn(1, 5, 196, 384)
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 196, 384)))

    def test_five_signals_many_deep_with_bottleneck_smaller_than_input(self) -> None:
        m: fusion.BottleneckFusionManyToOne = fusion.BottleneckFusionManyToOne(
            signals=5, dim=384, depth=8, bottleneck_units=8,
            heads=12, dim_head=384//12, mlp_dim=384*4)
        t: torch.Tensor = torch.randn(1, 5, 196, 384)
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 8, 384)))

    def test_five_signals_many_deep_with_bottleneck_equal_to_input(self) -> None:
        m: fusion.BottleneckFusionManyToOne = fusion.BottleneckFusionManyToOne(
            signals=5, dim=384, depth=8, bottleneck_units=196,
            heads=12, dim_head=384//12, mlp_dim=384*4)
        t: torch.Tensor = torch.randn(1, 5, 196, 384)
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((1, 196, 384)))


class TestPositionalWiseFusion(unittest.TestCase):
    def test_many_deep(self) -> None:
        m: fusion.PositionalWiseFusion = fusion.PositionalWiseFusion(
            positional_fusion_layers=6,
            long_rage_layers=6,
            dim=384,
            heads=12,
            dim_head=384//12,
            mlp_dim=384*4
        )
        t: torch.Tensor = torch.randn(8, 5, 196, 384)  # (B, S, H*W, C)
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((8, 196, 384)))

    def test_zero_deep_positional_fusion(self) -> None:
        m: fusion.PositionalWiseFusion = fusion.PositionalWiseFusion(
            positional_fusion_layers=0,
            long_rage_layers=6,
            dim=384,
            heads=12,
            dim_head=384//12,
            mlp_dim=384*4
        )
        t: torch.Tensor = torch.randn(8, 5, 196, 384)  # (B, S, H*W, C)
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((8, 196, 384)))

    def test_zero_deep_long_range_layers(self) -> None:
        m: fusion.PositionalWiseFusion = fusion.PositionalWiseFusion(
            positional_fusion_layers=6,
            long_rage_layers=0,
            dim=384,
            heads=12,
            dim_head=384//12,
            mlp_dim=384*4
        )
        t: torch.Tensor = torch.randn(8, 5, 196, 384)  # (B, S, H*W, C)
        out: torch.Tensor = m(t)
        self.assertEqual(out.size(), torch.Size((8, 196, 384)))

    def test_return_similarities(self) -> None:
        m: fusion.PositionalWiseFusion = fusion.PositionalWiseFusion(
            positional_fusion_layers=2,
            long_rage_layers=2,
            dim=384,
            heads=12,
            dim_head=384 // 12,
            mlp_dim=384 * 4
        )
        t: torch.Tensor = torch.randn(8, 5, 196, 384)  # (B, S, H*W, C)
        out: torch.Tensor
        similarities: list[torch.Tensor]
        out, similarities = m(t, return_similarities=True)
        self.assertEqual(out.size(), torch.Size((8, 196, 384)))
        self.assertEqual(len(similarities), 5)
        for s in similarities:
            self.assertEqual(s.size(), torch.Size((8, 196)))


class TestMaskedTransformer(unittest.TestCase):

    def test_masked_transformer_batch_1(self) -> None:
        m: fusion.MaskedTransformer = fusion.MaskedTransformer(
            dim=384, depth=8, heads=12, dim_head=384//12, mlp_dim=384*4
        )
        x: torch.Tensor = torch.randn(1, 196, 384)
        attention_mask: torch.Tensor = torch.randint(0, 2, [1, 196, 196]).bool()
        out: torch.Tensor = m(x, attention_mask)
        self.assertEqual(torch.Size((1, 196, 384)), out.size())

    def test_masked_transformer_batch_8(self) -> None:
        m: fusion.MaskedTransformer = fusion.MaskedTransformer(
            dim=384, depth=8, heads=12, dim_head=384//12, mlp_dim=384*4
        )
        x: torch.Tensor = torch.randn(8, 196, 384)
        attention_mask: torch.Tensor = torch.randint(0, 2, [8, 196, 196]).bool()
        out: torch.Tensor = m(x, attention_mask)
        self.assertEqual(torch.Size((8, 196, 384)), out.size())
