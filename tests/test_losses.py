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
from torch.nn import functional
import numpy as np
import einops

from omgfuser import losses


class TestLocalizationDetectionLoss(unittest.TestCase):

    def test_loss_batch_8(self) -> None:
        criterion: losses.LocalizationDetectionBCELoss = losses.LocalizationDetectionBCELoss()
        loc_predicted: torch.Tensor = torch.rand((8, 1, 224, 224))
        loc_actual: torch.Tensor = torch.randint(2, (8, 1, 224, 224))
        det_predicted: torch.Tensor = torch.rand((8, 2))
        det_actual: torch.Tensor = torch.randint(2, (8, 2))
        loss: torch.Tensor = criterion(loc_predicted, det_predicted,
                                       loc_actual.float(), det_actual.float())
        self.assertEqual(torch.Size(tuple()), loss.size())
        # self.assertLessEqual(float(loss.numpy()), 2.0)

    def test_loss_batch_1(self) -> None:
        criterion: losses.LocalizationDetectionBCELoss = losses.LocalizationDetectionBCELoss()
        loc_predicted: torch.Tensor = torch.rand((1, 1, 224, 224))
        loc_actual: torch.Tensor = torch.randint(2, (1, 1, 224, 224))
        det_predicted: torch.Tensor = torch.rand((1, 2))
        det_actual: torch.Tensor = torch.randint(2, (1, 2))
        loss: torch.Tensor = criterion(loc_predicted, det_predicted,
                                       loc_actual.float(), det_actual.float())
        self.assertEqual(torch.Size(tuple()), loss.size())
        # self.assertLessEqual(float(loss.numpy()), 2.0)

    def test_loss_batch_8_bce_with_logits(self) -> None:
        criterion: losses.LocalizationDetectionBCELoss = losses.LocalizationDetectionBCELoss(
            loss_type=losses.LossType.BCE_WITH_LOGITS_LOSS
        )
        loc_predicted: torch.Tensor = torch.rand((8, 1, 224, 224))
        loc_actual: torch.Tensor = torch.randint(2, (8, 1, 224, 224))
        det_predicted: torch.Tensor = torch.rand((8, 2))
        det_actual: torch.Tensor = torch.randint(2, (8, 2))
        loss: torch.Tensor = criterion(loc_predicted, det_predicted,
                                       loc_actual.float(), det_actual.float())
        self.assertEqual(torch.Size(tuple()), loss.size())
        # self.assertLessEqual(float(loss.numpy()), 2.0)

    def test_loss_batch_1_bce_with_logits(self) -> None:
        criterion: losses.LocalizationDetectionBCELoss = losses.LocalizationDetectionBCELoss(
            loss_type=losses.LossType.BCE_WITH_LOGITS_LOSS
        )
        loc_predicted: torch.Tensor = torch.rand((1, 1, 224, 224))
        loc_actual: torch.Tensor = torch.randint(2, (1, 1, 224, 224))
        det_predicted: torch.Tensor = torch.rand((1, 2))
        det_actual: torch.Tensor = torch.randint(2, (1, 2))
        loss: torch.Tensor = criterion(loc_predicted, det_predicted,
                                       loc_actual.float(), det_actual.float())
        self.assertEqual(torch.Size(tuple()), loss.size())
        # self.assertLessEqual(float(loss.numpy()), 2.0)

    def test_loss_batch_8_localization_only(self) -> None:
        criterion: losses.LocalizationDetectionBCELoss = losses.LocalizationDetectionBCELoss(
            detection_loss_weight=.0,
            localization_loss_weight=1.0
        )
        loc_input: torch.Tensor = torch.randint(2, (8, 1, 224, 224))

        # Check that detection input does not affect the output of the loss.
        for _ in range(0, 5):
            det_predicted: torch.Tensor = torch.rand((8, 2))
            det_actual: torch.Tensor = torch.randint(2, (8, 2))

            loss: torch.Tensor = criterion(loc_input.float(), det_predicted,
                                           loc_input.float(), det_actual.float())
            self.assertEqual(torch.Size(tuple()), loss.size())
            self.assertAlmostEqual(0.0, loss.numpy().item())

    def test_loss_batch_1(self) -> None:
        criterion: losses.LocalizationDetectionBCELoss = losses.LocalizationDetectionBCELoss(
            detection_loss_weight=.0,
            localization_loss_weight=1.0
        )
        loc_input: torch.Tensor = torch.randint(2, (1, 1, 224, 224))

        # Check that detection input does not affect the output of the loss.
        for _ in range(0, 5):
            det_predicted: torch.Tensor = torch.rand((1, 2))
            det_actual: torch.Tensor = torch.randint(2, (1, 2))

            loss: torch.Tensor = criterion(loc_input.float(), det_predicted,
                                           loc_input.float(), det_actual.float())
            self.assertEqual(torch.Size(tuple()), loss.size())
            self.assertAlmostEqual(0.0, loss.numpy().item())


class TestClassAwareLocalizationDetectionBCELoss(unittest.TestCase):

    def test_loss_batch_8(self) -> None:
        criterion: losses.ClassAwareLocalizationDetectionBCELoss = \
            losses.ClassAwareLocalizationDetectionBCELoss()

        loc_predicted: torch.Tensor = torch.rand((8, 1, 224, 224))
        loc_actual: torch.Tensor = torch.randint(2, (8, 1, 224, 224))
        det_predicted: torch.Tensor = torch.rand((8, 2))
        det_actual: torch.Tensor = torch.randint(2, (8, 2))

        loss: torch.Tensor = criterion(loc_predicted, det_predicted,
                                       loc_actual.float(), det_actual.float())

        self.assertEqual(torch.Size(tuple()), loss.size())

    def test_loss_batch_1(self) -> None:
        criterion: losses.ClassAwareLocalizationDetectionBCELoss = \
            losses.ClassAwareLocalizationDetectionBCELoss()

        loc_predicted: torch.Tensor = torch.rand((1, 1, 224, 224))
        loc_actual: torch.Tensor = torch.randint(2, (1, 1, 224, 224))
        det_predicted: torch.Tensor = torch.rand((1, 2))
        det_actual: torch.Tensor = torch.randint(2, (1, 2))

        loss: torch.Tensor = criterion(loc_predicted, det_predicted,
                                       loc_actual.float(), det_actual.float())

        self.assertEqual(torch.Size(tuple()), loss.size())

    def test_loss_batch_8_bce_with_logits(self) -> None:
        criterion: losses.ClassAwareLocalizationDetectionBCELoss = \
            losses.ClassAwareLocalizationDetectionBCELoss(
                loss_type=losses.LossType.BCE_WITH_LOGITS_LOSS
            )

        loc_predicted: torch.Tensor = torch.rand((8, 1, 224, 224))
        loc_actual: torch.Tensor = torch.randint(2, (8, 1, 224, 224))
        det_predicted: torch.Tensor = torch.rand((8, 2))
        det_actual: torch.Tensor = torch.randint(2, (8, 2))

        loss: torch.Tensor = criterion(loc_predicted, det_predicted,
                                       loc_actual.float(), det_actual.float())

        self.assertEqual(torch.Size(tuple()), loss.size())

    def test_loss_batch_1_bce_with_logits(self) -> None:
        criterion: losses.ClassAwareLocalizationDetectionBCELoss = \
            losses.ClassAwareLocalizationDetectionBCELoss(
                loss_type=losses.LossType.BCE_WITH_LOGITS_LOSS
            )

        loc_predicted: torch.Tensor = torch.rand((1, 1, 224, 224))
        loc_actual: torch.Tensor = torch.randint(2, (1, 1, 224, 224))
        det_predicted: torch.Tensor = torch.rand((1, 2))
        det_actual: torch.Tensor = torch.randint(2, (1, 2))

        loss: torch.Tensor = criterion(loc_predicted, det_predicted,
                                       loc_actual.float(), det_actual.float())

        self.assertEqual(torch.Size(tuple()), loss.size())

    def test_loss_perfect_match(self) -> None:
        criterion: losses.ClassAwareLocalizationDetectionBCELoss = \
            losses.ClassAwareLocalizationDetectionBCELoss()

        loc_predicted: torch.Tensor = torch.randint(2, (8, 1, 224, 224))
        det_predicted: torch.Tensor = torch.randint(2, (8, 2))

        loss: torch.Tensor = criterion(loc_predicted.float(), det_predicted.float(),
                                       loc_predicted.float(), det_predicted.float())

        self.assertAlmostEqual(0.0, float(loss.numpy()))

    def test_loss_perfect_match_bce_with_logits_lower_limit(self) -> None:
        criterion: losses.ClassAwareLocalizationDetectionBCELoss = \
            losses.ClassAwareLocalizationDetectionBCELoss(
                loss_type=losses.LossType.BCE_WITH_LOGITS_LOSS
            )

        loc_predicted: torch.Tensor = torch.randn((8, 1, 224, 224)) - 50
        loc_actual: torch.Tensor = torch.zeros((8, 1, 224, 224))
        det_predicted: torch.Tensor = torch.randn((8, 2)) - 50
        det_actual: torch.Tensor = torch.zeros((8, 2))

        loss: torch.Tensor = criterion(loc_predicted,
                                       det_predicted,
                                       loc_actual,
                                       det_actual)

        self.assertAlmostEqual(0.0, float(loss.numpy()))

    def test_loss_perfect_match_bce_with_logits_upper_limit(self) -> None:
        criterion: losses.ClassAwareLocalizationDetectionBCELoss = \
            losses.ClassAwareLocalizationDetectionBCELoss(
                loss_type=losses.LossType.BCE_WITH_LOGITS_LOSS
            )

        loc_predicted: torch.Tensor = torch.randn((8, 1, 224, 224)) + 50
        loc_actual: torch.Tensor = torch.ones((8, 1, 224, 224))
        det_predicted: torch.Tensor = torch.randn((8, 2)) + 50
        det_actual: torch.Tensor = torch.ones((8, 2))

        loss: torch.Tensor = criterion(loc_predicted,
                                       det_predicted,
                                       loc_actual,
                                       det_actual)

        self.assertAlmostEqual(.0, float(loss.numpy()))

    def test_loss_match(self) -> None:
        criterion: losses.ClassAwareLocalizationDetectionBCELoss = \
            losses.ClassAwareLocalizationDetectionBCELoss()

        loc_predicted: torch.Tensor = torch.full((1, 1, 224, 224), 0.5)
        loc_actual: torch.Tensor = torch.randint(2, (1, 1, 224, 224))
        det_predicted: torch.Tensor = torch.full((1, 2), 0.5)
        det_actual: torch.Tensor = torch.randint(2, (1, 2))

        loss: torch.Tensor = criterion(loc_predicted, det_predicted,
                                       loc_actual.float(), det_actual.float())

        self.assertAlmostEqual(0.6931, float(loss.numpy()), delta=0.001)

    def test_equality_with_simple_localization_detection_loss_only_authentic(self) -> None:
        criterion: losses.ClassAwareLocalizationDetectionBCELoss = \
            losses.ClassAwareLocalizationDetectionBCELoss(detection_loss_weight=1.0,
                                                          manipulated_loss_weight=1.0,
                                                          authentic_loss_weight=1.0)

        criterion_simple: losses.LocalizationDetectionBCELoss = \
            losses.LocalizationDetectionBCELoss()

        loc_predicted: torch.Tensor = torch.rand((8, 1, 224, 224))
        loc_actual: torch.Tensor = torch.zeros(8, 1, 224, 224)
        det_predicted: torch.Tensor = torch.rand((8, 2))
        det_actual: torch.Tensor = torch.zeros(8, 2)

        loss: torch.Tensor = criterion(loc_predicted, det_predicted,
                                       loc_actual.float(), det_actual.float())

        loss_simple: torch.Tensor = criterion_simple(loc_predicted, det_predicted,
                                                     loc_actual.float(), det_actual.float())

        self.assertAlmostEqual(float(loss_simple.numpy()), float(loss.numpy()), delta=1e-5)

    def test_equality_with_simple_localization_detection_loss_only_manipulated(self) -> None:
        criterion: losses.ClassAwareLocalizationDetectionBCELoss = \
            losses.ClassAwareLocalizationDetectionBCELoss(detection_loss_weight=1.0,
                                                          manipulated_loss_weight=1.0,
                                                          authentic_loss_weight=1.0)

        criterion_simple: losses.LocalizationDetectionBCELoss = \
            losses.LocalizationDetectionBCELoss()

        loc_predicted: torch.Tensor = torch.rand((8, 1, 224, 224))
        loc_actual: torch.Tensor = torch.ones(8, 1, 224, 224)
        det_predicted: torch.Tensor = torch.rand((8, 2))
        det_actual: torch.Tensor = torch.ones(8, 2)

        loss: torch.Tensor = criterion(loc_predicted, det_predicted,
                                       loc_actual.float(), det_actual.float())

        loss_simple: torch.Tensor = criterion_simple(loc_predicted, det_predicted,
                                                     loc_actual.float(), det_actual.float())

        self.assertAlmostEqual(float(loss_simple.numpy()), float(loss.numpy()), delta=1e-5)

    def test_loss_localization_only_batch_8(self) -> None:
        criterion: losses.ClassAwareLocalizationDetectionBCELoss = \
            losses.ClassAwareLocalizationDetectionBCELoss(disable_detection_loss=True)

        loc_predicted: torch.Tensor = torch.rand((8, 1, 224, 224))
        loc_actual: torch.Tensor = torch.randint(2, (8, 1, 224, 224))
        det_predicted: torch.Tensor = torch.rand((8, 2))
        det_actual: torch.Tensor = torch.randint(2, (8, 2))

        loss: torch.Tensor = criterion(loc_predicted, det_predicted,
                                       loc_actual.float(), det_actual.float())

        self.assertEqual(torch.Size(tuple()), loss.size())

    def test_loss_localization_only_batch_1(self) -> None:
        criterion: losses.ClassAwareLocalizationDetectionBCELoss = \
            losses.ClassAwareLocalizationDetectionBCELoss(disable_detection_loss=True)

        loc_predicted: torch.Tensor = torch.rand((1, 1, 224, 224))
        loc_actual: torch.Tensor = torch.randint(2, (1, 1, 224, 224))
        det_predicted: torch.Tensor = torch.rand((1, 2))
        det_actual: torch.Tensor = torch.randint(2, (1, 2))

        loss: torch.Tensor = criterion(loc_predicted, det_predicted,
                                       loc_actual.float(), det_actual.float())

        self.assertEqual(torch.Size(tuple()), loss.size())

    def test_loss_localization_only_perfect_match(self) -> None:
        criterion: losses.ClassAwareLocalizationDetectionBCELoss = \
            losses.ClassAwareLocalizationDetectionBCELoss(disable_detection_loss=True)

        loc_predicted: torch.Tensor = torch.randint(2, (8, 1, 224, 224))
        det_predicted: torch.Tensor = torch.randint(2, (8, 2))

        loss: torch.Tensor = criterion(loc_predicted.float(), det_predicted.float(),
                                       loc_predicted.float(), det_predicted.float())

        self.assertAlmostEqual(0.0, float(loss.numpy()))

    def test_loss_localization_only_match(self) -> None:
        criterion: losses.ClassAwareLocalizationDetectionBCELoss = \
            losses.ClassAwareLocalizationDetectionBCELoss(disable_detection_loss=True)

        loc_predicted: torch.Tensor = torch.full((1, 1, 224, 224), 0.5)
        loc_actual: torch.Tensor = torch.randint(2, (1, 1, 224, 224))
        det_predicted: torch.Tensor = torch.full((1, 2), 0.5)
        det_actual: torch.Tensor = torch.randint(2, (1, 2))

        loss: torch.Tensor = criterion(loc_predicted, det_predicted,
                                       loc_actual.float(), det_actual.float())

        self.assertAlmostEqual(0.46209, float(loss.numpy()), delta=0.001)

    def test_loss_batch_8_one_hot_localization(self) -> None:
        criterion: losses.ClassAwareLocalizationDetectionBCELoss = \
            losses.ClassAwareLocalizationDetectionBCELoss()

        # Create loss inputs.
        loc_predicted: torch.Tensor = torch.rand((8, 2, 224, 224))
        loc_actual: torch.Tensor = torch.randint(2, (8, 1, 224, 224))
        loc_actual = functional.one_hot(
            einops.rearrange(loc_actual, "b c h w -> b h w c"), num_classes=2
        ).squeeze()
        loc_actual = einops.rearrange(loc_actual, "b h w c -> b c h w")
        det_predicted: torch.Tensor = torch.rand((8, 2))
        det_actual: torch.Tensor = torch.randint(2, (8, 1))
        det_actual = functional.one_hot(det_actual, num_classes=2).squeeze()

        loss: torch.Tensor = criterion(loc_predicted, det_predicted,
                                       loc_actual.float(), det_actual.float())

        self.assertEqual(torch.Size(tuple()), loss.size())

    def test_loss_batch_1_one_hot_localization(self) -> None:
        criterion: losses.ClassAwareLocalizationDetectionBCELoss = \
            losses.ClassAwareLocalizationDetectionBCELoss()

        # Create loss inputs.
        loc_predicted: torch.Tensor = torch.rand((1, 2, 224, 224))
        loc_actual: torch.Tensor = torch.randint(2, (1, 1, 224, 224))
        loc_actual = functional.one_hot(
            einops.rearrange(loc_actual, "b c h w -> b h w c"), num_classes=2
        ).squeeze(dim=3)
        loc_actual = einops.rearrange(loc_actual, "b h w c -> b c h w")
        det_predicted: torch.Tensor = torch.rand((1, 2))
        det_actual: torch.Tensor = torch.randint(2, (1, 1))
        det_actual = functional.one_hot(det_actual, num_classes=2).squeeze(dim=1)

        loss: torch.Tensor = criterion(loc_predicted, det_predicted,
                                       loc_actual.float(), det_actual.float())

        self.assertEqual(torch.Size(tuple()), loss.size())


class TestLocalizationDetectionBootstrappedBCE(unittest.TestCase):

    def test_loss_batch_8_before_warmup(self) -> None:
        criterion: losses.LocalizationDetectionBootstrappedBCE = \
            losses.LocalizationDetectionBootstrappedBCE()
        loc_predicted: torch.Tensor = torch.rand((8, 1, 224, 224))
        loc_actual: torch.Tensor = torch.randint(2, (8, 1, 224, 224))
        det_predicted: torch.Tensor = torch.rand((8, 2))
        det_actual: torch.Tensor = torch.randint(2, (8, 2))
        loss, p = criterion(loc_predicted, det_predicted,
                            loc_actual.float(), det_actual.float(),
                            it=0)
        self.assertEqual(torch.Size(tuple()), loss.size())
        self.assertAlmostEqual(p, 1.0)

    def test_loss_batch_8_during_warmup(self) -> None:
        criterion: losses.LocalizationDetectionBootstrappedBCE = \
            losses.LocalizationDetectionBootstrappedBCE()
        loc_predicted: torch.Tensor = torch.rand((8, 1, 224, 224))
        loc_actual: torch.Tensor = torch.randint(2, (8, 1, 224, 224))
        det_predicted: torch.Tensor = torch.rand((8, 2))
        det_actual: torch.Tensor = torch.randint(2, (8, 2))
        loss, p = criterion(loc_predicted, det_predicted,
                            loc_actual.float(), det_actual.float(),
                            it=30000)
        self.assertEqual(torch.Size(tuple()), loss.size())
        self.assertLess(p, 1.0)
        self.assertGreater(p, 0.15)

    def test_loss_batch_8_after_warmup(self) -> None:
        criterion: losses.LocalizationDetectionBootstrappedBCE = \
            losses.LocalizationDetectionBootstrappedBCE()
        loc_predicted: torch.Tensor = torch.rand((8, 1, 224, 224))
        loc_actual: torch.Tensor = torch.randint(2, (8, 1, 224, 224))
        det_predicted: torch.Tensor = torch.rand((8, 2))
        det_actual: torch.Tensor = torch.randint(2, (8, 2))
        loss, p = criterion(loc_predicted, det_predicted,
                            loc_actual.float(), det_actual.float(),
                            it=120000)
        self.assertEqual(torch.Size(tuple()), loss.size())
        self.assertAlmostEqual(p, 0.15)

    def test_loss_batch_8_full_cycle(self) -> None:
        criterion: losses.LocalizationDetectionBootstrappedBCE = \
            losses.LocalizationDetectionBootstrappedBCE(
                start_warm=100,
                end_warm=500,
                top_p=0.15,
            )
        loc_predicted: torch.Tensor = torch.rand((8, 1, 224, 224))
        loc_actual: torch.Tensor = torch.randint(2, (8, 1, 224, 224))
        det_predicted: torch.Tensor = torch.rand((8, 2))
        det_actual: torch.Tensor = torch.randint(2, (8, 2))

        all_p: list[float] = []
        for i in range(0, 1000):
            _, p = criterion(loc_predicted, det_predicted,
                             loc_actual.float(), det_actual.float(),
                             it=i)
            all_p.append(p)

        self.assertAlmostEqual(max(all_p), 1.0)
        self.assertAlmostEqual(min(all_p), 0.15)
        diffs = np.diff(all_p)
        self.assertTrue(np.all(diffs <= 0))

    def test_loss_batch_1_before_warmup(self) -> None:
        criterion: losses.LocalizationDetectionBootstrappedBCE = \
            losses.LocalizationDetectionBootstrappedBCE()
        loc_predicted: torch.Tensor = torch.rand((1, 1, 224, 224))
        loc_actual: torch.Tensor = torch.randint(2, (1, 1, 224, 224))
        det_predicted: torch.Tensor = torch.rand((1, 2))
        det_actual: torch.Tensor = torch.randint(2, (1, 2))
        loss, p = criterion(loc_predicted, det_predicted,
                            loc_actual.float(), det_actual.float(),
                            it=0)
        self.assertEqual(torch.Size(tuple()), loss.size())
        self.assertAlmostEqual(p, 1.0)

    def test_loss_batch_1_during_warmup(self) -> None:
        criterion: losses.LocalizationDetectionBootstrappedBCE = \
            losses.LocalizationDetectionBootstrappedBCE()
        loc_predicted: torch.Tensor = torch.rand((1, 1, 224, 224))
        loc_actual: torch.Tensor = torch.randint(2, (1, 1, 224, 224))
        det_predicted: torch.Tensor = torch.rand((1, 2))
        det_actual: torch.Tensor = torch.randint(2, (1, 2))
        loss, p = criterion(loc_predicted, det_predicted,
                                       loc_actual.float(), det_actual.float(),
                                       it=30000)
        self.assertEqual(torch.Size(tuple()), loss.size())
        self.assertLess(p, 1.0)
        self.assertGreater(p, 0.15)

    def test_loss_batch_1_after_warmup(self) -> None:
        criterion: losses.LocalizationDetectionBootstrappedBCE = \
            losses.LocalizationDetectionBootstrappedBCE()
        loc_predicted: torch.Tensor = torch.rand((1, 1, 224, 224))
        loc_actual: torch.Tensor = torch.randint(2, (1, 1, 224, 224))
        det_predicted: torch.Tensor = torch.rand((1, 2))
        det_actual: torch.Tensor = torch.randint(2, (1, 2))
        loss, p = criterion(loc_predicted, det_predicted,
                            loc_actual.float(), det_actual.float(),
                            it=120000)
        self.assertEqual(torch.Size(tuple()), loss.size())
        self.assertAlmostEqual(p, 0.15)

    def test_loss_batch_8_bce_with_logits_during_warmup(self) -> None:
        criterion: losses.LocalizationDetectionBootstrappedBCE = \
            losses.LocalizationDetectionBootstrappedBCE(
                loss_type=losses.LossType.BCE_WITH_LOGITS_LOSS
            )
        loc_predicted: torch.Tensor = torch.rand((8, 1, 224, 224))
        loc_actual: torch.Tensor = torch.randint(2, (8, 1, 224, 224))
        det_predicted: torch.Tensor = torch.rand((8, 2))
        det_actual: torch.Tensor = torch.randint(2, (8, 2))
        loss, p = criterion(loc_predicted, det_predicted,
                            loc_actual.float(), det_actual.float(),
                            it=30000)
        self.assertEqual(torch.Size(tuple()), loss.size())
        self.assertLess(p, 1.0)
        self.assertGreater(p, 0.15)

    def test_loss_batch_1_bce_with_logits_during_warmup(self) -> None:
        criterion: losses.LocalizationDetectionBootstrappedBCE = \
            losses.LocalizationDetectionBootstrappedBCE(
                loss_type=losses.LossType.BCE_WITH_LOGITS_LOSS
            )
        loc_predicted: torch.Tensor = torch.rand((1, 1, 224, 224))
        loc_actual: torch.Tensor = torch.randint(2, (1, 1, 224, 224))
        det_predicted: torch.Tensor = torch.rand((1, 2))
        det_actual: torch.Tensor = torch.randint(2, (1, 2))
        loss, p = criterion(loc_predicted, det_predicted,
                            loc_actual.float(), det_actual.float(),
                            it=30000)
        self.assertEqual(torch.Size(tuple()), loss.size())
        self.assertLess(p, 1.0)
        self.assertGreater(p, 0.15)


class TestLocalizationDetectionBCEDiceLoss(unittest.TestCase):

    def test_loss(self) -> None:
        batch_sizes: list[int] = [1, 5, 8]

        for b in batch_sizes:
            criterion = losses.LocalizationDetectionBCEDiceLoss()

            loc_predicted: torch.Tensor = torch.rand((b, 1, 224, 224))
            loc_actual: torch.Tensor = torch.randint(2, (b, 1, 224, 224))
            det_predicted: torch.Tensor = torch.rand((b, 2))
            det_actual: torch.Tensor = torch.randint(2, (b, 2))

            loss: torch.Tensor = criterion(loc_predicted, det_predicted,
                                           loc_actual.float(), det_actual.float())

            self.assertEqual(torch.Size(tuple()), loss.size())


class TestClassAwareLocalizationDetectionBCEDiceLoss(unittest.TestCase):

    def test_loss(self) -> None:
        batch_sizes: list[int] = [1, 5, 8]

        for b in batch_sizes:
            criterion = losses.ClassAwareLocalizationDetectionBCEDiceLoss()

            loc_predicted: torch.Tensor = torch.rand((b, 1, 224, 224))
            loc_actual: torch.Tensor = torch.randint(2, (b, 1, 224, 224))
            det_predicted: torch.Tensor = torch.rand((b, 2))
            det_actual: torch.Tensor = torch.randint(2, (b, 2))

            loss: torch.Tensor = criterion(loc_predicted, det_predicted,
                                           loc_actual.float(), det_actual.float())

            self.assertEqual(torch.Size(tuple()), loss.size())


class TestDiceLossSmooth(unittest.TestCase):

    def test_loss(self) -> None:
        batch_sizes: list[int] = [1, 5, 8]

        for b in batch_sizes:
            criterion = losses.DiceLossSmooth()

            loc_predicted: torch.Tensor = torch.rand((b, 1, 224, 224))
            loc_actual: torch.Tensor = torch.randint(2, (b, 1, 224, 224))

            loss: torch.Tensor = criterion(loc_predicted, loc_actual.float())

            self.assertEqual(torch.Size(tuple()), loss.size())
            self.assertTrue(torch.all(loss >= .0))
            self.assertTrue(torch.all(loss <= 1.))
