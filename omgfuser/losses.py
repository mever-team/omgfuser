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

from enum import Enum
from typing import Union

import torch
from torch import nn
import einops


class LossType(Enum):
    BCE_LOSS = "bce"
    BCE_WITH_LOGITS_LOSS = "bce_with_logits"


class LocalizationDetectionBCELoss(nn.Module):

    def __init__(self,
                 detection_loss_weight: float = 1.0,
                 localization_loss_weight: float = 1.0,
                 loss_type: LossType = LossType.BCE_LOSS):
        super().__init__()

        self.detection_loss_weight: float = detection_loss_weight
        self.localization_loss_weight: float = localization_loss_weight

        self.localization_loss: Union[nn.BCELoss, nn.BCEWithLogitsLoss] = \
            _create_bce_loss(loss_type)
        self.detection_loss: Union[nn.BCELoss, nn.BCEWithLogitsLoss] = \
            _create_bce_loss(loss_type)

    def forward(self,
                predicted_map: torch.Tensor,
                predicted_class: torch.Tensor,
                actual_map: torch.Tensor,
                actual_class: torch.Tensor) -> torch.Tensor:
        loc_loss = self.localization_loss(predicted_map, actual_map)
        det_loss = self.detection_loss(predicted_class, actual_class)
        loss = loc_loss * self.localization_loss_weight + det_loss * self.detection_loss_weight
        return loss


class ClassAwareLocalizationDetectionBCELoss(nn.Module):

    def __init__(self,
                 detection_loss_weight: float = 1.0 / 3,
                 manipulated_loss_weight: float = 1.0 / 3,
                 authentic_loss_weight: float = 1.0 / 3,
                 mask_threshold: float = 0.5,
                 disable_detection_loss: bool = False,
                 loss_type: LossType = LossType.BCE_LOSS):
        super().__init__()
        self.mask_threshold: float = mask_threshold
        self.detection_loss_weight: float = detection_loss_weight
        self.manipulated_loss_weight: float = manipulated_loss_weight
        self.authentic_loss_weight: float = authentic_loss_weight
        self.disable_detection_loss: bool = disable_detection_loss

        self.localization_manipulated_loss: Union[nn.BCELoss, nn.BCEWithLogitsLoss] = \
            _create_bce_loss(loss_type)
        self.localization_authentic_loss: Union[nn.BCELoss, nn.BCEWithLogitsLoss] = \
            _create_bce_loss(loss_type)
        self.detection_loss: Union[nn.BCELoss, nn.BCEWithLogitsLoss] = \
            _create_bce_loss(loss_type)

    def forward(self,
                predicted_map: torch.Tensor,
                predicted_class: torch.Tensor,
                actual_map: torch.Tensor,
                actual_class: torch.Tensor) -> torch.Tensor:

        losses: list[torch.Tensor] = []

        is_localization_one_hot: bool = predicted_map.size(dim=1) > 1
        assert actual_map.size(dim=1) == predicted_map.size(dim=1), \
            "Number of channels in predicted map and ground truth map does not match."

        # Different samples in the batch have different sizes of authentic/manipulated regions.
        # So, when splitting these regions into different tensors, it is impossible to create
        # a batch tensor with a size that works for all of them.
        for i in range(predicted_map.size(dim=0)):
            # Extract each sample from the mini-batch.
            sample_predicted_class: torch.Tensor = predicted_class[i, :]
            sample_actual_class: torch.Tensor = actual_class[i, :]
            sample_predicted_map: torch.Tensor = predicted_map[i, :, :, :]
            sample_actual_map: torch.Tensor = actual_map[i, :, :, :]

            # Compute detection loss.
            det_los = self.detection_loss(sample_predicted_class, sample_actual_class)
            if not self.disable_detection_loss:
                loss: torch.Tensor = det_los * self.detection_loss_weight
            else:
                loss: torch.Tensor = det_los * .0

            # Compute localization loss.
            if is_localization_one_hot:
                loss = self._compute_one_hot_loss(sample_predicted_map, sample_actual_map, loss)
            else:
                loss = self._compute_single_value_loss(sample_predicted_map, sample_actual_map,
                                                       loss)

            losses.append(loss)

        total_loss: torch.Tensor = torch.stack(losses)
        total_loss = torch.mean(total_loss)

        return total_loss

    def _compute_single_value_loss(self,
                                   sample_predicted_map: torch.Tensor,
                                   sample_actual_map: torch.Tensor,
                                   loss: torch.Tensor) -> torch.Tensor:
        # Flatten predicted and ground-truth localization maps.
        sample_predicted_map = torch.flatten(sample_predicted_map)
        sample_actual_map = torch.flatten(sample_actual_map)

        # Binarize ground-truth map to be 100% sure that it contains exactly two values.
        sample_actual_map = torch.where(sample_actual_map > self.mask_threshold, 1.0, 0.0)

        manipulated_indices: torch.Tensor = torch.argwhere(sample_actual_map)
        authentic_indices: torch.Tensor = torch.argwhere(1 - sample_actual_map)

        # Compute loss of the manipulated region.
        manipulated_actual_map: torch.Tensor = torch.transpose(
            torch.take(sample_actual_map, manipulated_indices), 0, 1)
        if manipulated_actual_map.nelement() > 0:
            manipulated_predicted_map: torch.Tensor = torch.transpose(
                torch.take(sample_predicted_map, manipulated_indices), 0, 1)

            loc_manipulated_loss = self.localization_manipulated_loss(manipulated_predicted_map,
                                                                      manipulated_actual_map)
            loss += loc_manipulated_loss * self.manipulated_loss_weight

        # Compute loss of the authentic region.
        authentic_actual_map: torch.Tensor = torch.transpose(
            torch.take(sample_actual_map, authentic_indices), 0, 1)
        if authentic_actual_map.nelement() > 0:
            authentic_predicted_map: torch.Tensor = torch.transpose(
                torch.take(sample_predicted_map, authentic_indices), 0, 1)
            loc_authentic_loss = self.localization_authentic_loss(authentic_predicted_map,
                                                                  authentic_actual_map)
            loss += loc_authentic_loss * self.authentic_loss_weight

        return loss

    def _compute_one_hot_loss(self,
                              sample_predicted_map: torch.Tensor,
                              sample_actual_map: torch.Tensor,
                              loss: torch.Tensor) -> torch.Tensor:

        single_value_actual_map = torch.argmax(sample_actual_map, dim=0, keepdim=True)
        single_value_actual_map = single_value_actual_map.flatten()

        # Convert 2D image shape to 1D.
        sample_predicted_map = einops.rearrange(sample_predicted_map, "v h w -> (h w) v")
        sample_actual_map = einops.rearrange(sample_actual_map, "v h w -> (h w) v")

        manipulated_indices: torch.Tensor = torch.argwhere(single_value_actual_map).squeeze(dim=1)
        authentic_indices: torch.Tensor = torch.argwhere(1 - single_value_actual_map).squeeze(dim=1)

        # Compute loss of the manipulated region.
        manipulated_actual_map: torch.Tensor = torch.index_select(
            sample_actual_map, dim=0, index=manipulated_indices
        )
        if manipulated_actual_map.nelement() > 0:
            manipulated_predicted_map: torch.Tensor = torch.index_select(
                sample_predicted_map, dim=0, index=manipulated_indices
            )

            loc_manipulated_loss = self.localization_manipulated_loss(manipulated_predicted_map,
                                                                      manipulated_actual_map)
            loss += loc_manipulated_loss * self.manipulated_loss_weight

        # Compute loss of the authentic region.
        authentic_actual_map: torch.Tensor = torch.index_select(
            sample_actual_map, dim=0, index=authentic_indices
        )
        if authentic_actual_map.nelement() > 0:
            authentic_predicted_map: torch.Tensor = torch.index_select(
                sample_predicted_map, dim=0, index=authentic_indices
            )
            loc_authentic_loss = self.localization_authentic_loss(authentic_predicted_map,
                                                                  authentic_actual_map)
            loss += loc_authentic_loss * self.authentic_loss_weight

        return loss


# Inspired from code in:
#    https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
class LocalizationDetectionBootstrappedBCE(nn.Module):

    def __init__(
        self,
        start_warm: int = 20000,
        end_warm: int = 70000,
        top_p: float = 0.15,
        detection_loss_weight: float = 1.0 / 2,
        localization_loss_weight: float = 1.0 / 2,
        loss_type: LossType = LossType.BCE_LOSS
    ):
        super().__init__()

        self.start_warm: int = start_warm
        self.end_warm: int = end_warm
        self.top_p: float = top_p
        self.detection_loss_weight: float = detection_loss_weight
        self.localization_loss_weight: float = localization_loss_weight

        self.localization_loss: Union[nn.BCELoss, nn.BCEWithLogitsLoss] = \
            _create_bce_loss(loss_type, reduction="none")
        self.detection_loss: Union[nn.BCELoss, nn.BCEWithLogitsLoss] = \
            _create_bce_loss(loss_type)

    def forward(self,
                predicted_map: torch.Tensor,
                predicted_class: torch.Tensor,
                actual_map: torch.Tensor,
                actual_class: torch.Tensor,
                it: int = 0) -> tuple[torch.Tensor, float]:

        det_loss = self.detection_loss(predicted_class, actual_class) * self.detection_loss_weight

        loc_loss_raw = self.localization_loss(predicted_map, actual_map).view(-1)
        this_p: float = 1.0
        if it >= self.start_warm:
            num_pixels = loc_loss_raw.numel()

            if it > self.end_warm:
                this_p: float = self.top_p
            else:
                this_p: float = self.top_p + (1 - self.top_p) * (
                        (self.end_warm-it)/(self.end_warm-self.start_warm))
            loc_loss_raw, _ = torch.topk(loc_loss_raw, int(num_pixels * this_p), sorted=False)
        loc_loss = loc_loss_raw.mean() * self.localization_loss_weight

        loss = loc_loss + det_loss
        return loss, this_p


class LocalizationDetectionBCEDiceLoss(nn.Module):
    def __init__(self,
                 detection_loss_weight: float = 0.2,
                 localization_bce_loss_weight: float = 0.2,
                 dice_loss_weight: float = 0.6,
                 loss_type: LossType = LossType.BCE_LOSS):
        super().__init__()

        self.localization_detection_bce = LocalizationDetectionBCELoss(
            detection_loss_weight=detection_loss_weight,
            localization_loss_weight=localization_bce_loss_weight,
            loss_type=loss_type
        )
        self.dice = DiceLossSmooth()
        self.dice_loss_weight: float = dice_loss_weight

    def forward(
            self,
            predicted_map: torch.Tensor,
            predicted_class: torch.Tensor,
            actual_map: torch.Tensor,
            actual_class: torch.Tensor
    ) -> torch.Tensor:
        dice_loss: torch.Tensor = self.dice(predicted_map, actual_map)
        dice_loss = dice_loss * self.dice_loss_weight

        bce_loss: torch.Tensor = self.localization_detection_bce(
            predicted_map, predicted_class,
            actual_map, actual_class
        )

        return dice_loss + bce_loss


class ClassAwareLocalizationDetectionBCEDiceLoss(nn.Module):
    def __init__(self,
                 detection_loss_weight: float = 0.2,
                 manipulated_bce_loss_weight: float = 0.15,
                 authentic_bce_loss_weight: float = 0.15,
                 dice_loss_weight: float = 0.5,
                 bce_type: LossType = LossType.BCE_LOSS):
        super().__init__()

        self.class_aware_localization_detection_bce = ClassAwareLocalizationDetectionBCELoss(
            detection_loss_weight=detection_loss_weight,
            manipulated_loss_weight=manipulated_bce_loss_weight,
            authentic_loss_weight=authentic_bce_loss_weight,
            loss_type=bce_type
        )
        self.dice = DiceLossSmooth()
        self.dice_loss_weight: float = dice_loss_weight

    def forward(
        self,
        predicted_map: torch.Tensor,
        predicted_class: torch.Tensor,
        actual_map: torch.Tensor,
        actual_class: torch.Tensor
    ) -> torch.Tensor:
        dice_loss: torch.Tensor = self.dice(predicted_map, actual_map)
        dice_loss = dice_loss * self.dice_loss_weight

        bce_loss: torch.Tensor = self.class_aware_localization_detection_bce(
            predicted_map, predicted_class,
            actual_map, actual_class
        )

        return dice_loss + bce_loss


class DiceLossSmooth(nn.Module):
    """Code adaptation from:
        https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
    """
    def __init__(self, smooth: float = 1.):
        super().__init__()
        self.smooth: float = smooth

    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor, ) -> torch.Tensor:
        # Flatten label and prediction tensors.
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection: torch.Tensor = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice


def _create_bce_loss(loss_type: LossType,
                     reduction: str = "mean") -> Union[nn.BCELoss, nn.BCEWithLogitsLoss]:
    if loss_type == loss_type.BCE_LOSS:
        return nn.BCELoss(reduction=reduction)
    elif loss_type == loss_type.BCE_WITH_LOGITS_LOSS:
        return nn.BCEWithLogitsLoss(reduction=reduction)
    else:
        raise RuntimeError(f"{loss_type.value} is not a valid loss type.")
