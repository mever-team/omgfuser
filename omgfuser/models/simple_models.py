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

import math

import einops
import torch
from torch import nn

from omgfuser.models.fusion import (
    BackboneType,
    UpscalerType,
    DetectionHeadType,
    _create_localization_head,
    _init_weights,
    _create_detection_head
)
from omgfuser.models import backbones


class TransformerModel(nn.Module):

    def __init__(self,
                 height: int,
                 width: int,
                 image_channels: int = 3,
                 dropout: float = 0.,
                 upscaler_type: UpscalerType = UpscalerType.CONV_UPSCALER,
                 detection_head_type: DetectionHeadType = DetectionHeadType.ONE_HOT_OUT,
                 add_final_logit_layers: bool = True,
                 image_backbone_type: BackboneType = BackboneType.CONVOLUTIONAL,
                 cnn_group_norm: bool = False):
        """Creates a model for predicting forgery from an RGB image.

        Input to the model should be a tensor of size: (N, 3, H, W)
        Where:
            - N: Batch size.
            - 3: Image_channels.
            - H: Height of the inputs.
            - W: Width of the inputs.

        :param height: Height of the input image.
        :param width: Width of the input image
        :param image_channels: Number of channels in the image.
        :param dropout:
        :param upscaler_type: Type of the upscaler to use for the localization head.
        :param detection_head_type: Type of the detection head.
        :param add_final_logit_layers: When set to False, the detection and localization
            outputs of the model will not pass through a final sigmoid layer, so they will not
            be bounded to [0, 1]. Useful for utilizing BCEWithLogitsLoss, since BCELoss cannot
            be applied with mixed precision training.
        :param cnn_group_norm: When set to True, GroupNorm is used in convolutional layers
            instead of BatchNorm.
            Defaults to bottleneck fusion.
        """
        super().__init__()

        self.height: int = height
        self.width: int = width
        self.image_channels: int = image_channels
        self.dropout: float = dropout

        if image_backbone_type not in [
            BackboneType.DINO,
            BackboneType.DINOv2,
            BackboneType.DINOv2_PATCH_EMBED_FROZEN_FEATURE_INTERPOLATION
        ]:
            raise RuntimeError(f"Non-supported backbone: {image_backbone_type.name}")

        vertical_tokens: int = self.height // 16
        horizontal_tokens: int = self.width // 16

        # Localization head.
        self.localization_head: nn.Module = _create_localization_head(
            upscaler_type, add_final_logit_layers, cnn_group_norm
        )

        # Detection head.
        self.detection_head: nn.Module = _create_detection_head(
            detection_head_type, add_logits_layer=add_final_logit_layers
        )

        # Initialize the weights of the model.
        self.apply(_init_weights)

        # Define the pretrained backbone models after weight initialization, in order to
        # not truncate their pretrained weights.
        if image_backbone_type == BackboneType.DINO:
            self.image_feature_extractor = backbones.DINOBackbone()
        elif image_backbone_type == BackboneType.DINOv2:
            self.image_feature_extractor = backbones.DINOv2Backbone(
                target_token_size=(vertical_tokens, horizontal_tokens),
                features_interpolation=True
            )
        elif image_backbone_type == BackboneType.DINOv2_FEATURE_INTERPOLATION:
            self.image_feature_extractor = backbones.DINOv2Backbone(
                target_token_size=(vertical_tokens, horizontal_tokens)
            )
        elif image_backbone_type == BackboneType.DINOv2_FROZEN:
            self.image_feature_extractor = backbones.DINOv2Backbone(
                frozen=True, target_token_size=(vertical_tokens, horizontal_tokens)
            )
        elif image_backbone_type == BackboneType.DINOv2_FROZEN_FEATURE_INTERPOLATION:
            self.image_feature_extractor = backbones.DINOv2Backbone(
                frozen=True, features_interpolation=True,
                target_token_size=(vertical_tokens, horizontal_tokens)
            )
        elif image_backbone_type == BackboneType.DINOv2_FROZEN_BILINEAR_FEAT_INT:
            self.image_feature_extractor = backbones.DINOv2Backbone(
                frozen=True, features_interpolation=True,
                features_interpolation_type=backbones.DINOv2FeaturesInterpolationType.BILINEAR,
                target_token_size=(vertical_tokens, horizontal_tokens)
            )
        elif image_backbone_type == BackboneType.DINOv2_FROZEN_PENULTIMATE_LAYER:
            self.image_feature_extractor = backbones.DINOv2Backbone(
                frozen=True, features_layer=2,
                target_token_size=(vertical_tokens, horizontal_tokens)
            )
        elif (image_backbone_type
                == BackboneType.DINOv2_FROZEN_PENULTIMATE_LAYER_FEATURE_INTERPOLATION):
            self.image_feature_extractor = backbones.DINOv2Backbone(
                frozen=True, features_layer=2, features_interpolation=True,
                target_token_size=(vertical_tokens, horizontal_tokens)
            )
        elif image_backbone_type == BackboneType.DINOv2_FROZEN_MULTISCALE:
            self.image_feature_extractor = backbones.DINOv2Backbone(
                frozen=True, features_layer=[2, 5, 8, 11],
                target_token_size=(vertical_tokens, horizontal_tokens)
            )
        elif image_backbone_type == BackboneType.DINOv2_FROZEN_MULTISCALE_FEATURE_INTERPOLATION:
            self.image_feature_extractor = backbones.DINOv2Backbone(
                frozen=True, features_interpolation=True, features_layer=[2, 5, 8, 11],
                target_token_size=(vertical_tokens, horizontal_tokens)
            )
        elif image_backbone_type == BackboneType.DINOv2_FROZEN_MULTISCALE_BILINEAR_FEAT_INT:
            self.image_feature_extractor = backbones.DINOv2Backbone(
                frozen=True, features_interpolation=True, features_layer=[2, 5, 8, 11],
                features_interpolation_type=backbones.DINOv2FeaturesInterpolationType.BILINEAR,
                target_token_size=(vertical_tokens, horizontal_tokens)
            )
        elif image_backbone_type == BackboneType.DINOv2_PATCH_EMBED_FROZEN_FEATURE_INTERPOLATION:
            self.image_feature_extractor = backbones.DINOv2Backbone(
                frozen=True,
                freeze_only_patch_embed=True,
                features_interpolation=True,
                target_token_size=(vertical_tokens, horizontal_tokens)
            )

    def forward(self, x: torch.Tensor) -> dict[str: torch.Tensor]:
        """Predicts forgery localization map and detection score from an RGB image.

        :param x: A tensor of size: (N, 3, H, W)
            Where:
                - N: Batch size.
                - 3: image_channels
                - H: Height of the image.
                - W: Width of the image.
            Currently only square images are supported, thus H == W!

        :return: A dict containing the following keys:
            'localization': A tensor of size (N, 1, H, W) containing the localization map.
            'detection': A tensor of size (N, 1) or (N, 2) containing the output of
                binary detection. The size of the output depends on whether single-value
                or one-hot detection head has been selected.
        """
        # Extract image features.
        x = self.image_feature_extractor(x)

        # Output localization map and binary detection.
        x = einops.rearrange(x, "b (h w) c -> b c h w", h=int(math.sqrt(x.size(dim=1))))
        localization_map: torch.Tensor = self.localization_head(x)
        binary_detection: torch.Tensor = self.detection_head(x)
        out: dict[str, torch.Tensor] = {
            "localization": localization_map,
            "detection": binary_detection
        }

        return out
