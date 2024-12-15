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

import functools
from enum import Enum
from typing import Union

import torch
from torch import nn
# from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

import einops


class DINOv2FeaturesInterpolationType(Enum):
    CONVOLUTIONAL = "conv"
    BILINEAR = "bilinear"


class ConvolutionalBackbone(nn.Module):
    """Feature extractor that uses 5 stacked convolutional layers."""

    def __init__(self, c_in: int, group_norm: bool = False):
        super().__init__()

        self.feature_extractor: nn.Module = nn.Sequential(
            # Conv layer 1.
            nn.Conv2d(c_in, out_channels=48, kernel_size=3, stride=2,
                      padding=1, padding_mode="circular"),
            nn.GroupNorm(16, 48) if group_norm else nn.BatchNorm2d(48),
            nn.ReLU(),
            # Conv layer 2.
            nn.Conv2d(48, out_channels=96, kernel_size=3, stride=2,
                      padding=1, padding_mode="circular"),
            nn.GroupNorm(16, 96) if group_norm else nn.BatchNorm2d(96),
            nn.ReLU(),
            # Conv layer 3.
            nn.Conv2d(96, out_channels=192, kernel_size=3, stride=2,
                      padding=1, padding_mode="circular"),
            nn.GroupNorm(16, 192) if group_norm else nn.BatchNorm2d(192),
            nn.ReLU(),
            # Conv layer 4.
            nn.Conv2d(192, out_channels=384, kernel_size=3, stride=2,
                      padding=1, padding_mode="circular"),
            nn.GroupNorm(16, 384) if group_norm else nn.BatchNorm2d(384),
            nn.ReLU(),
            # Conv layer 5.
            nn.Conv2d(384, out_channels=384, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)


class ConvolutionalBackboneHighDim(nn.Module):
    """Feature extractor that uses 5 stacked convolutional layers."""

    def __init__(self, c_in: int, group_norm: bool = False):
        super().__init__()

        self.feature_extractor: nn.Module = nn.Sequential(
            # Conv layer 1.
            nn.Conv2d(c_in, out_channels=672, kernel_size=3, stride=2,
                      padding=1, padding_mode="circular"),
            nn.GroupNorm(16, 672) if group_norm else nn.BatchNorm2d(672),
            nn.ReLU(),
            # Conv layer 2.
            nn.Conv2d(672, out_channels=672, kernel_size=3, stride=2,
                      padding=1, padding_mode="circular"),
            nn.GroupNorm(16, 672) if group_norm else nn.BatchNorm2d(672),
            nn.ReLU(),
            # Conv layer 3.
            nn.Conv2d(672, out_channels=528, kernel_size=3, stride=2,
                      padding=1, padding_mode="circular"),
            nn.GroupNorm(16, 528) if group_norm else nn.BatchNorm2d(528),
            nn.ReLU(),
            # Conv layer 4.
            nn.Conv2d(528, out_channels=384, kernel_size=3, stride=2,
                      padding=1, padding_mode="circular"),
            nn.GroupNorm(16, 384) if group_norm else nn.BatchNorm2d(384),
            nn.ReLU(),
            # Conv layer 5.
            nn.Conv2d(384, out_channels=384, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)


# class Backbone(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
#         self.backbone_network = efficientnet_v2_s(weights=weights)
#         self.transforms = weights.transforms()
#
#     def forward(self, x):
#         y = self.transforms(x)
#         y = self.backbone_network(y)
#
#         return y


class DINOBackbone(nn.Module):

    def __init__(self):
        super().__init__()

        self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts features utilizing the DINO backbone.

        :param x: A tensor of size (B, C=3, H, W) where:
            - B: Batch size.
            - C: Number of image channels. Image should be RGB, so C should be equal to 3.
            - H: Image height.
            - W: Image width.
        :return: A tensor of size (B, H*W, 384)
        """
        x = self.dino.get_intermediate_layers(x)
        x = x[0][:, 1:, :]
        return x


class DINOv2Backbone(nn.Module):

    def __init__(
        self,
        frozen: bool = False,
        features_layer: Union[int, list[int]] = 1,
        features_interpolation: bool = False,
        features_interpolation_type: DINOv2FeaturesInterpolationType =
                DINOv2FeaturesInterpolationType.CONVOLUTIONAL,
        target_token_size: tuple[int, int] = (14, 14),
        freeze_only_patch_embed: bool = False
    ):
        """Initializes a pretrained DINOv2 Vit-S\14 based backbone.

        :param frozen: If set to True, the gradients for parameters of the model will
            be disabled.
        :param features_layer: An integer or a list that defines the DINOv2 layers
            whose features will be returned. When a single integer is provided, the counting
            begins from the last layer. So, `1` means the features of the last layer,
            `2` the features of the penultimate layer and so on. When a list of integers
            is provided, the counting refers to the actual numbers of layers in DINOv2,
            starting from 0 and ending to 11. In such case, a list of tensors is returned,
            containing the features of the corresponding layers. Defaults to 1.
        :param features_interpolation: By default, the input image is interpolated, in order
            the output size to match the output size of DINOv1, that was using a patch size of 16
            pixels instead of 14. When this flag is provided, the output is transformed through
            a convolutional layer, instead of the input.
        :param features_interpolation_type: Specifies the type of features interpolation to
            be performed when `features_interpolation` flag is set to True.
        :param target_token_size: The height and width of the tokenized image.
        :param freeze_only_patch_embed: When this flag is set to True and `frozen` is also set
            to True, only the embedding layer of DINOv2 will be frozen.
        """
        super().__init__()

        # Initialize DINOv2 Vit-s\14 pretrained weights.
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        if frozen:
            if freeze_only_patch_embed:
                for p in self.dino.patch_embed.parameters():
                    p.requires_grad = False
            else:
                for p in self.dino.parameters():
                    p.requires_grad = False

        # Select the layers whose features will be extracted.
        if isinstance(features_layer, int):
            # When a single int is provided, counting starts from the last layer.
            self.features_layers: list[int] = [12-features_layer]
        elif isinstance(features_layer, list):
            self.features_layers: list[int] = features_layer
        else:
            raise RuntimeError(f"{type(features_layer)} is an unsupported type for features_layer")

        self.target_token_size: tuple[int, int] = target_token_size

        # When features interpolation is requested, add a convolutional layer for each layer of
        # DINOv2 whose features are extracted.
        self.features_interpolation: bool = features_interpolation
        if self.features_interpolation:
            if features_interpolation_type == DINOv2FeaturesInterpolationType.CONVOLUTIONAL:
                self.interpolators: nn.ModuleList = nn.ModuleList([
                    nn.Conv2d(384, out_channels=384, kernel_size=1+(16-(target_token_size[0]%16)))
                    for _ in range(len(self.features_layers))
                ])
                for conv_layer in self.interpolators:
                    nn.init.xavier_uniform_(conv_layer.weight)
                    nn.init.zeros_(conv_layer.bias)
            elif features_interpolation_type == DINOv2FeaturesInterpolationType.BILINEAR:
                self.interpolators: list = [
                    functools.partial(nn.functional.interpolate,
                                      size=(target_token_size[0], target_token_size[1]),
                                      mode="bilinear")
                    for _ in range(len(self.features_layers))
                ]
            else:
                raise RuntimeError("Unsupported features interpolation method: {}".format(
                    features_interpolation_type.value
                ))

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Extracts features utilizing the DINOv2 Vit-S\14 backbone.

        :param x: A tensor of size (B, C=3, H, W) where:
            - B: Batch size.
            - C: Number of image channels. Image should be RGB, so C should be equal to 3.
            - H: Image height.
            - W: Image width.
        :return: A tensor of size (B, H*W, 384)
        """
        # Keep height for reshaping output when using features interpolation.
        height: int = x.size(dim=2)

        if not self.features_interpolation:
            x = torch.nn.functional.interpolate(
                x, (self.target_token_size[0]*14, self.target_token_size[1]*14), mode="bilinear"
            )

        x: tuple[torch.Tensor] = self.dino.get_intermediate_layers(x, self.features_layers)

        if self.features_interpolation:
            # Fix the output of DINOv2 to match the output of DINOv1.
            x = tuple(
                self._interpolate_features(f, c, height) for f, c in zip(x, self.interpolators)
            )

        if len(self.features_layers) < 2:
            x: torch.Tensor = x[0]
        else:
            x: list[torch.Tensor] = list(x)

        return x

    @staticmethod
    def _interpolate_features(x: torch.Tensor, interpolator, height: int) -> torch.Tensor:
        x = einops.rearrange(x, "b (h w) d -> b d h w", h=height // 14)
        x = interpolator(x)
        x = einops.rearrange(x, "b d h w -> b (h w) d")
        return x
