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
import enum
from typing import Union

import einops
import timm
import torch
from torch import nn

from .transformer import Transformer
from .backbones import ConvolutionalBackbone
from .upscalers import (
    ConvolutionalUpscaler,
    DoubleConvolutionalUpscaler,
    DoubleConvolutionalUpscaler2,
    ResidualUpscaler
)
from .cape import CAPE2d
from . import detectors
from . import backbones


class UpscalerType(enum.Enum):
    CONV_UPSCALER = "conv"
    DOUBLE_CONV_UPSCALER = "double_conv"
    DOUBLE_CONV_UPSCALER_2 = "double_conv_2"
    DOUBLE_CONV_UPSCALER_ONE_HOT = "double_conv_one_hot"
    RESIDUAL_UPSCALER = "residual"


class DetectionHeadType(enum.Enum):
    ONE_HOT_OUT = "one_hot_out"
    SINGLE_OUT = "single_out"
    TRANSFORMER_TWO_OUT_SOFTMAX = "transformer_two_out_softmax"
    TRANSFORMER_TWO_OUT_SIGMOID = "transformer_two_out_sigmoid"
    TRANSFORMER_SINGLE_OUT_SIGMOID = "transformer_single_out_sigmoid"
    TRANSFORMER_ONE_LAYER_MLP_SINGLE_OUT_SIGMOID = "transformer_one_layer_mlp_single_out_sigmoid"
    TRANSFORMER_LITE_ONE_LAYER_MLP_SINGLE_OUT_SIGMOID = "transf_lt_one_layer_mlp_single_out_sigmoid"


class PositionalEmbeddingsType(enum.Enum):
    SINUSOIDAL = "sinusoidal"
    CAPE = "cape"


class FusionType(enum.Enum):
    BOTTLENECK_FUSION = "bottleneck"
    POSITIONAL_FUSION = "positional"
    POSITIONAL_FUSION_NO_TFT = "positional_no_tft"
    POSITIONAL_FUSION_NO_LDT = "positional_no_ldt"


class BackboneType(enum.Enum):
    NONE = "none"
    CONVOLUTIONAL = "conv"
    DINO = "dino"
    DINOv2 = "dinov2"
    DINOv2_FEATURE_INTERPOLATION = "dinov2_feat_int"
    DINOv2_FROZEN = "dinov2_frozen"
    DINOv2_FROZEN_FEATURE_INTERPOLATION = "dinov2_frozen_feat_int"
    DINOv2_FROZEN_BILINEAR_FEAT_INT = "dinov2_frozen_bilinear_feat_int"
    DINOv2_FROZEN_PENULTIMATE_LAYER = "dinov2_frozen_penultimate"
    DINOv2_FROZEN_PENULTIMATE_LAYER_FEATURE_INTERPOLATION = "dinov2_frozen_penultimate_feat_int"
    DINOv2_FROZEN_MULTISCALE = "dinov2_frozen_multiscale"
    DINOv2_FROZEN_MULTISCALE_FEATURE_INTERPOLATION = "dinov2_frozen_multiscale_feat_int"
    DINOv2_FROZEN_MULTISCALE_BILINEAR_FEAT_INT = "dinov2_frozen_multiscaler_bilinear_feat_int"
    DINOv2_PATCH_EMBED_FROZEN_FEATURE_INTERPOLATION = "dinov2_patch_embed_frozen_feat_int"


class FusionModel(nn.Module):

    def __init__(self,
                 inputs_num: int,
                 first_stage_depth: int,
                 first_stage_heads: int,
                 second_stage_depth: int,
                 second_stage_heads: int,
                 height: int,
                 width: int,
                 mask_channels: int = 1,
                 map_channels: int = 3,
                 dropout: float = 0.):
        """Creates a model for fusing forensics signals with segmentation masks.

        :param inputs_num: Number of forensics inputs to fuse.
        :param first_stage_depth: Number of fusion layers in 1st stage.
        :param height: Height of the input masks.
        :param width: Width of the input masks.
        :param mask_channels: Number of channels in the forensics masks.
        :param map_channels: Number of channels in the segmentation maps.
        """
        super().__init__()

        self.inputs_num: int = inputs_num
        self.first_stage_depth: int = first_stage_depth
        self.first_stage_heads: int = first_stage_heads
        self.second_stage_depth: int = second_stage_depth
        self.second_stage_heads: int = second_stage_heads
        self.height: int = height
        self.width: int = width
        self.mask_channels: int = mask_channels
        self.map_channels: int = map_channels
        self.dropout: float = dropout

        # Feature extractors.
        self.mask_feature_extractors = nn.ModuleList([
            ConvolutionalBackbone(c_in=self.mask_channels) for _ in range(self.inputs_num)
        ])
        self.seg_map_feature_extractors = nn.ModuleList([
            ConvolutionalBackbone(c_in=self.map_channels) for _ in range(self.inputs_num)
        ])

        # Positional embeddings.
        self.pos_embed = nn.Parameter(positionalencoding2d(384, 14, 14))
        self.pos_embed.requires_grad = False

        # First stage fusions.
        assert (384 % self.first_stage_heads) == 0
        self.simple_fusions = nn.ModuleList([
            BottleneckFusionTwoToOne(dim=384,
                                     depth=self.first_stage_depth,
                                     # Equal to the size of an image after feature extraction.
                                     bottleneck_units=14*14,
                                     heads=self.first_stage_heads,
                                     dim_head=384//self.first_stage_heads,
                                     mlp_dim=384*4,
                                     dropout=self.dropout)
            for _ in range(inputs_num)
        ])

        # Second stage fusion.
        assert (384 % self.second_stage_heads) == 0
        self.overall_fusion = BottleneckFusionManyToOne(
            signals=self.inputs_num,
            dim=384,
            depth=self.second_stage_depth,
            # Equal to the size of an image after feature extraction.
            bottleneck_units=14*14,
            heads=self.second_stage_heads,
            dim_head=384//self.second_stage_heads,
            mlp_dim=384*4,
            dropout=self.dropout
        )

        # Localization head.
        self.localization_head: ConvolutionalUpscaler = ConvolutionalUpscaler(c_in=384, c_out=1)

        # Detection head.
        self.detection_head: nn.Module = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=384, out_features=2),
            nn.Softmax(dim=1)
        )

        # Initialize the weights of the model.
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> dict[str: torch.Tensor]:
        """Input shape: (N, mask_num*4, H, W)"""
        # Perform feature extraction.
        mask_features: list[torch.Tensor] = []
        map_features: list[torch.Tensor] = []
        for i, (mask_fe, map_fe) in enumerate(zip(self.mask_feature_extractors,
                                                  self.seg_map_feature_extractors)):
            mask_features.append(mask_fe(x[:, i*4, :, :].unsqueeze(1)))
            map_features.append(map_fe(x[:, i*4+1:(i+1)*4, :, :]))

        # Perform first stage fusion.
        first_stage_fusions: list[torch.Tensor] = []
        for mask_f, map_f, fusion in zip(mask_features, map_features, self.simple_fusions):
            # Add sinusoidal position embeddings.
            pos_embed: torch.Tensor = self.pos_embed.unsqueeze(
                dim=0).repeat(mask_f.size(dim=0), 1, 1, 1)
            mask_f = mask_f + pos_embed
            map_f = map_f + pos_embed

            mask_f = einops.rearrange(mask_f, "b c h w -> b (h w) c").unsqueeze(dim=1)
            map_f = einops.rearrange(map_f, "b c h w -> b (h w) c").unsqueeze(dim=1)

            x = torch.cat([mask_f, map_f], dim=1)
            x = fusion(x)
            first_stage_fusions.append(x)

        # Perform second stage fusion.
        second_stage_input: torch.Tensor = torch.stack(first_stage_fusions, dim=1)
        final_features: torch.Tensor = self.overall_fusion(second_stage_input)

        # Output localization map and binary detection.
        localization_features = einops.rearrange(final_features,
                                                 "b (h w) c -> b c h w",
                                                 h=int(math.sqrt(final_features.size(dim=1))))
        localization_map: torch.Tensor = self.localization_head(localization_features)
        binary_detection: torch.Tensor = self.detection_head(localization_features)
        out: dict[str, torch.Tensor] = {
            "localization": localization_map,
            "detection": binary_detection
        }

        return out


class FusionModelWithImage(nn.Module):

    def __init__(self,
                 inputs_num: int,
                 first_stage_depth: int,
                 first_stage_heads: int,
                 second_stage_depth: int,
                 second_stage_heads: int,
                 height: int,
                 width: int,
                 mask_channels: int = 1,
                 map_channels: int = 3,
                 image_channels: int = 3,
                 dropout: float = 0.,
                 single_image_feature_extractor: bool = False,
                 single_segmentation_map_feature_extractor: bool = False,
                 upscaler_type: UpscalerType = UpscalerType.CONV_UPSCALER,
                 detection_head_type: DetectionHeadType = DetectionHeadType.ONE_HOT_OUT,
                 add_final_logit_layers: bool = True):
        """Creates a model for fusing forensics signals with segmentation masks.

        Inputs to the model should be a tensor of size:
            (N, C, H, W)
        Where:
            - N: Batch size.
            - C: image_channels + map_channels + (masks_num * mask_channels)

                For example, when fusing 5 single channel masks, using an RGB image
                and an RGB segmentation map: C = 3 + 3 + (5 * 1) = 11

                The order of inputs should be (image, segmentation_map, ...masks...)!
                Otherwise, expect weird results!
            - H: Height of the inputs.
            - W: Width of the inputs.

        :param inputs_num: Number of forensics inputs to fuse.
        :param first_stage_depth: Number of fusion layers in 1st stage.
        :param first_stage_heads:
        :param second_stage_depth: Number of fusion layers in 2nd stage.
        :param second_stage_heads:
        :param height: Height of the input masks.
        :param width: Width of the input masks.
        :param mask_channels: Number of channels in the forensics masks.
        :param map_channels: Number of channels in the segmentation maps.
        :param image_channels: Number of channels in the image.
        :param dropout:
        :param single_image_feature_extractor: If set to True, a single feature extraction
            stage will be used for the image. Otherwise, a different feature extractor
            will be utilized for every input signal.
        :param single_segmentation_map_feature_extractor: If set to True, a single feature
            extraction stage will be used for the segmentation map. Otherwise, a different
            feature extractor will be utilized for every input signal.
        :param upscaler_type: Type of the upscaler to use for the localization head.
        :param detection_head_type: Type of the detection head.
        :param add_final_logit_layers: When set to False, the detection and localization
            outputs of the model will not pass through a final sigmoid layer, so they will not
            be bounded to [0, 1]. Useful for utilizing BCEWithLogitsLoss, since BCELoss cannot
            be applied with mixed precision training.
        """
        super().__init__()

        self.inputs_num: int = inputs_num
        self.first_stage_depth: int = first_stage_depth
        self.first_stage_heads: int = first_stage_heads
        self.second_stage_depth: int = second_stage_depth
        self.second_stage_heads: int = second_stage_heads
        self.height: int = height
        self.width: int = width
        self.mask_channels: int = mask_channels
        self.map_channels: int = map_channels
        self.image_channels: int = image_channels
        self.dropout: float = dropout
        self.single_image_feature_extractor: bool = single_image_feature_extractor
        self.single_segmentation_map_feature_extractor: bool = \
            single_segmentation_map_feature_extractor

        # Feature extractors.
        self.mask_feature_extractors = nn.ModuleList([
            ConvolutionalBackbone(c_in=self.mask_channels) for _ in range(self.inputs_num)
        ])
        if self.single_segmentation_map_feature_extractor:
            self.seg_map_feature_extractors = ConvolutionalBackbone(c_in=self.map_channels)
        else:
            self.seg_map_feature_extractors = nn.ModuleList([
                ConvolutionalBackbone(c_in=self.map_channels) for _ in range(self.inputs_num)
            ])
        if self.single_image_feature_extractor:
            self.image_feature_extractors = ConvolutionalBackbone(c_in=self.image_channels)
        else:
            self.image_feature_extractors = nn.ModuleList([
                ConvolutionalBackbone(c_in=self.image_channels) for _ in range(self.inputs_num)
            ])

        # Positional embeddings.
        self.pos_embed = nn.Parameter(positionalencoding2d(384, 14, 14))
        self.pos_embed.requires_grad = False

        # First stage fusions.
        assert (384 % self.first_stage_heads) == 0
        self.simple_fusions = nn.ModuleList([
            BottleneckFusionManyToOne(signals=3,
                                      dim=384,
                                      depth=self.first_stage_depth,
                                      # Equal to the size of an image after feature extraction.
                                      bottleneck_units=14*14,
                                      heads=self.first_stage_heads,
                                      dim_head=384//self.first_stage_heads,
                                      mlp_dim=384*4,
                                      dropout=self.dropout)
            for _ in range(inputs_num)
        ])

        # Second stage fusion.
        assert (384 % self.second_stage_heads) == 0
        self.overall_fusion = BottleneckFusionManyToOne(
            signals=self.inputs_num,
            dim=384,
            depth=self.second_stage_depth,
            # Equal to the size of an image after feature extraction.
            bottleneck_units=14*14,
            heads=self.second_stage_heads,
            dim_head=384//self.second_stage_heads,
            mlp_dim=384*4,
            dropout=self.dropout
        )

        # Localization head.
        if upscaler_type == UpscalerType.CONV_UPSCALER:
            self.localization_head: ConvolutionalUpscaler = ConvolutionalUpscaler(
                c_in=384, c_out=1, add_final_sigmoid_layer=add_final_logit_layers
            )
        elif upscaler_type == UpscalerType.RESIDUAL_UPSCALER:
            self.localization_head: ResidualUpscaler = ResidualUpscaler(
                c_in=384, c_out=1, add_final_sigmoid_layer=add_final_logit_layers
            )

        # Detection head.
        self.detection_head: nn.Module = _create_detection_head(
            detection_head_type, add_logits_layer=add_final_logit_layers
        )

        # Initialize the weights of the model.
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> dict[str: torch.Tensor]:
        """Fuses an image, a segmentation map and multiple image forensics masks.

        Inputs to the model should be a tensor of size:
            (N, C, H, W)
        Where:
            - N: Batch size.
            - C: image_channels + map_channels + (masks_num * mask_channels)

                For example, when fusing 5 single channel masks, using an RGB image
                and an RGB segmentation map: C = 3 + 3 + (5 * 1) = 11

                The order of inputs should be (image, segmentation_map, ...masks...)!
                Otherwise, expect weird results!
            - H: Height of the inputs.
            - W: Width of the inputs.
        """
        # Unpack the inputs.
        image_index: int = 0
        seg_map_index: int = self.image_channels
        masks_index: int = self.image_channels + self.map_channels
        image: torch.Tensor = x[:, image_index:seg_map_index, :, :]
        segmentation_map: torch.Tensor = x[:, seg_map_index:masks_index, :, :]
        masks: torch.Tensor = x[:, masks_index:, :, :]

        # Perform feature extraction.
        mask_features: list[torch.Tensor] = []
        map_features: list[torch.Tensor] = []
        image_features: list[torch.Tensor] = []
        # Extract image features.
        if self.single_image_feature_extractor:
            image_features.extend([self.image_feature_extractors(image)] * self.inputs_num)
        else:
            for image_fe in self.image_feature_extractors:
                image_features.append(image_fe(image))
        # Extract segmentation map features.
        if self.single_segmentation_map_feature_extractor:
            map_features.extend(
                [self.seg_map_feature_extractors(segmentation_map)] * self.inputs_num)
        else:
            for map_fe in self.seg_map_feature_extractors:
                map_features.append(map_fe(segmentation_map))
        # Extract forensics masks features.
        for i, mask_fe in enumerate(self.mask_feature_extractors):
            # TODO: Currently, it does not work for self.mask_channels != 1. Fix it if needed.
            if self.mask_channels > 1:
                raise RuntimeError("Model does not support mask channels > 1")
            mask_features.append(mask_fe(masks[:, i, :, :].unsqueeze(1)))

        # Perform first stage fusion.
        first_stage_fusions: list[torch.Tensor] = []
        for image_f, mask_f, map_f, fusion in zip(image_features,
                                                  mask_features,
                                                  map_features,
                                                  self.simple_fusions):
            # Add sinusoidal position embeddings.
            pos_embed: torch.Tensor = self.pos_embed.unsqueeze(
                dim=0).repeat(mask_f.size(dim=0), 1, 1, 1)
            image_f = image_f + pos_embed
            mask_f = mask_f + pos_embed
            map_f = map_f + pos_embed

            image_f = einops.rearrange(image_f, "b c h w -> b (h w) c").unsqueeze(dim=1)
            mask_f = einops.rearrange(mask_f, "b c h w -> b (h w) c").unsqueeze(dim=1)
            map_f = einops.rearrange(map_f, "b c h w -> b (h w) c").unsqueeze(dim=1)

            fused: torch.Tensor = torch.cat([image_f, mask_f, map_f], dim=1)
            fused = fusion(fused)
            first_stage_fusions.append(fused)

        # Perform second stage fusion.
        second_stage_input: torch.Tensor = torch.stack(first_stage_fusions, dim=1)
        final_features: torch.Tensor = self.overall_fusion(second_stage_input)

        # Output localization map and binary detection.
        localization_features = einops.rearrange(final_features,
                                                 "b (h w) c -> b c h w",
                                                 h=int(math.sqrt(final_features.size(dim=1))))
        localization_map: torch.Tensor = self.localization_head(localization_features)
        binary_detection: torch.Tensor = self.detection_head(localization_features)
        out: dict[str, torch.Tensor] = {
            "localization": localization_map,
            "detection": binary_detection
        }

        return out


class MaskedAttentionFusionModel(nn.Module):

    def __init__(
        self,
        inputs_num: int,
        first_stage_depth: int,
        first_stage_heads: int,
        second_stage_depth: int,
        second_stage_heads: int,
        height: int,
        width: int,
        mask_channels: int = 1,
        dropout: float = 0.,
        upscaler_type: UpscalerType = UpscalerType.CONV_UPSCALER,
        detection_head_type: DetectionHeadType = DetectionHeadType.ONE_HOT_OUT,
        add_final_logit_layers: bool = True,
        cnn_group_norm: bool = False,
        pos_embeddings_type: PositionalEmbeddingsType = PositionalEmbeddingsType.SINUSOIDAL,
        second_stage_fusion_type: FusionType = FusionType.BOTTLENECK_FUSION
    ):
        """Creates a model for fusing forensics signals with segmentation masks.

        Inputs to the model should be two tensor of size:
            (N, C, H, W) and (N, H*W, H*W)
        Where:
            - N: Batch size.
            - C: image_channels + map_channels + (masks_num * mask_channels)

                For example, when fusing 5 single channel masks, using an RGB image
                and an RGB segmentation map: C = 3 + 3 + (5 * 1) = 11

                The order of inputs should be (image, segmentation_map, ...masks...)!
                Otherwise, expect weird results!
            - H: Height of the inputs.
            - W: Width of the inputs.

        :param inputs_num: Number of forensics inputs to fuse.
        :param first_stage_depth: Number of fusion layers in 1st stage.
        :param first_stage_heads:
        :param second_stage_depth: Number of fusion layers in 2nd stage.
        :param second_stage_heads:
        :param height: Height of the input masks.
        :param width: Width of the input masks.
        :param mask_channels: Number of channels in the forensics masks.
        :param dropout:
        :param upscaler_type: Type of the upscaler to use for the localization head.
        :param detection_head_type: Type of the detection head.
        :param add_final_logit_layers: When set to False, the detection and localization
            outputs of the model will not pass through a final sigmoid layer, so they will not
            be bounded to [0, 1]. Useful for utilizing BCEWithLogitsLoss, since BCELoss cannot
            be applied with mixed precision training.
        :param cnn_group_norm: When set to True, GroupNorm is used in convolutional layers
            instead of BatchNorm.
        :param pos_embeddings_type: The type of positional embeddings to use for encoding
            positional information to transformers.
        :param second_stage_fusion_type: Defines the type of fusion for the second stage.
            Defaults to bottleneck fusion.
        """
        super().__init__()

        self.inputs_num: int = inputs_num
        self.first_stage_depth: int = first_stage_depth
        self.first_stage_heads: int = first_stage_heads
        self.second_stage_depth: int = second_stage_depth
        self.second_stage_heads: int = second_stage_heads
        self.height: int = height
        self.width: int = width
        self.mask_channels: int = mask_channels
        self.dropout: float = dropout
        self.pos_embeddings_type: PositionalEmbeddingsType = pos_embeddings_type

        # Feature extractors.
        self.mask_feature_extractors = nn.ModuleList([
            ConvolutionalBackbone(c_in=self.mask_channels, group_norm=cnn_group_norm)
            for _ in range(self.inputs_num)
        ])

        # Positional embeddings.
        if self.pos_embeddings_type == PositionalEmbeddingsType.SINUSOIDAL:
            self.pos_embed = nn.Parameter(positionalencoding2d(384, 14, 14))
            self.pos_embed.requires_grad = False
        elif self.pos_embeddings_type == PositionalEmbeddingsType.CAPE:
            self.pos_embed: CAPE2d = CAPE2d(d_model=384,
                                            max_global_shift=0.5,
                                            max_local_shift=0.5,
                                            max_global_scaling=1.4)

        # First stage masked transformers.
        assert (384 % self.first_stage_heads) == 0
        self.masked_transformers = nn.ModuleList([
            MaskedTransformer(dim=384,
                              depth=self.first_stage_depth,
                              heads=self.first_stage_heads,
                              dim_head=384//self.first_stage_heads,
                              mlp_dim=384*4,
                              dropout=self.dropout)
            for _ in range(inputs_num)
        ])

        # Second stage fusion.
        assert (384 % self.second_stage_heads) == 0
        if second_stage_fusion_type == FusionType.BOTTLENECK_FUSION:
            self.overall_fusion = BottleneckFusionManyToOne(
                signals=self.inputs_num,
                dim=384,
                depth=self.second_stage_depth,
                # Equal to the size of an image after feature extraction.
                bottleneck_units=14*14,
                heads=self.second_stage_heads,
                dim_head=384//self.second_stage_heads,
                mlp_dim=384*4,
                dropout=self.dropout
            )
        elif second_stage_fusion_type == FusionType.POSITIONAL_FUSION:
            self.overall_fusion = PositionalWiseFusion(
                positional_fusion_layers=self.second_stage_depth,
                long_rage_layers=self.second_stage_depth,
                dim=384,
                heads=self.second_stage_heads,
                dim_head=384//self.second_stage_heads,
                mlp_dim=384*4,
                dropout=self.dropout
            )
        else:
            raise RuntimeError(f"Non-supported fusion type: {second_stage_fusion_type.value}")

        # Localization head.
        self.localization_head: nn.Module = _create_localization_head(
            upscaler_type, add_final_logit_layers, cnn_group_norm
        )

        # Detection head.
        self.detection_head: nn.Module = _create_detection_head(
            detection_head_type, add_logits_layer=add_final_logit_layers, dropout=dropout
        )

        # Initialize the weights of the model.
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> dict[str: torch.Tensor]:
        """Fuses multiple forensics signals with respect to an attention mask.

        :param x: A tensor of size: (N, C, H, W)
            Where:
                - N: Batch size.
                - C: masks_num * mask_channels
                - H: Height of the inputs.
                - W: Width of the inputs.
        :param attention_mask: A bool tensor of size (N, H*W/16, H*W/16)

        :return: A dict containing the following keys:
            'localization': A tensor of size (N, 1, H, W) containing the localization map.
            'detection': A tensor of size (N, 1) or (N, 2) containing the output of
                binary detection. The size of the output depends on whether single-value
                or one-hot detection head has been selected.
        """
        # Extract forensics masks features.
        features: list[torch.Tensor] = []
        for i, mask_fe in enumerate(self.mask_feature_extractors):
            # TODO: Currently, it does not work for self.mask_channels != 1. Fix it if needed.
            if self.mask_channels > 1:
                raise RuntimeError("Model does not support mask channels > 1")
            features.append(mask_fe(x[:, i, :, :].unsqueeze(1)))

        # Perform first stage masked attention.
        first_stage_outs: list[torch.Tensor] = []
        for signal_features, masked_transformer in zip(features, self.masked_transformers):

            if self.pos_embeddings_type == PositionalEmbeddingsType.SINUSOIDAL:
                # Add sinusoidal position embeddings.
                pos_embed: torch.Tensor = self.pos_embed.unsqueeze(
                    dim=0).repeat(signal_features.size(dim=0), 1, 1, 1)
                signal_features = signal_features + pos_embed
            elif self.pos_embeddings_type == PositionalEmbeddingsType.CAPE:
                signal_features = einops.rearrange(signal_features, "b c h w -> h w b c")
                signal_features = self.pos_embed(signal_features)
                signal_features = einops.rearrange(signal_features, "h w b c -> b c h w")
            else:
                raise RuntimeError(f"{self.pos_embeddings_type.value} is not a "
                                   "supported type of positional embeddings.")

            # Patchify.
            signal_features = einops.rearrange(
                signal_features, "b c h w -> b (h w) c")

            signal_features = masked_transformer(signal_features, attention_mask)
            first_stage_outs.append(signal_features)

        # Perform second stage fusion.
        second_stage_input: torch.Tensor = torch.stack(first_stage_outs, dim=1)
        final_features: torch.Tensor = self.overall_fusion(second_stage_input)

        # Output localization map and binary detection.
        localization_features = einops.rearrange(final_features,
                                                 "b (h w) c -> b c h w",
                                                 h=int(math.sqrt(final_features.size(dim=1))))
        localization_map: torch.Tensor = self.localization_head(localization_features)
        binary_detection: torch.Tensor = self.detection_head(localization_features)
        out: dict[str, torch.Tensor] = {
            "localization": localization_map,
            "detection": binary_detection
        }

        return out


class MaskedAttentionFusionModelWithImage(nn.Module):

    def __init__(self,
                 inputs_num: int,
                 first_stage_depth: int,
                 first_stage_heads: int,
                 second_stage_depth: int,
                 second_stage_heads: int,
                 height: int,
                 width: int,
                 mask_channels: Union[int, list[int]] = 1,
                 image_channels: int = 3,
                 dropout: float = 0.,
                 drop_stream_probability: float = 0.,
                 drop_path_probability: float = 0.,
                 upscaler_type: UpscalerType = UpscalerType.CONV_UPSCALER,
                 detection_head_type: DetectionHeadType = DetectionHeadType.ONE_HOT_OUT,
                 add_final_logit_layers: bool = True,
                 cnn_group_norm: bool = False,
                 second_stage_fusion_type: FusionType = FusionType.BOTTLENECK_FUSION,
                 image_backbone_type: BackboneType = BackboneType.CONVOLUTIONAL,
                 pass_dino_features_through_first_stage: bool = False):
        """Creates a model for fusing forensics signals with segmentation masks.

        Inputs to the model should be two tensor of size:
            (N, C, H, W) and (N, H*W, H*W)
        Where:
            - N: Batch size.
            - C: image_channels + map_channels + (masks_num * mask_channels)

                For example, when fusing 5 single channel masks, using an RGB image
                and an RGB segmentation map: C = 3 + 3 + (5 * 1) = 11

                The order of inputs should be (image, segmentation_map, ...masks...)!
                Otherwise, expect weird results!
            - H: Height of the inputs.
            - W: Width of the inputs.

        :param inputs_num: Number of forensics inputs to fuse.
        :param first_stage_depth: Number of fusion layers in 1st stage.
        :param first_stage_heads:
        :param second_stage_depth: Number of fusion layers in 2nd stage.
        :param second_stage_heads:
        :param height: Height of the input masks.
        :param width: Width of the input masks.
        :param mask_channels: Number of channels in the forensics masks.
        :param image_channels: Number of channels in the image.
        :param dropout:
        :param drop_stream_probability: Probability of dropping a stream during training.
        :param drop_path_probability: Probability of dropping transformer residual paths
            on training.
        :param upscaler_type: Type of the upscaler to use for the localization head.
        :param detection_head_type: Type of the detection head.
        :param add_final_logit_layers: When set to False, the detection and localization
            outputs of the model will not pass through a final sigmoid layer, so they will not
            be bounded to [0, 1]. Useful for utilizing BCEWithLogitsLoss, since BCELoss cannot
            be applied with mixed precision training.
        :param cnn_group_norm: When set to True, GroupNorm is used in convolutional layers
            instead of BatchNorm.
        :param second_stage_fusion_type: Defines the type of fusion for the second stage.
            Defaults to bottleneck fusion.
        :param image_backbone_type: Defines the type of backbone utilized for extracting
            features from the image.
        :param pass_dino_features_through_first_stage: When set to True, image features
            extracted through the DINO backbone are passed through a first-stage
            branch, for extracting instance-level features. Defaults to False.
        """
        super().__init__()

        self.inputs_num: int = inputs_num
        self.first_stage_depth: int = first_stage_depth
        self.first_stage_heads: int = first_stage_heads
        self.second_stage_depth: int = second_stage_depth
        self.second_stage_heads: int = second_stage_heads
        self.height: int = height
        self.width: int = width
        self.mask_channels: Union[int, list[int]] = mask_channels
        self.image_channels: int = image_channels
        self.dropout: float = dropout

        # Feature extractors.
        if isinstance(self.mask_channels, int):
            self.mask_feature_extractors = nn.ModuleList([
                ConvolutionalBackbone(c_in=self.mask_channels, group_norm=cnn_group_norm)
                for _ in range(self.inputs_num)
            ])
        elif isinstance(self.mask_channels, list):
            feature_extractors = []
            for ch in self.mask_channels:
                if ch > 3:
                    feature_extractors.append(
                        backbones.ConvolutionalBackboneHighDim(
                            c_in=ch, group_norm=cnn_group_norm
                        )
                    )
                else:
                    feature_extractors.append(
                        backbones.ConvolutionalBackbone(
                            c_in=ch, group_norm=cnn_group_norm
                        )
                    )
            self.mask_feature_extractors = nn.ModuleList(feature_extractors)
        else:
            raise RuntimeError(f"Unsupported type for mask_channels: {type(self.mask_channels)}")

        if image_backbone_type == BackboneType.CONVOLUTIONAL:
            self.image_feature_extractor = ConvolutionalBackbone(c_in=self.image_channels)
            self.pass_image_features_through_first_stage: bool = True
            self.add_pos_embeddings_to_image: bool = True
        elif image_backbone_type in [
            BackboneType.DINO,
            BackboneType.DINOv2,
            BackboneType.DINOv2_FEATURE_INTERPOLATION,
            BackboneType.DINOv2_FROZEN,
            BackboneType.DINOv2_FROZEN_FEATURE_INTERPOLATION,
            BackboneType.DINOv2_FROZEN_BILINEAR_FEAT_INT,
            BackboneType.DINOv2_FROZEN_PENULTIMATE_LAYER,
            BackboneType.DINOv2_FROZEN_PENULTIMATE_LAYER_FEATURE_INTERPOLATION,
            BackboneType.DINOv2_FROZEN_MULTISCALE,
            BackboneType.DINOv2_FROZEN_MULTISCALE_FEATURE_INTERPOLATION,
            BackboneType.DINOv2_FROZEN_MULTISCALE_BILINEAR_FEAT_INT,
            BackboneType.DINOv2_PATCH_EMBED_FROZEN_FEATURE_INTERPOLATION
        ]:
            # Define only the misc attributes here and initialize the pretrained DINO
            # models only after the weights initialization has been completed.
            self.pass_image_features_through_first_stage: bool = \
                pass_dino_features_through_first_stage
            self.add_pos_embeddings_to_image: bool = False
        elif image_backbone_type == BackboneType.NONE:
            self.image_feature_extractor = None
            self.pass_image_features_through_first_stage: bool = False
            self.add_pos_embeddings_to_image: bool = False
        else:
            raise RuntimeError(f"Non-supported backbone: {image_backbone_type.name}")

        self.image_features_scales: int = 1
        if image_backbone_type in [
            BackboneType.DINOv2_FROZEN_MULTISCALE,
            BackboneType.DINOv2_FROZEN_MULTISCALE_FEATURE_INTERPOLATION,
            BackboneType.DINOv2_FROZEN_MULTISCALE_BILINEAR_FEAT_INT
        ]:
            self.image_features_scales = 4

        vertical_tokens: int = self.height // 16
        horizontal_tokens: int = self.width // 16

        # Positional embeddings.
        self.pos_embed = nn.Parameter(positionalencoding2d(384, vertical_tokens, horizontal_tokens))
        self.pos_embed.requires_grad = False

        # First stage masked transformers.
        if self.pass_image_features_through_first_stage:
            # Include an additional transformer for the image features at each scale.
            first_stage_streams: int = inputs_num + self.image_features_scales
        else:
            first_stage_streams: int = inputs_num
        assert (384 % self.first_stage_heads) == 0
        self.masked_transformers = nn.ModuleList([
            MaskedTransformer(dim=384,
                              depth=self.first_stage_depth,
                              heads=self.first_stage_heads,
                              dim_head=384//self.first_stage_heads,
                              mlp_dim=384*4,
                              dropout=self.dropout,
                              drop_path_prob=drop_path_probability)
            for _ in range(first_stage_streams)
        ])

        # First stage drop path layer.
        self.drop_path = timm.layers.DropPath(
            drop_prob=drop_stream_probability) if drop_stream_probability > .0 else nn.Identity()

        # Second stage fusion.
        assert (384 % self.second_stage_heads) == 0
        if second_stage_fusion_type == FusionType.BOTTLENECK_FUSION:
            self.overall_fusion = BottleneckFusionManyToOne(
                signals=(self.inputs_num if image_backbone_type == BackboneType.NONE
                         else self.inputs_num+1),  # Include the image too.
                dim=384,
                depth=self.second_stage_depth,
                # Equal to the size of an image after feature extraction.
                bottleneck_units=vertical_tokens * horizontal_tokens,
                heads=self.second_stage_heads,
                dim_head=384 // self.second_stage_heads,
                mlp_dim=384 * 4,
                dropout=self.dropout
            )
        elif second_stage_fusion_type == FusionType.POSITIONAL_FUSION:
            self.overall_fusion = PositionalWiseFusion(
                positional_fusion_layers=self.second_stage_depth,
                long_rage_layers=self.second_stage_depth,
                dim=384,
                heads=self.second_stage_heads,
                dim_head=384 // self.second_stage_heads,
                mlp_dim=384 * 4,
                dropout=self.dropout,
                drop_path_prob=drop_path_probability
            )
        elif second_stage_fusion_type == FusionType.POSITIONAL_FUSION_NO_TFT:
            self.overall_fusion = PositionalWiseFusion(
                positional_fusion_layers=0,  # Disable TFT
                long_rage_layers=self.second_stage_depth,
                dim=384,
                heads=self.second_stage_heads,
                dim_head=384 // self.second_stage_heads,
                mlp_dim=384 * 4,
                dropout=self.dropout,
                drop_path_prob=drop_path_probability
            )
        elif second_stage_fusion_type == FusionType.POSITIONAL_FUSION_NO_LDT:
            self.overall_fusion = PositionalWiseFusion(
                positional_fusion_layers=self.second_stage_depth,
                long_rage_layers=0,  # Disable LDT
                dim=384,
                heads=self.second_stage_heads,
                dim_head=384 // self.second_stage_heads,
                mlp_dim=384 * 4,
                dropout=self.dropout,
                drop_path_prob=drop_path_probability
            )
        else:
            raise RuntimeError(f"Non-supported fusion type: {second_stage_fusion_type.value}")

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

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        return_similarities: bool = False,
        similarities_type: str = "cos",
    ) -> dict[str: Union[torch.Tensor, list[torch.Tensor]]]:
        """Fuses an image and multiple forensics signals with respect to an attention mask.

        :param x: A tensor of size: (N, C, H, W)
            Where:
                - N: Batch size.
                - C: image_channels + (masks_num * mask_channels)

                    For example, when fusing 5 single channel masks, using an RGB image:
                        C = 3 + (5 * 1) = 8

                    The order of inputs should be (image, ...masks...)!
                    Otherwise, expect weird results!
                - H: Height of the inputs.
                - W: Width of the inputs.
        :param attention_mask: A bool tensor of size (N, H*W/16, H*W/16)

        :return: A dict containing the following keys:
            'localization': A tensor of size (N, 1, H, W) containing the localization map.
            'detection': A tensor of size (N, 1) or (N, 2) containing the output of
                binary detection. The size of the output depends on whether single-value
                or one-hot detection head has been selected.
            'similarities': (Optional) A list of (N, H, W) maps denoting the similarity
                between the fused tokens and the tokens of each signal.
        """
        # Unpack the inputs (image, masks).
        image_index: int = 0
        masks_index: int = self.image_channels
        image: torch.Tensor = x[:, image_index:masks_index, :, :]
        masks: torch.Tensor = x[:, masks_index:, :, :]

        features: list[torch.Tensor] = []
        # Extract image features.
        image_features: Union[torch.Tensor, list[torch.Tensor], None] = None
        if self.image_feature_extractor is not None:
            image_features = self.image_feature_extractor(image)
            if self.pass_image_features_through_first_stage:
                if self.image_features_scales > 1:
                    features.extend(image_features)
                else:
                    features.append(image_features)
        # Extract forensics masks features.
        for i, mask_fe in enumerate(self.mask_feature_extractors):
            # TODO: Currently, it does not work for self.mask_channels != 1. Fix it if needed.
            if self.mask_channels > 1:
                raise RuntimeError("Model does not support mask channels > 1")
            features.append(mask_fe(masks[:, i, :, :].unsqueeze(1)))

        # Perform first stage masked attention.
        first_stage_outs: list[torch.Tensor] = []
        for i, (signal_features, masked_transformer) in enumerate(zip(features,
                                                                      self.masked_transformers)):
            if (
                (i < self.image_features_scales
                    and self.pass_image_features_through_first_stage  # Processing image
                    and self.add_pos_embeddings_to_image)  # Image requires pos. embeddings.
                or i >= self.image_features_scales   # Processing a signal, not an image.
                or not self.pass_image_features_through_first_stage  # Only signals exists.
            ):
                # Add sinusoidal position embeddings.
                pos_embed: torch.Tensor = self.pos_embed.unsqueeze(
                    dim=0).repeat(signal_features.size(dim=0), 1, 1, 1)
                signal_features = signal_features + pos_embed

                # Patchify.
                signal_features = einops.rearrange(
                    signal_features, "b c h w -> b (h w) c")

            signal_features = masked_transformer(signal_features, attention_mask)
            first_stage_outs.append(signal_features)
        if not self.pass_image_features_through_first_stage and image_features is not None:
            if self.image_features_scales > 1:
                first_stage_outs.extend(image_features)
            else:
                first_stage_outs.append(image_features)

        # Apply drop path on the features of the first stage.
        first_stage_outs = [self.drop_path(f) for f in first_stage_outs]

        # Perform second stage fusion.
        second_stage_input: torch.Tensor = torch.stack(first_stage_outs, dim=1)
        if return_similarities:
            final_features: torch.Tensor
            similarities: list[torch.Tensor]
            final_features, similarities = self.overall_fusion(
                second_stage_input,
                return_similarities=return_similarities,
                similarities_type=similarities_type
            )
        else:
            final_features = self.overall_fusion(second_stage_input)

        # Output localization map and binary detection.
        localization_features = einops.rearrange(final_features,
                                                 "b (h w) c -> b c h w",
                                                 h=int(math.sqrt(final_features.size(dim=1))))
        localization_map: torch.Tensor = self.localization_head(localization_features)
        binary_detection: torch.Tensor = self.detection_head(localization_features)
        out: dict[str, Union[torch.Tensor, list[torch.Tensor]]] = {
            "localization": localization_map,
            "detection": binary_detection
        }

        if return_similarities:
            if similarities_type == "cos":
                out["similarities"] = [
                    einops.rearrange(
                        s, "b (h w) -> b h w", h=int(math.sqrt(final_features.size(dim=1))))
                    for s in similarities
                ]
            elif similarities_type == "attn":
                similarities = [
                    einops.rearrange(
                        s, "b n (h w) s -> b s n h w", h=int(math.sqrt(final_features.size(dim=1))))
                    for s in similarities
                ]
                stacked_sim = torch.stack(similarities, dim=1)
                out["similarities"] = [stacked_sim[:, :, i, :, :, :]
                                       for i in range(stacked_sim.size(dim=2))]  # (batch, tr_layers, heads, height, width)

        return out

    def extend_pretrained_model(
        self,
        state_dict: dict[str, torch.Tensor],
        whole_network_finetune: bool = False
    ) -> None:
        model_parameters: set[str] = set(self.state_dict())
        previous_model_parameters: set[str] = set(state_dict.keys())
        for k in previous_model_parameters:
            assert k in model_parameters, f"{k} not in model parameters"

        self.load_state_dict(state_dict, strict=False)

        if not whole_network_finetune:
            for n, p in self.named_parameters():
                if n in previous_model_parameters:
                    p.requires_grad = False


class BottleneckFusionTwoToOne(nn.Module):
    """Implementation of Bottleneck Fusion for two signals."""

    def __init__(self,
                 dim: int,
                 depth: int,
                 bottleneck_units: int,
                 heads,
                 dim_head,
                 mlp_dim,
                 dropout=0.):
        """
        :param depth: Number of fusion layers.
        :param bottleneck_units: Number of bottleneck units.
        """
        super().__init__()

        self.depth: int = depth
        self.bottleneck_units: int = bottleneck_units

        self.layers = nn.ModuleList([
            nn.ModuleList([
                Transformer(dim, heads, dim_head, mlp_dim, dropout),
                Transformer(dim, heads, dim_head, mlp_dim, dropout)
            ]) for _ in range(self.depth)
        ])

        self.bottleneck_token = nn.Parameter(torch.randn((bottleneck_units, dim)))
        nn.init.trunc_normal_(self.bottleneck_token, std=.02)

    def forward(self, x):
        """
        :param x: A tensor of shape (B, 2, H*W, C)
            where:
            - B: Batch size.
            - 2: Number of 2D signals to fuse. Should always be 2.
            - H: Height of each 2D signal.
            - W: Width of each 2D signal.
            - C: Dimensionality of each patch.

        :returns: A tensor of shape (B, bottleneck_units, C)
        """
        # Split the two signals into two (B, H*W, C) tensors.
        signal_1 = torch.squeeze(x[:, 0, :, :], dim=1)
        signal_2 = torch.squeeze(x[:, 1, :, :], dim=1)

        bottleneck = torch.unsqueeze(self.bottleneck_token, 0).repeat((x.size(dim=0), 1, 1))

        for trans1, trans2 in self.layers:
            # Append the bottleneck units to each signal.
            signal_1 = torch.cat([bottleneck, signal_1], dim=1)
            signal_2 = torch.cat([bottleneck, signal_2], dim=1)

            # Pass each signal through a separate transformer.
            signal_1 = trans1(signal_1)
            signal_2 = trans2(signal_2)

            # Extract output bottleneck tokens.
            signal_1 = signal_1.split([self.bottleneck_units, x.size(dim=2)], dim=1)
            bottleneck_1 = signal_1[0]
            signal_2 = signal_2.split([self.bottleneck_units, x.size(dim=2)], dim=1)
            bottleneck_2 = signal_2[0]

            bottleneck = torch.mean(torch.stack([bottleneck_1, bottleneck_2]), dim=0)
            signal_1 = signal_1[1]
            signal_2 = signal_2[1]

        return bottleneck


class BottleneckFusionManyToOne(nn.Module):
    """Implementation of Bottleneck Fusion for two signals."""

    def __init__(self,
                 signals: int,
                 dim: int,
                 depth: int,
                 bottleneck_units: int,
                 heads: int,
                 dim_head: int,
                 mlp_dim,
                 dropout=0.):
        """
        :param signals: Number of signals to fuse.
        :param dim: Dimensionality of the input patches.
        :param depth: Number of fusion layers.
        :param bottleneck_units: Number of bottleneck units.
        :param heads: Number of attention heads in transformer layers.
        :param dim_head: Dimensionality of the input patch processed in each SA head.
        :param mlp_dim: Dimensionality of the MLP layer of the transformer.
        :param dropout: Dropout rate.
        """
        super().__init__()

        self.signals: int = signals
        self.depth: int = depth
        self.bottleneck_units: int = bottleneck_units

        self.layers: nn.ModuleList = nn.ModuleList([
            nn.ModuleList([
                Transformer(dim, heads, dim_head, mlp_dim, dropout) for _ in range(self.signals)
            ]) for _ in range(self.depth)
        ])

        self.bottleneck_token = nn.Parameter(torch.randn((bottleneck_units, dim)))
        nn.init.trunc_normal_(self.bottleneck_token, std=.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: A tensor of shape (B, S, H*W, C)
            where:
            - B: Batch size.
            - S: Number of 2D signals to fuse.
            - H: Height of each 2D signal.
            - W: Width of each 2D signal.
            - C: Dimensionality of each patch.

        :returns: A tensor of shape (B, bottleneck_units, C)
        """
        # Split input signals into (B, H*W, C) tensors.
        signals: list[torch.Tensor] = [torch.squeeze(x[:, i, :, :], dim=1)
                                       for i in range(self.signals)]

        # Repeat the bottleneck units for each sample in batch.
        bottleneck = torch.unsqueeze(self.bottleneck_token, 0).repeat((x.size(dim=0), 1, 1))

        for transformers in self.layers:
            # Append the bottleneck units to each signal.
            signals = [torch.cat([bottleneck, s], dim=1) for s in signals]

            # Pass each signal through a separate transformer.
            signals = [t(s) for t, s in zip(transformers, signals)]

            # Extract output bottleneck tokens.
            signals = [s.split([self.bottleneck_units, x.size(dim=2)], dim=1) for s in signals]
            partial_bottleneck_tokens: list[torch.Tensor] = [s[0] for s in signals]

            bottleneck = torch.mean(torch.stack(partial_bottleneck_tokens), dim=0)
            signals = [s[1] for s in signals]

        return bottleneck


class PositionalWiseFusion(nn.Module):

    def __init__(
        self,
        positional_fusion_layers: int,
        long_rage_layers: int,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.,
        drop_path_prob: float = 0.,
    ):
        super().__init__()

        self.positional_fusion_transformers: nn.ModuleList = nn.ModuleList([
            Transformer(dim, heads, dim_head, mlp_dim, dropout,
                        drop_path_prob=drop_path_prob)
            for _ in range(positional_fusion_layers)
        ])
        self.long_range_transformers: nn.ModuleList = nn.ModuleList([
            Transformer(dim, heads, dim_head, mlp_dim, dropout,
                        drop_path_prob=drop_path_prob)
            for _ in range(long_rage_layers)
        ])

        # Forensics token is the token that after fusion will hold the information
        # of each signal. Its size matches the size of a single token in the input
        # sequence, so, for a 2D input, it matches the size of a patch token.
        self.forensics_token = nn.Parameter(torch.zeros((1, 1, 1, dim)))
        nn.init.trunc_normal_(self.forensics_token, std=.02)

        self.cossim = torch.nn.CosineSimilarity(dim=2)

    def forward(
        self,
        x: torch.Tensor,
        return_similarities: bool = False,
        similarities_type: str = "cos"
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
        """Fuses multiple signals into a single one.

        :param x: A tensor of shape (B, S, H*W, C)
            where:
            - B: Batch size.
            - S: Number of 2D signals to fuse.
            - H: Height of each 2D signal.
            - W: Width of each 2D signal.
            - C: Dimensionality of each patch.
        :param return_similarities: A flag that when set to True returns the cosine similarities
            between the input tokens and the fused token.
        :param similarities_type: "cos" or "attn"

        :returns: A tensor of shape (B, sequence_length, C). When the `return_similarities` flag is
            set to True, a tuple is returned, containing as a second item `S` (B, H*W) tensors,
            with the cosine similarities between the tokens of the `S` input signals and the
            output signal of the Token Fusion Transformer.
        """
        if return_similarities:
            if similarities_type == "cos":
                inputs: torch.Tensor = x
            elif similarities_type == "attn":
                # Initialize a list where the attention maps of TFTs will be stored.
                attn_maps: list[torch.Tensor] = []

        if len(self.positional_fusion_transformers) > 0:
            batch_size: int = x.size(dim=0)

            # Repeat the forensics token for each sample in the mini-batch and for each input token.
            forensics_token: torch.Tensor = self.forensics_token.repeat(
                (batch_size, 1, x.size(dim=2), 1)
            )  # (B, 1, H*W, C)

            # Concatenate the forensics token with the input signals.
            x = torch.cat([forensics_token, x], dim=1)  # (B, S+1, H*W, C)

            # Transform the input to attend between the different tokens for a specific
            # input token.
            x = einops.rearrange(x, "b s l c -> (b l) s c")  # (B*H*W, S+1, C)

            # Pass the sequence through positional fusion layers.
            for positional_fusion_transformer in self.positional_fusion_transformers:
                if return_similarities and similarities_type == "attn":
                    x, attn = positional_fusion_transformer(
                        x, return_attention=True
                    )  # (B*H*W, S+1, C)
                    attn = attn[:, :, 0, 1:]  # Attention only for fused token.
                    attn = einops.rearrange(attn, "(b l) h s -> b h l s", b=batch_size)
                    attn_maps.append(attn)
                else:
                    x = positional_fusion_transformer(x)

            # Retain only the fused forensics token.
            x = x[:, 0, :]  # (B*H*W,  384)

            # Convert back to a normal transformer sequence.
            x = einops.rearrange(x, "(b l) c -> b l c", b=batch_size)   # (B, H*W, C)
        else:
            # Fuse the input features using their average.
            x = torch.mean(x, dim=1)

        if return_similarities:
            if similarities_type == "cos":
                similarities: list[torch.Tensor] = [self.cossim(inputs[:, i, :, :], x)
                                                    for i in range(inputs.size(dim=1))]
            elif similarities_type == "attn":
                similarities = attn_maps
            else:
                raise RuntimeError(f"Unsupported similarities type: {similarities_type}")

        # Pass the fused token through transformer layers to take into account long range
        # dependencies.
        for long_range_transformer in self.long_range_transformers:
            x = long_range_transformer(x)  # (B, H*W, C)

        if return_similarities:
            return x, similarities
        return x


class MaskedTransformer(nn.Module):
    """Multilayer transformer with support for attention mask."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.,
        drop_path_prob: float = 0.,
    ):
        super().__init__()

        self.depth: int = depth
        self.layers: nn.ModuleList = nn.ModuleList([
            Transformer(dim, heads, dim_head, mlp_dim, dropout,
                        drop_path_prob=drop_path_prob)
            for _ in range(self.depth)
        ])

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        for transformer_layer in self.layers:
            x = transformer_layer(x, attention_mask)
        return x


class Reshape(nn.Module):
    def __init__(self, pattern: str, **axes_lengths):
        super().__init__()
        self.pattern: str = pattern
        self.axes_lengths = axes_lengths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(x, self.pattern, **self.axes_lengths)


def positionalencoding2d(d_model, height, width):
    """Generates sinusoidal position embedding matrix.

    Code from:
      https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py

    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(
        pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(
        pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.trunc_normal_(m.weight, std=.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def _create_detection_head(
    detection_head_type: DetectionHeadType,
    add_logits_layer: bool = True,
    dropout: float = .0
) -> nn.Module:
    if detection_head_type == DetectionHeadType.ONE_HOT_OUT:
        detection_head_layers: list[nn.Module] = [
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=384, out_features=2),
        ]
        if add_logits_layer:
            detection_head_layers.append(nn.Softmax(dim=1))
        detection_head = nn.Sequential(*detection_head_layers)
    elif detection_head_type == DetectionHeadType.SINGLE_OUT:
        detection_head_layers: list[nn.Module] = [
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=384, out_features=1),
        ]
        if add_logits_layer:
            detection_head_layers.append(nn.Sigmoid())
        detection_head = nn.Sequential(*detection_head_layers)
    elif detection_head_type == DetectionHeadType.TRANSFORMER_TWO_OUT_SOFTMAX:
        detection_head = nn.Sequential(
            Reshape("b c h w -> b (h w) c"),
            detectors.TransformerDetector(
                layers=6,
                dim=384,
                out_dim=2,
                heads=12,
                dim_head=384 // 12,
                mlp_dim=384 * 4,
                classification_hidden_dim=384 * 4,
                last_layer_activation_type=(detectors.ActivationType.SOFTMAX
                                            if add_logits_layer else None),
                dropout=dropout
            )
        )
    elif detection_head_type == DetectionHeadType.TRANSFORMER_TWO_OUT_SIGMOID:
        detection_head = nn.Sequential(
            Reshape("b c h w -> b (h w) c"),
            detectors.TransformerDetector(
                layers=6,
                dim=384,
                out_dim=2,
                heads=12,
                dim_head=384 // 12,
                mlp_dim=384 * 4,
                classification_hidden_dim=384 * 4,
                last_layer_activation_type=(detectors.ActivationType.SIGMOID
                                            if add_logits_layer else None),
                dropout=dropout
            )
        )
    elif detection_head_type == DetectionHeadType.TRANSFORMER_SINGLE_OUT_SIGMOID:
        detection_head = nn.Sequential(
            Reshape("b c h w -> b (h w) c"),
            detectors.TransformerDetector(
                layers=6,
                dim=384,
                out_dim=1,
                heads=12,
                dim_head=384 // 12,
                mlp_dim=384 * 4,
                classification_hidden_dim=384 * 4,
                last_layer_activation_type=(detectors.ActivationType.SIGMOID
                                            if add_logits_layer else None),
                dropout=dropout
            )
        )
    elif detection_head_type == DetectionHeadType.TRANSFORMER_ONE_LAYER_MLP_SINGLE_OUT_SIGMOID:
        detection_head = nn.Sequential(
            Reshape("b c h w -> b (h w) c"),
            detectors.TransformerDetector(
                layers=4,
                dim=384,
                out_dim=1,
                heads=12,
                dim_head=384 // 12,
                mlp_dim=384 * 4,
                classification_hidden_dim=0,
                last_layer_activation_type=(detectors.ActivationType.SIGMOID
                                            if add_logits_layer else None),
                dropout=dropout
            )
        )
    elif detection_head_type == DetectionHeadType.TRANSFORMER_LITE_ONE_LAYER_MLP_SINGLE_OUT_SIGMOID:
        detection_head = nn.Sequential(
            Reshape("b c h w -> b (h w) c"),
            detectors.TransformerDetector(
                layers=2,
                dim=384,
                out_dim=1,
                heads=12,
                dim_head=384 // 12,
                mlp_dim=384 * 4,
                classification_hidden_dim=0,
                last_layer_activation_type=(detectors.ActivationType.SIGMOID
                                            if add_logits_layer else None),
                dropout=dropout
            )
        )
    else:
        raise RuntimeError(f"{detection_head_type.value} is not a valid detection head type.")

    return detection_head


def _create_localization_head(
    upscaler_type: UpscalerType,
    add_final_logit_layers: bool,
    cnn_group_norm: bool
) -> nn.Module:
    localization_head: nn.Module

    if upscaler_type == UpscalerType.CONV_UPSCALER:
        localization_head = ConvolutionalUpscaler(
            c_in=384, c_out=1,
            add_final_sigmoid_layer=add_final_logit_layers,
            group_norm=cnn_group_norm
        )
    elif upscaler_type == UpscalerType.DOUBLE_CONV_UPSCALER:
        localization_head = DoubleConvolutionalUpscaler(
            c_in=384, c_out=1,
            add_final_sigmoid_layer=add_final_logit_layers,
            group_norm=cnn_group_norm
        )
    elif upscaler_type == UpscalerType.DOUBLE_CONV_UPSCALER_2:
        localization_head = DoubleConvolutionalUpscaler2(
            c_in=384, c_out=1,
            add_final_sigmoid_layer=add_final_logit_layers,
            group_norm=cnn_group_norm
        )
    elif upscaler_type == UpscalerType.DOUBLE_CONV_UPSCALER_ONE_HOT:
        localization_head = DoubleConvolutionalUpscaler(
            c_in=384, c_out=2,
            use_softmax_activation=True,
            add_final_sigmoid_layer=False,
            group_norm=cnn_group_norm
        )
    elif upscaler_type == UpscalerType.RESIDUAL_UPSCALER:
        localization_head = ResidualUpscaler(
            c_in=384, c_out=1, add_final_sigmoid_layer=add_final_logit_layers
        )
    else:
        raise RuntimeError(f"Non-implemented upscaler type: {upscaler_type.name}")

    return localization_head
