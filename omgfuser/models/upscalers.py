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

import torch
from torch import nn

from .resnet_blocks import ResidualBlock


class DoubleConvolutionalUpscaler3(nn.Module):

    def __init__(
        self,
        c_in: int,
        c_out: int,
        add_final_sigmoid_layer: bool = True,
        group_norm: bool = False,
        use_softmax_activation: bool = False
    ):
        super().__init__()

        assert not use_softmax_activation or not add_final_sigmoid_layer, \
            "Sigmoid and softmax activations cannot be utilized together."
        assert not use_softmax_activation or c_out == 2, \
            "c_out is required to be 2 when softmax activation is used."

        upscaler_layers: list[nn.Module] = [
            # Transpose Conv layer 1.
            nn.ConvTranspose2d(in_channels=c_in, out_channels=c_in,
                               kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(16, c_in) if group_norm else nn.BatchNorm2d(c_in),
            nn.ReLU(),
            # Transpose Conv layer 2.
            nn.ConvTranspose2d(in_channels=c_in, out_channels=c_in//2, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.Conv2d(in_channels=c_in//2, out_channels=c_in//2,
                      kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, c_in//2) if group_norm else nn.BatchNorm2d(c_in//2),
            nn.ReLU(),
            # Transpose Conv layer 3.
            nn.ConvTranspose2d(in_channels=c_in//2, out_channels=c_in//4, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.Conv2d(in_channels=c_in//4, out_channels=c_in//4,
                      kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, c_in//4) if group_norm else nn.BatchNorm2d(c_in//4),
            nn.ReLU(),
            # Transpose Conv layer 4.
            nn.ConvTranspose2d(in_channels=c_in//4, out_channels=c_in//8, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.Conv2d(in_channels=c_in//8, out_channels=c_in//8,
                      kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, c_in//8) if group_norm else nn.BatchNorm2d(c_in//8),
            nn.ReLU(),
            # Transpose Conv layer 5.
            nn.ConvTranspose2d(in_channels=c_in//8, out_channels=c_out, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.GroupNorm(c_out, c_out) if group_norm else nn.BatchNorm2d(c_out),
            nn.Conv2d(in_channels=c_out, out_channels=c_out,
                      kernel_size=3, stride=1, padding=1),
        ]
        if add_final_sigmoid_layer:
            upscaler_layers.append(nn.Sigmoid())
        if use_softmax_activation:
            upscaler_layers.append(nn.Softmax2d())

        self.upscaler: nn.Module = nn.Sequential(*upscaler_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upscaler(x)


class DoubleConvolutionalUpscaler2(nn.Module):

    def __init__(
        self,
        c_in: int,
        c_out: int,
        add_final_sigmoid_layer: bool = True,
        group_norm: bool = False,
        use_softmax_activation: bool = False
    ):
        super().__init__()

        assert not use_softmax_activation or not add_final_sigmoid_layer, \
            "Sigmoid and softmax activations cannot be utilized together."
        assert not use_softmax_activation or c_out == 2, \
            "c_out is required to be 2 when softmax activation is used."

        upscaler_layers: list[nn.Module] = [
            # Transpose Conv layer 1.
            nn.ConvTranspose2d(in_channels=c_in, out_channels=c_in,
                               kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(16, c_in) if group_norm else nn.BatchNorm2d(c_in),
            nn.ReLU(),
            # Transpose Conv layer 2.
            nn.ConvTranspose2d(in_channels=c_in, out_channels=c_in//2, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, c_in//2) if group_norm else nn.BatchNorm2d(c_in//2),
            nn.Conv2d(in_channels=c_in//2, out_channels=c_in//2,
                      kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, c_in//2) if group_norm else nn.BatchNorm2d(c_in//2),
            nn.ReLU(),
            # Transpose Conv layer 3.
            nn.ConvTranspose2d(in_channels=c_in//2, out_channels=c_in//4, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, c_in//4) if group_norm else nn.BatchNorm2d(c_in//4),
            nn.Conv2d(in_channels=c_in//4, out_channels=c_in//4,
                      kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, c_in//4) if group_norm else nn.BatchNorm2d(c_in//4),
            nn.ReLU(),
            # Transpose Conv layer 4.
            nn.ConvTranspose2d(in_channels=c_in//4, out_channels=c_in//8, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, c_in//8) if group_norm else nn.BatchNorm2d(c_in//8),
            nn.Conv2d(in_channels=c_in//8, out_channels=c_in//8,
                      kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, c_in//8) if group_norm else nn.BatchNorm2d(c_in//8),
            nn.ReLU(),
            # Transpose Conv layer 5.
            nn.ConvTranspose2d(in_channels=c_in//8, out_channels=c_out, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.Conv2d(in_channels=c_out, out_channels=c_out,
                      kernel_size=3, stride=1, padding=1),
        ]
        if add_final_sigmoid_layer:
            upscaler_layers.append(nn.Sigmoid())
        if use_softmax_activation:
            upscaler_layers.append(nn.Softmax2d())

        self.upscaler: nn.Module = nn.Sequential(*upscaler_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upscaler(x)


class DoubleConvolutionalUpscaler(nn.Module):

    def __init__(
        self,
        c_in: int,
        c_out: int,
        add_final_sigmoid_layer: bool = True,
        group_norm: bool = False,
        use_softmax_activation: bool = False
    ):
        super().__init__()

        assert not use_softmax_activation or not add_final_sigmoid_layer, \
            "Sigmoid and softmax activations cannot be utilized together."
        assert not use_softmax_activation or c_out == 2, \
            "c_out is required to be 2 when softmax activation is used."

        upscaler_layers: list[nn.Module] = [
            # Transpose Conv layer 1.
            nn.ConvTranspose2d(in_channels=c_in, out_channels=c_in,
                               kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(16, c_in) if group_norm else nn.BatchNorm2d(c_in),
            nn.ReLU(),
            # Transpose Conv layer 2.
            nn.ConvTranspose2d(in_channels=c_in, out_channels=c_in//2, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, c_in//2) if group_norm else nn.BatchNorm2d(c_in//2),
            nn.Conv2d(in_channels=c_in//2, out_channels=c_in//2,
                      kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, c_in//2) if group_norm else nn.BatchNorm2d(c_in//2),
            nn.ReLU(),
            # Transpose Conv layer 3.
            nn.ConvTranspose2d(in_channels=c_in//2, out_channels=c_in//4, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, c_in//4) if group_norm else nn.BatchNorm2d(c_in//4),
            nn.Conv2d(in_channels=c_in//4, out_channels=c_in//4,
                      kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, c_in//4) if group_norm else nn.BatchNorm2d(c_in//4),
            nn.ReLU(),
            # Transpose Conv layer 4.
            nn.ConvTranspose2d(in_channels=c_in//4, out_channels=c_in//8, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, c_in//8) if group_norm else nn.BatchNorm2d(c_in//8),
            nn.Conv2d(in_channels=c_in//8, out_channels=c_in//8,
                      kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, c_in//8) if group_norm else nn.BatchNorm2d(c_in//8),
            nn.ReLU(),
            # Transpose Conv layer 5.
            nn.ConvTranspose2d(in_channels=c_in//8, out_channels=c_out, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.GroupNorm(c_out, c_out) if group_norm else nn.BatchNorm2d(c_out),
            nn.Conv2d(in_channels=c_out, out_channels=c_out,
                      kernel_size=3, stride=1, padding=1),
        ]
        if add_final_sigmoid_layer:
            upscaler_layers.append(nn.Sigmoid())
        if use_softmax_activation:
            upscaler_layers.append(nn.Softmax2d())

        self.upscaler: nn.Module = nn.Sequential(*upscaler_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upscaler(x)


class ConvolutionalUpscaler(nn.Module):

    def __init__(
        self,
        c_in: int,
        c_out: int,
        add_final_sigmoid_layer: bool = True,
        group_norm: bool = False
    ):
        super().__init__()

        upscaler_layers: list[nn.Module] = [
            # Transpose Conv layer 1.
            nn.ConvTranspose2d(in_channels=c_in, out_channels=c_in,
                               kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(16, c_in) if group_norm else nn.BatchNorm2d(c_in),
            # Transpose Conv layer 2.
            nn.ConvTranspose2d(in_channels=c_in, out_channels=c_in // 2, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, c_in // 2) if group_norm else nn.BatchNorm2d(c_in // 2),
            nn.ReLU(),
            # Transpose Conv layer 3.
            nn.ConvTranspose2d(in_channels=c_in // 2, out_channels=c_in // 4, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, c_in // 4) if group_norm else nn.BatchNorm2d(c_in // 4),
            nn.ReLU(),
            # Transpose Conv layer 4.
            nn.ConvTranspose2d(in_channels=c_in // 4, out_channels=c_in // 8, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, c_in // 8) if group_norm else nn.BatchNorm2d(c_in // 8),
            nn.ReLU(),
            # Transpose Conv layer 5.
            nn.ConvTranspose2d(in_channels=c_in // 8, out_channels=c_out, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
        ]
        if add_final_sigmoid_layer:
            upscaler_layers.append(nn.Sigmoid())

        self.upscaler: nn.Module = nn.Sequential(*upscaler_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upscaler(x)


class ResidualUpscaler(nn.Module):

    def __init__(self, c_in: int, c_out: int, add_final_sigmoid_layer: bool = True):
        super().__init__()

        upscaler_layers: list[nn.Module] = [
            ResidualBlock(c_in, c_in),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ResidualBlock(
                c_in, c_in // 2,
                downsample=nn.Sequential(nn.Conv2d(c_in, c_in // 2, kernel_size=1),
                                         nn.BatchNorm2d(c_in // 2))
            ),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ResidualBlock(
                c_in // 2, c_in // 4,
                downsample=nn.Sequential(nn.Conv2d(c_in // 2, c_in // 4, kernel_size=1),
                                         nn.BatchNorm2d(c_in // 4))
            ),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ResidualBlock(
                c_in // 4, c_in // 8,
                downsample=nn.Sequential(nn.Conv2d(c_in // 4, c_in // 8, kernel_size=1),
                                         nn.BatchNorm2d(c_in // 8))
            ),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ResidualBlock(
                c_in // 8, c_out,
                downsample=nn.Sequential(nn.Conv2d(c_in // 8, c_out, kernel_size=1),
                                         nn.BatchNorm2d(c_out)),
                last_layer_relu=False
            ),
        ]
        if add_final_sigmoid_layer:
            upscaler_layers.append(nn.Sigmoid())

        self.upscaler: nn.Module = nn.Sequential(*upscaler_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upscaler(x)
