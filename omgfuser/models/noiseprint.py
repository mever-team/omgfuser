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

import pathlib

import torch
from torch import nn

from . import DnCNN


class NoiseprintPlusPlus(nn.Module):
    def __init__(self, pretrained_weights: pathlib.Path = "checkpoints/noiseprintplusplus.pth"):
        super().__init__()

        num_levels: int = 17
        out_channel: int = 1
        self.dncnn = DnCNN.make_net(
            3, kernels=[3, ] * num_levels,
            features=[64, ] * (num_levels - 1) + [out_channel],
            bns=[False, ] + [True, ] * (num_levels - 2) + [False, ],
            acts=['relu', ] * (num_levels - 1) + ['linear', ],
            dilats=[1, ] * num_levels,
            bn_momentum=0.1, padding=1
        )

        weights = torch.load(pretrained_weights, map_location="cpu")
        self.load_state_dict(weights["state_dict"])

        # Freeze the model.
        for p in self.dncnn.parameters():
            p.requires_grad = False

    def forward(self, rgb):
        modal_x = self.dncnn(rgb)
        return modal_x
