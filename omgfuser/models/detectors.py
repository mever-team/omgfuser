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
from typing import Optional

import torch
from torch import nn

from .transformer import Transformer


class ActivationType(Enum):
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"


class TransformerDetector(nn.Module):

    def __init__(
        self,
        layers: int,
        dim: int,
        out_dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        classification_hidden_dim: int,
        dropout: float = .0,
        last_layer_activation_type: Optional[ActivationType] = ActivationType.SOFTMAX,
    ):
        """

        :param layers:
        :param dim:
        :param out_dim:
        :param heads:
        :param dim_head:
        :param mlp_dim:
        :param classification_hidden_dim: Defines the dimensionality of the hidden MLP layer
            utilized on the final stage. If 0 is provided, then no hidden layer is utilized
            and instead a single-layer MLP is used for classification.
        :param dropout:
        :param last_layer_activation_type:
        """
        super().__init__()

        self.dim: int = dim

        self.transformers: nn.Sequential = nn.Sequential(*[
            Transformer(dim, heads, dim_head, mlp_dim, dropout) for _ in range(layers)
        ])

        self.detection_token = nn.Parameter(torch.zeros((1, dim)))
        nn.init.trunc_normal_(self.detection_token, std=.02)

        # Select between single-layer MLP and two-layer MLP.
        if classification_hidden_dim == 0:
            classification_head_layers: list[nn.Module] = [
                nn.Linear(dim, out_dim)
            ]
        else:
            classification_head_layers: list[nn.Module] = [
                nn.Linear(dim, classification_hidden_dim),
                nn.ReLU(),
                nn.Linear(classification_hidden_dim, out_dim)
            ]

        if last_layer_activation_type is None:
            pass
        elif last_layer_activation_type == ActivationType.SOFTMAX:
            classification_head_layers.append(nn.Softmax(dim=1))
        elif last_layer_activation_type == ActivationType.SIGMOID:
            classification_head_layers.append(nn.Sigmoid())
        else:
            raise RuntimeError(
                f"Non-supported activation function: {last_layer_activation_type.value}"
            )
        self.classification_head: nn.Sequential = nn.Sequential(*classification_head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of size: (B, H*W, C)
        :return: Tensor of size: (B, 1, 1)
        """

        # Expand detection token for each sample in batch.
        det_tokens: torch.Tensor = self.detection_token.expand((x.size(dim=0), 1, self.dim))

        # Add detection token to the sequence.
        x = torch.cat([det_tokens, x], dim=1)

        # Pass the sequence through a number of transformer layers to predict the final token.
        x = self.transformers(x)
        x = x[:, 0, :]

        # Pass the final detection token through an MLP classifier.
        x = self.classification_head(x)

        return x
