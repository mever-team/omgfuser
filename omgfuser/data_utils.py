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

import einops
import torch
from torch.nn import functional


def convert_2d_map_to_binary_one_hot(map2d: torch.Tensor) -> torch.Tensor:
    """Converts a 2D single-channel map to a 2D map with one-hot encoding.

    :param map2d:
        Tensor of size: (B, C==1, H, W)
            where:
                - B: Batch size.
                - C: Channels of the map. Currently, C is required to be 1.

    :return:
        Tensor of size: (B, 2, H, W)
    """
    assert map2d.size(dim=1) == 1, "Only single-channel maps are currently supported."

    # Convert the float map to an integer map of indexes.
    map2d = torch.where(map2d > 0.5, 1.0, 0.0)
    map2d = map2d.long()
    map2d = functional.one_hot(
        einops.rearrange(map2d, "b c h w -> b h w c"), num_classes=2
    ).squeeze(dim=3)
    map2d = einops.rearrange(map2d, "b h w c -> b c h w")

    return map2d
