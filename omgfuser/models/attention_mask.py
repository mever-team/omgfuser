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
from typing import Optional

import numpy as np
import torch
import einops


def instances_to_model_ready_attention_mask(
    instances: Optional[torch.Tensor],
    model_input_size: tuple[int, int]
) -> torch.Tensor:
    """Converts a set of instance maps to an attention mask ready for input to the model.

    The case of zero instance maps is handled by generating an attention mask
    that treats the whole image as background (attention is enabled between all
    image patches).

    :param instances: A tensor of size (I, H, W), containing I instance maps,
        when I > 0. When I==0, None should be given to generate the default map.
    :param model_input_size: The dimensions of the images provided to the model -> (H, W).
        They should be dividable by 16.

    :return: A tensor of size (H*W/16^2, H*W/16^2) containing the attention mask for all
        the regions in the image.
    """
    assert model_input_size[0] % 16 == 0 and model_input_size[1] % 16 == 0
    mask_size: tuple[int, int] = (model_input_size[0] // 16, model_input_size[1] // 16)

    if instances is None:
        instances = torch.zeros((1, model_input_size[0], model_input_size[1]))

    return instances_to_patched_attention_mask(instances, mask_size)


def instances_to_patched_attention_mask(
    instances: torch.Tensor,
    mask_size: tuple[int, int],
    add_background_mask: bool = True
) -> torch.Tensor:
    """Converts a set of instance maps to an attention mask of requested size.

    :param instances: A tensor of size (I, H, W), containing I instance maps.
    :param mask_size: A two-tuple containing the target height and width of
        the image for which the attention mask will be generated. -> (h, w)
    :param add_background_mask: When True, an additional background mask is generated
        for the areas not covered by any of the provided masks.

    :return: A tensor of size (h*w, h*w) containing the attention mask for all
        the regions in the image.
    """
    # Scale the map to the target size. Area interpolation is utilized since it splits
    # the images into patches and computes the average for each patch. So, when
    # the average inside each patch is not zero, we can say that this patch
    # belongs to an instance map. So, after interpolation, an addition binarization
    # step is performed, to conclude which patches belong to an instance map or not,
    instances = instances.unsqueeze(dim=1)
    instances = torch.nn.functional.interpolate(
        instances, mask_size, mode="area",
    )
    instances = torch.where(instances > 0, 1, 0)
    instances = instances.squeeze(dim=1)

    mask: torch.Tensor = instances_to_attention_mask(
        instances, add_background_mask=add_background_mask
    )

    return mask


def instances_to_attention_mask(
    instances: torch.Tensor,
    add_background_mask: bool = True
) -> torch.Tensor:
    """Converts a set of instance maps to an attention mask.

    :param instances: A tensor of size (I, H, W), containing I instance maps.
    :param add_background_mask: When True, an additional background mask is generated
        for the areas not covered by any of the provided masks.

    :return: A tensor of size (H*W, H*W) containing the attention mask for all
        the regions in the image.
    """
    sequence_length: int = instances.size(dim=1) * instances.size(dim=2)
    mask: torch.Tensor = torch.zeros((sequence_length, sequence_length), dtype=torch.int32)

    if add_background_mask:
        merged_instances, _ = torch.max(instances, dim=0, keepdim=True)
        background_mask = torch.where(merged_instances > 0, 0, 1)
        instances = torch.cat((instances, background_mask), dim=0)

    for i in range(instances.size(dim=0)):
        instance = einops.rearrange(instances[i, :, :], "h w -> (h w)")
        indices_to_attend: torch.Tensor = torch.argwhere(instance)
        instance_mask = torch.zeros((sequence_length, sequence_length), dtype=torch.int32)
        instance_mask[indices_to_attend, :] = instance.int()

        mask = torch.maximum(mask, instance_mask)

    return mask


def model_ready_attention_mask_to_attention_region(att_mask: torch.Tensor) -> list[np.ndarray]:
    attention_regions: set[tuple] = set()
    for c in range(att_mask.size(dim=1)):
        region = tuple(i for i in att_mask[:, c].detach().cpu().numpy().flatten())
        attention_regions.add(region)

    att_reg_dim: int = int(math.sqrt(att_mask.size(dim=0)))

    attention_maps: list[np.ndarray] = []
    for att_reg in attention_regions:
        att_reg_map = np.array(att_reg).reshape((att_reg_dim, att_reg_dim))
        np.repeat(np.repeat(att_reg_map, 16, axis=0), 16, axis=1)
        attention_maps.append(att_reg_map)

    attention_maps.sort(key=lambda x: np.sum(x), reverse=True)

    return attention_maps
