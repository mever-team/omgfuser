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
import einops

from omgfuser.models import attention_mask


class TestAttentionMask(unittest.TestCase):

    def test_instances_to_model_ready_attention_mask_with_no_instances(self) -> None:
        # Generate three instance maps with rectangular instance objects.
        instances: torch.Tensor = torch.zeros([3, 224, 224])
        instances[0, 10:20, 10:20] = 1
        instances[1, 50:70, 50:70] = 1
        instances[2, 120:140, 180:210] = 1

        mask: torch.Tensor = attention_mask.instances_to_model_ready_attention_mask(
            instances, (224, 224)
        )

        self.assertEqual(torch.Size([14 * 14, 14 * 14]), mask.size())
        self.assertFalse(torch.all(mask == 0))
        self.assertFalse(torch.all(mask == 1))

    def test_instances_to_model_ready_attention_mask_with_no_instances(self) -> None:
        mask: torch.Tensor = attention_mask.instances_to_model_ready_attention_mask(
            None, (224, 224)
        )

        self.assertEqual(torch.Size([14*14, 14*14]), mask.size())
        self.assertTrue(torch.all(mask == 1))

    def test_instances_to_patched_attention_mask_with_equal_size(self) -> None:
        instances: torch.Tensor = torch.Tensor(
            [
                [
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ]
        )

        mask: torch.Tensor = attention_mask.instances_to_patched_attention_mask(instances, (10, 10))
        self.assertEqual(torch.Size([100, 100]), mask.size())

        merged_instances, _ = torch.max(instances, dim=0)
        merged_instances_flat = einops.rearrange(merged_instances, "h w -> (h w)")
        masked_merged_instances = merged_instances_flat.unsqueeze(dim=1)
        masked_merged_instances = torch.matmul(mask, masked_merged_instances.int())
        masked_merged_instances = masked_merged_instances.squeeze(dim=1)
        masked_merged_instances = einops.rearrange(masked_merged_instances, "(h w) -> h w", h=10)

        self.assertTrue(torch.all(
            merged_instances.int() == torch.clip(masked_merged_instances, 0, 1)
        ))

    def test_instances_to_patched_attention_mask_with_smaller_size(self) -> None:
        instances: torch.Tensor = torch.Tensor(
            [
                [
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ]
        )

        mask: torch.Tensor = attention_mask.instances_to_patched_attention_mask(instances, (5, 5))
        self.assertEqual(torch.Size([25, 25]), mask.size())

        merged_instances, _ = torch.max(instances, dim=0)
        merged_instances = torch.nn.functional.interpolate(
            merged_instances.unsqueeze(dim=0).unsqueeze(dim=0), (5, 5), mode="area")
        merged_instances = torch.where(merged_instances > 0, 1, 0)
        merged_instances_flat = einops.rearrange(merged_instances.squeeze(), "h w -> (h w)")
        masked_merged_instances = merged_instances_flat.unsqueeze(dim=1)
        masked_merged_instances = torch.matmul(mask, masked_merged_instances.int())
        masked_merged_instances = masked_merged_instances.squeeze(dim=1)
        masked_merged_instances = einops.rearrange(masked_merged_instances, "(h w) -> h w", h=5)

        self.assertTrue(torch.all(
            merged_instances.int() == torch.clip(masked_merged_instances, 0, 1)
        ))

    def test_instances_to_patched_attention_mask_with_greater_size(self) -> None:
        instances: torch.Tensor = torch.Tensor(
            [
                [
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ]
        )

        mask: torch.Tensor = attention_mask.instances_to_patched_attention_mask(instances, (20, 20))
        self.assertEqual(torch.Size([400, 400]), mask.size())

    def test_instances_to_attention_mask(self) -> None:
        instances: torch.Tensor = torch.Tensor(
            [
                [
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ]
        )

        mask: torch.Tensor = attention_mask.instances_to_attention_mask(instances)
        self.assertEqual(torch.Size([100, 100]), mask.size())

        merged_instances, _ = torch.max(instances, dim=0)
        merged_instances_flat = einops.rearrange(merged_instances, "h w -> (h w)")
        masked_merged_instances = merged_instances_flat.unsqueeze(dim=1)
        masked_merged_instances = torch.matmul(mask, masked_merged_instances.int())
        masked_merged_instances = masked_merged_instances.squeeze(dim=1)
        masked_merged_instances = einops.rearrange(masked_merged_instances, "(h w) -> h w", h=10)

        self.assertTrue(torch.all(
            merged_instances.int() == torch.clip(masked_merged_instances, 0, 1)
        ))
