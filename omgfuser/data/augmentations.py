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

from typing import Optional

import cv2
import numpy as np
import albumentations as A


class MultiTargetTransformation:
    """Class for transforming differently more targets than the ones supported by Albumentations."""
    def __init__(self):
        self.transformations: list[A.Compose] = []
        self.targets: list[dict[str, str]] = []

    def __call__(self, **targets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outputs: dict[str, np.ndarray] = {}

        for trans, trans_targets in zip(self.transformations, self.targets):
            # Albumentation transformations require "image" and "mask" targets to be present
            # for other targets to be mapped to them. So, if there are targets mapping to them,
            # but they are not included into the targets of current transformation, temporary
            # replace the name of such a target and restore it back after the computation.
            fixed_trans_targets: dict[str, str] = trans_targets.copy()
            fixed_image_target: Optional[str] = None
            fixed_mask_target: Optional[str] = None
            if ("image" in list(fixed_trans_targets.values()) and
                    "image" not in list(fixed_trans_targets.keys())):
                for k, v in fixed_trans_targets.items():
                    if v == "image":
                        fixed_image_target = k
                        break
                fixed_trans_targets = fixed_trans_targets.copy()
                fixed_trans_targets["image"] = fixed_trans_targets.pop(fixed_image_target)
            if ("mask" in list(fixed_trans_targets.values()) and
                    "mask" not in list(fixed_trans_targets.keys())):
                for k, v in fixed_trans_targets.items():
                    if v == "mask":
                        fixed_mask_target = k
                        break
                fixed_trans_targets = fixed_trans_targets.copy()
                fixed_trans_targets["mask"] = fixed_trans_targets.pop(fixed_mask_target)

            # Compute transformed inputs.
            t_in: dict[str, np.ndarray] = {}
            for t in fixed_trans_targets.keys():
                if t == "image" and fixed_image_target is not None:
                    t_actual: str = fixed_image_target
                elif t == "mask" and fixed_mask_target is not None:
                    t_actual: str = fixed_mask_target
                else:
                    t_actual: str = t
                t_in[t] = outputs[t_actual] if t_actual in outputs else targets[t_actual]
            t_outs: dict[str, np.ndarray] = trans(**t_in)

            # Restore fixed targets.
            if fixed_image_target is not None:
                t_outs[fixed_image_target] = t_outs.pop("image")
            if fixed_mask_target is not None:
                t_outs[fixed_mask_target] = t_outs.pop("mask")

            # Update the outputs included in the transformation.
            outputs.update(t_outs)

        return outputs

    def add_transformation(self, transformation: A.Compose, targets: dict[str, str]) -> None:
        self.transformations.append(transformation)
        self.targets.append(targets)


def train_augmentations(
    target_shape: tuple[int, int],
    additional_targets: dict[str, str]
) -> A.Compose:
    return A.Compose(
        [
            # Make all the inputs of equal size first! Otherwise, the rest of transformations
            # will be applied to inputs of different sizes and will be irrelevant.
            A.Resize(target_shape[0]*2, target_shape[1]*2, p=1),
            A.Rotate(limit=(-90, 90), crop_border=True),
            A.RandomResizedCrop(
                height=target_shape[0],
                width=target_shape[1],
                scale=(0.75, 1.0),
                p=1.0
            ),
            A.Flip(p=0.5),
            A.Normalize(mean=0.5,
                        std=0.5,
                        max_pixel_value=255.0)
        ],
        additional_targets=additional_targets,
        is_check_shapes=False
    )


def test_augmentations(target_shape: tuple[int, int], additional_targets: dict[str, str]):
    return A.Compose(
        [
            A.Resize(target_shape[0], target_shape[1], p=1),
            A.Normalize(mean=0.5,
                        std=0.5,
                        max_pixel_value=255.0)
        ],
        additional_targets=additional_targets,
        is_check_shapes=False
    )


def train_augmentations_with_image(
    target_shape: tuple[int, int],
    additional_targets: dict[str, str]
) -> 'MultiTargetTransformation':
    transformations: 'MultiTargetTransformation' = MultiTargetTransformation()
    # Transformations applied to all inputs.
    transformations.add_transformation(A.Compose(
        [
            # Make all the inputs of equal size first! Otherwise, the rest of transformations
            # will be applied to inputs of different sizes and will be irrelevant.
            A.Resize(target_shape[0]*2, target_shape[1]*2, p=1),
            A.Rotate(limit=(-90, 90), crop_border=True),
            A.RandomResizedCrop(
                height=target_shape[0],
                width=target_shape[1],
                scale=(0.75, 1.0),
                p=1.0
            ),
            A.Flip(p=0.5),
        ],
        additional_targets=additional_targets,
        is_check_shapes=False
    ), targets={"image": "image", "mask": "mask", **additional_targets})
    # Normalization applied to the image.
    transformations.add_transformation(A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                        max_pixel_value=255.0)
        ]
    ), targets={"image": "image"})
    if len(additional_targets) > 0:
        # Normalization applied to the rest inputs.
        transformations.add_transformation(A.Compose(
            [
                A.Normalize(mean=0.5, std=0.5, max_pixel_value=255.0)
            ],
            additional_targets=additional_targets
        ), targets=additional_targets)
    return transformations


def test_augmentations_with_image(
    target_shape: tuple[int, int],
    additional_targets: dict[str, str]
) -> 'MultiTargetTransformation':
    transformations: 'MultiTargetTransformation' = MultiTargetTransformation()
    # Transformations applied to all inputs.
    transformations.add_transformation(A.Compose(
        [
            A.Resize(target_shape[0], target_shape[1], p=1),
        ],
        additional_targets=additional_targets,
        is_check_shapes=False
    ), targets={"image": "image", "mask": "mask", **additional_targets})
    # Normalization applied to the image.
    transformations.add_transformation(A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                        max_pixel_value=255.0)
        ]
    ), targets={"image": "image"})
    if len(additional_targets) > 0:
        # Normalization applied to the rest inputs.
        transformations.add_transformation(A.Compose(
            [
                A.Normalize(mean=0.5, std=0.5, max_pixel_value=255.0)
            ],
            additional_targets=additional_targets
        ), targets=additional_targets)
    return transformations


def train_augmentations_geometric_only(
    resize_target: tuple[int, int],
    additional_targets: dict[str, str]
) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=resize_target[0], width=resize_target[1]),
            A.Flip(p=0.7),
            # A.Rotate(limit=(-90, 90), crop_border=True, p=0.5),
            # A.RandomScale(scale_limit=(-0.25, .0), p=0.5)
        ],
        additional_targets=additional_targets,
        is_check_shapes=False
    )


def random_crop_augmentation(
    target_shape: tuple[int, int],
    additional_targets: dict[str, str]
) -> A.Compose:
    return A.Compose(
        [
            A.RandomCrop(height=target_shape[0], width=target_shape[1])
        ],
        additional_targets=additional_targets
    )


def imagenet_normalization_transformation(additional_targets: dict[str, str]) -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                        max_pixel_value=255.0)
        ],
        additional_targets=additional_targets
    )


def common_normalization_transformation(additional_targets: dict[str, str]) -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=0.5, std=0.5, max_pixel_value=255.0)
        ],
        additional_targets=additional_targets
    )


def noiseprint_normalization_transformation(additional_targets: dict[str, str]) -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=0., std=1.0, max_pixel_value=256.0)
        ],
        additional_targets=additional_targets
    )


def resize_transformation(
    target_shape: tuple[int, int],
    additional_targets: dict[str, str]
) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=target_shape[0], width=target_shape[1])
        ],
        additional_targets=additional_targets,
        is_check_shapes=False
    )


def resize_keep_aspect_ratio_transformation(
    max_size: int,
    additional_targets: dict[str, str]
) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max_size)
        ],
        additional_targets=additional_targets
    )


def pad_if_needed_transformation(
    target_shape: tuple[int, int],
    additional_targets: dict[str, str]
) -> A.Compose:
    return A.Compose(
        [
            A.PadIfNeeded(min_height=target_shape[0], min_width=target_shape[1],
                          position=A.PadIfNeeded.PositionType.TOP_LEFT,
                          border_mode=cv2.BORDER_CONSTANT,
                          value=.0)
        ],
        additional_targets=additional_targets
    )
