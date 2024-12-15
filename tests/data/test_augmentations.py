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

import numpy as np
import albumentations as A

from omgfuser.data import augmentations


class TestMultiTargetTransformation(unittest.TestCase):

    def test_two_branches(self) -> None:
        transformations: augmentations.MultiTargetTransformation = \
            augmentations.MultiTargetTransformation()
        # Transformations applied to all inputs.
        transformations.add_transformation(A.Compose(
            [
                A.Resize(224, 224, p=1)
            ],
            additional_targets={"signal_1": "image"}
        ), targets={"image": "image", "mask": "mask", "signal_1": "image"})
        # Normalization applied to the image.
        transformations.add_transformation(A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225),
                            max_pixel_value=255.0)
            ]
        ), targets={"image": "image"})
        # Normalization applied to the rest inputs.
        transformations.add_transformation(A.Compose(
            [
                A.Normalize(mean=0.5, std=0.5, max_pixel_value=255.0)
            ],
            additional_targets={"signal_1": "image"}
        ), targets={"mask": "mask", "signal_1": "image"})

        image: np.ndarray = np.random.randint(0, 256, (256, 256, 3)).astype(np.uint8)
        signal_1: np.ndarray = np.random.randint(0, 256, (256, 256, 1)).astype(np.uint8)
        mask: np.ndarray = np.random.randint(0, 256, (256, 256, 1)).astype(np.uint8)

        transformed: dict[str, np.ndarray] = transformations(**{
            "image": image,
            "mask": mask,
            "signal_1": signal_1
        })

        # Check transformed image.
        t_image: np.ndarray = transformed["image"]
        self.assertEqual((224, 224, 3), t_image.shape)
        restored_image: np.ndarray = (t_image * (np.array([0.229, 0.224, 0.225]) * 255.0) +
                                      (np.array([0.485, 0.456, 0.406]) * 255.0))
        self.assertTrue(np.all(restored_image >= 0.))
        self.assertTrue(np.all(restored_image <= 255.))
        # Check transformed signal 1.
        t_signal: np.ndarray = transformed["signal_1"]
        self.assertEqual((224, 224, 1), t_signal.shape)
        restored_signal: np.ndarray = (t_signal * 0.5 * 255.0) + (0.5 * 255.0)
        self.assertTrue(np.all(restored_signal >= 0.))
        self.assertTrue(np.all(restored_signal <= 255.))
        # Check transformed mask.
        t_mask: np.ndarray = transformed["mask"]
        self.assertEqual((224, 224, 1), t_mask.shape)
        self.assertTrue(np.all(t_mask >= 0.))
        self.assertTrue(np.all(t_mask <= 255.))
