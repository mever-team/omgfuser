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
import pathlib

import numpy as np
from PIL import Image

from omgfuser.data import dct_loader


current_dir: pathlib.Path = pathlib.Path(__file__).parent


class TestDCTLoader(unittest.TestCase):

    def test_create_dct_volume(self) -> None:
        image_path: pathlib.Path = current_dir / "test_data" / "dctvol_test_image.jpg"
        ground_truth_path: pathlib.Path = current_dir / "test_data" / "example_image_dctvol.npz"

        dct_vol, qtable = dct_loader.create_dct_volume(image_path)

        ground_truth_data = np.load(str(ground_truth_path))
        gt_dct_vol = ground_truth_data["dctvol"]
        gt_qtable = ground_truth_data["qtable"]

        self.assertTrue(np.all(np.equal(dct_vol.numpy(), gt_dct_vol)))
        self.assertTrue(np.all(np.equal(qtable.numpy(), gt_qtable)))

    def test_create_dct_volume_with_filelike(self) -> None:
        image_path: pathlib.Path = current_dir / "test_data" / "dctvol_test_image.jpg"
        ground_truth_path: pathlib.Path = current_dir / "test_data" / "example_image_dctvol.npz"

        dct_vol, qtable = dct_loader.create_dct_volume(image_path.open(mode="rb"))

        ground_truth_data = np.load(str(ground_truth_path))
        gt_dct_vol = ground_truth_data["dctvol"]
        gt_qtable = ground_truth_data["qtable"]

        self.assertTrue(np.all(np.equal(dct_vol.numpy(), gt_dct_vol)))
        self.assertTrue(np.all(np.equal(qtable.numpy(), gt_qtable)))

    def test_create_dct_volume_with_png_image(self) -> None:
        image_path: pathlib.Path = current_dir / "test_data" / "normal-07.png"
        dct_vol, qtable = dct_loader.create_dct_volume(image_path)

        self.assertIsNotNone(dct_vol)
        self.assertIsNotNone(qtable)

    def test_create_dct_volume_with_filelike_png_image(self) -> None:
        image_path: pathlib.Path = current_dir / "test_data" / "normal-07.png"
        dct_vol, qtable = dct_loader.create_dct_volume(image_path.open(mode="rb"))

        self.assertIsNotNone(dct_vol)
        self.assertIsNotNone(qtable)

    def test_create_dct_volume_with_augmented(self) -> None:
        image_path: pathlib.Path = current_dir / "test_data" / "dctvol_test_image.jpg"
        ground_truth_path: pathlib.Path = current_dir / "test_data" / "example_image_dctvol.npz"

        with Image.open(image_path) as img:
            img_arr = np.array(img.convert("RGB"))
        dct_vol, qtable = dct_loader.create_dct_volume_from_augmented(image_path, img_arr)

        ground_truth_data = np.load(str(ground_truth_path))
        gt_dct_vol = ground_truth_data["dctvol"]
        gt_qtable = ground_truth_data["qtable"]

        self.assertEqual(dct_vol.numpy().shape, gt_dct_vol.shape)
        self.assertTrue(np.all(np.equal(qtable.numpy(), gt_qtable)))

    def test_create_dct_volume_with_augmented_filelike(self) -> None:
        image_path: pathlib.Path = current_dir / "test_data" / "dctvol_test_image.jpg"
        ground_truth_path: pathlib.Path = current_dir / "test_data" / "example_image_dctvol.npz"

        with Image.open(image_path) as img:
            img_arr = np.array(img.convert("RGB"))
        dct_vol, qtable = dct_loader.create_dct_volume_from_augmented(
            image_path.open(mode="rb"), img_arr
        )

        ground_truth_data = np.load(str(ground_truth_path))
        gt_dct_vol = ground_truth_data["dctvol"]
        gt_qtable = ground_truth_data["qtable"]

        self.assertEqual(dct_vol.numpy().shape, gt_dct_vol.shape)
        self.assertTrue(np.all(np.equal(qtable.numpy(), gt_qtable)))

    def test_create_dct_volume_with_augmented_png_image(self) -> None:
        image_path: pathlib.Path = current_dir / "test_data" / "normal-07.png"

        with Image.open(image_path) as img:
            img_arr = np.array(img.convert("RGB"))
        dct_vol, qtable = dct_loader.create_dct_volume_from_augmented(
            image_path.open(mode="rb"), img_arr
        )
        dct_vol, qtable = dct_loader.create_dct_volume(image_path)

        self.assertIsNotNone(dct_vol)
        self.assertIsNotNone(qtable)

    def test_create_dct_volume_with_augmented_filelike_png_image(self) -> None:
        image_path: pathlib.Path = current_dir / "test_data" / "normal-07.png"

        with Image.open(image_path) as img:
            img_arr = np.array(img.convert("RGB"))
        dct_vol, qtable = dct_loader.create_dct_volume_from_augmented(
            image_path.open(mode="rb"), img_arr
        )

        self.assertIsNotNone(dct_vol)
        self.assertIsNotNone(qtable)
