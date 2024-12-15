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

import random
import unittest
import pathlib
import tempfile
from typing import Set, Any, Union, Optional

import numpy as np
import torch
from torch.utils import data
from PIL import Image

from omgfuser import datasets
from omgfuser import utils


current_dir: pathlib.Path = pathlib.Path(__file__).parent


class TestForensicsDataset(unittest.TestCase):

    def test_loading_dataset(self) -> None:
        test_csv: pathlib.Path = current_dir / "test_data" / "test_data.csv"
        assert test_csv.exists()
        splits: list[datasets.Split] = [datasets.Split.TRAIN_SPLIT,
                                        datasets.Split.EVAL_SPLIT,
                                        datasets.Split.TEST_SPLIT]
        input_signals: list[str] = ["image", "segmentation"]
        input_channels: list[int] = [3, 3]
        target_size: tuple[int, int] = (224, 224)

        for split in splits:
            dataset: datasets.ForensicsDataset = datasets.ForensicsDataset(
                test_csv, current_dir/"test_data", input_signals, input_channels, split, target_size
            )

            # Check that the number of items loaded from the csv is correct.
            self.assertEqual(2, len(dataset))

            # Check that the content of each sample has been loaded correctly.
            manipulated_count: int = 0
            returned_indices: Set[int] = set()
            for i in range(len(dataset)):
                inputs, mask, manipulated, index = dataset[i]

                returned_indices.add(index)

                # Check input signals.
                self.assertEqual(target_size, inputs.shape[1:])
                self.assertEqual(6, inputs.shape[0])
                self.assertAlmostEqual(-1.0, inputs.min())
                self.assertAlmostEqual(1.0, inputs.max())

                # Check mask.
                self.assertEqual(target_size, mask.shape[1:])
                self.assertEqual(1, mask.shape[0])
                if manipulated:
                    self.assertTrue(np.any(mask > 0))
                    manipulated_count += 1
                else:
                    self.assertTrue(np.all(mask == 0))
                self.assertGreaterEqual(mask.min(), 0.0)
                self.assertLessEqual(mask.max(), 1.0)

            # Check that detection labels has been loaded correctly.
            self.assertEqual(1, manipulated_count)
            # Check that all indices have been returned correctly.
            for i in range(0, len(dataset)):
                self.assertIn(i, returned_indices)

    def test_loading_dataset_with_csv_column(self) -> None:
        test_csv: pathlib.Path = current_dir / "test_data" / "test_data.csv"
        assert test_csv.exists()
        splits: list[datasets.Split] = [datasets.Split.TRAIN_SPLIT,
                                        datasets.Split.EVAL_SPLIT,
                                        datasets.Split.TEST_SPLIT]
        input_signals: list[str] = ["image", "segmentation", "segmentation_raw"]
        input_channels: list[int] = [3, 3, 0]
        target_size: tuple[int, int] = (224, 224)

        for split in splits:
            dataset: datasets.ForensicsDataset = datasets.ForensicsDataset(
                test_csv, current_dir/"test_data", input_signals, input_channels, split, target_size
            )

            # Check that the number of items loaded from the csv is correct.
            self.assertEqual(2, len(dataset))

            # Check that the content of each sample has been loaded correctly.
            manipulated_count: int = 0
            returned_indices: Set[int] = set()
            for i in range(len(dataset)):
                inputs, mask, manipulated, index = dataset[i]

                returned_indices.add(index)

                # Check input signals.
                self.assertEqual(target_size, inputs.shape[1:])
                self.assertGreaterEqual(inputs.shape[0], 6)
                self.assertAlmostEqual(-1.0, inputs.min())
                self.assertAlmostEqual(1.0, inputs.max())

                # Check mask.
                self.assertEqual(target_size, mask.shape[1:])
                self.assertEqual(1, mask.shape[0])
                if manipulated:
                    self.assertTrue(np.any(mask > 0))
                    manipulated_count += 1
                else:
                    self.assertTrue(np.all(mask == 0))
                self.assertGreaterEqual(mask.min(), 0.0)
                self.assertLessEqual(mask.max(), 1.0)

            # Check that detection labels has been loaded correctly.
            self.assertEqual(1, manipulated_count)
            # Check that all indices have been returned correctly.
            for i in range(0, len(dataset)):
                self.assertIn(i, returned_indices)

    def test_loading_dataset_with_csv_column_from_lmdb_storage(self) -> None:
        test_csv: pathlib.Path = current_dir / "test_data" / "test_data.csv"
        assert test_csv.exists()
        splits: list[datasets.Split] = [datasets.Split.TRAIN_SPLIT,
                                        datasets.Split.EVAL_SPLIT,
                                        datasets.Split.TEST_SPLIT]
        input_signals: list[str] = ["image", "segmentation", "segmentation_raw"]
        input_channels: list[int] = [3, 3, 0]
        target_size: tuple[int, int] = (224, 224)

        for split in splits:
            dataset: datasets.ForensicsDataset = datasets.ForensicsDataset(
                test_csv, current_dir/"test_data", input_signals, input_channels, split,
                target_size, lmdb_file_storage_path=current_dir/"test_data"/"test_data.lmdb"
            )

            # Check that the number of items loaded from the csv is correct.
            self.assertEqual(2, len(dataset))

            # Check that the content of each sample has been loaded correctly.
            manipulated_count: int = 0
            returned_indices: Set[int] = set()
            for i in range(len(dataset)):
                inputs, mask, manipulated, index = dataset[i]

                returned_indices.add(index)

                # Check input signals.
                self.assertEqual(target_size, inputs.shape[1:])
                self.assertGreaterEqual(inputs.shape[0], 6)
                self.assertAlmostEqual(-1.0, inputs.min())
                self.assertAlmostEqual(1.0, inputs.max())

                # Check mask.
                self.assertEqual(target_size, mask.shape[1:])
                self.assertEqual(1, mask.shape[0])
                if manipulated:
                    self.assertTrue(np.any(mask > 0))
                    manipulated_count += 1
                else:
                    self.assertTrue(np.all(mask == 0))
                self.assertGreaterEqual(mask.min(), 0.0)
                self.assertLessEqual(mask.max(), 1.0)

            # Check that detection labels has been loaded correctly.
            self.assertEqual(1, manipulated_count)
            # Check that all indices have been returned correctly.
            for i in range(0, len(dataset)):
                self.assertIn(i, returned_indices)

    def test_loading_dataset_with_dataloader(self) -> None:
        test_csv: pathlib.Path = current_dir / "test_data" / "test_data.csv"
        assert test_csv.exists()
        splits: list[datasets.Split] = [datasets.Split.TRAIN_SPLIT,
                                        datasets.Split.EVAL_SPLIT,
                                        datasets.Split.TEST_SPLIT]
        input_signals: list[str] = ["image", "segmentation"]
        input_channels: list[int] = [3, 3]
        target_size: tuple[int, int] = (224, 224)
        batch_size: int = 2

        for split in splits:
            dataset: datasets.ForensicsDataset = datasets.ForensicsDataset(
                test_csv, current_dir / "test_data", input_signals, input_channels, split,
                target_size
            )
            dataloader: data.DataLoader = data.DataLoader(dataset,
                                                          batch_size=batch_size,
                                                          num_workers=2)
            returned_indices: Set[int] = set()
            for batch in dataloader:
                inputs, mask, manipulated, indices = batch

                self.assertIsInstance(indices, torch.Tensor)
                returned_indices.update(indices.numpy().tolist())

                self.assertIsInstance(inputs, torch.Tensor)
                self.assertIsInstance(mask, torch.Tensor)
                self.assertEqual(4, len(inputs.size()))
                self.assertEqual(4, len(mask.size()))
                self.assertEqual(
                    torch.Size((batch_size, 6, target_size[0], target_size[1])), inputs.size())
                self.assertEqual(
                    torch.Size((batch_size, 1, target_size[0], target_size[1])), mask.size())
                self.assertEqual(torch.Size((batch_size,)), manipulated.size())
                for i in range(batch_size):
                    if manipulated[i] == 1:
                        self.assertTrue(np.any(mask[i, :, :, :].detach().cpu().numpy() > .0))
                    else:
                        self.assertTrue(np.all(mask[i, :, :, :].detach().cpu().numpy() == .0))

            # Check that all indices have been returned correctly.
            for i in range(0, len(dataset)):
                self.assertIn(i, returned_indices)

    def test_loading_dataset_without_mask_resizing(self) -> None:
        test_csv: pathlib.Path = current_dir / "test_data" / "test_data.csv"
        assert test_csv.exists()
        splits: list[datasets.Split] = [datasets.Split.TRAIN_SPLIT,
                                        datasets.Split.EVAL_SPLIT,
                                        datasets.Split.TEST_SPLIT]
        input_signals: list[str] = ["image", "segmentation"]
        input_channels: list[int] = [3, 3]
        target_size: tuple[int, int] = (224, 224)
        masks_size: tuple[int, int] = (256, 384)  # Specific to the test data.

        for split in splits:
            dataset: datasets.ForensicsDataset = datasets.ForensicsDataset(
                test_csv, current_dir / "test_data", input_signals, input_channels, split,
                target_size, resize_mask=False
            )

            # Check that the number of items loaded from the csv is correct.
            self.assertEqual(2, len(dataset))

            # Check that the content of each sample has been loaded correctly.
            manipulated_count: int = 0
            returned_indices: Set[int] = set()
            for i in range(len(dataset)):
                inputs, mask, manipulated, index = dataset[i]

                returned_indices.add(index)

                # Check input signals.
                self.assertEqual(target_size, inputs.shape[1:])
                self.assertEqual(6, inputs.shape[0])
                self.assertAlmostEqual(-1.0, inputs.min())
                self.assertAlmostEqual(1.0, inputs.max())

                # Check mask.
                self.assertEqual(masks_size, mask.shape[1:])
                self.assertEqual(1, mask.shape[0])
                if manipulated:
                    self.assertTrue(np.any(mask > 0))
                    manipulated_count += 1
                else:
                    self.assertTrue(np.all(mask == 0))
                self.assertGreaterEqual(mask.min(), 0.0)
                self.assertLessEqual(mask.max(), 1.0)

            # Check that detection labels has been loaded correctly.
            self.assertEqual(1, manipulated_count)
            # Check that all indices have been returned correctly.
            for i in range(0, len(dataset)):
                self.assertIn(i, returned_indices)

    def test_loading_dataset_with_imagenet_transformations(self) -> None:
        test_csv: pathlib.Path = current_dir / "test_data" / "test_data.csv"
        assert test_csv.exists()
        splits: list[datasets.Split] = [datasets.Split.TRAIN_SPLIT,
                                        datasets.Split.EVAL_SPLIT,
                                        datasets.Split.TEST_SPLIT]
        input_signals: list[str] = ["image", "segmentation"]
        input_channels: list[int] = [3, 3]
        target_size: tuple[int, int] = (224, 224)

        for split in splits:
            dataset: datasets.ForensicsDataset = datasets.ForensicsDataset(
                test_csv,
                current_dir/"test_data",
                input_signals,
                input_channels,
                split,
                target_size,
                imagenet_image_normalization=True
            )

            # Check that the number of items loaded from the csv is correct.
            self.assertEqual(2, len(dataset))

            # Check that the content of each sample has been loaded correctly.
            manipulated_count: int = 0
            returned_indices: Set[int] = set()
            for i in range(len(dataset)):
                inputs, mask, manipulated, index = dataset[i]

                returned_indices.add(index)

                # Check input signals.
                self.assertEqual(target_size, inputs.shape[1:])
                self.assertEqual(6, inputs.shape[0])
                # self.assertAlmostEqual(-1.0, inputs.min())
                # self.assertAlmostEqual(1.0, inputs.max())

                # Check mask.
                self.assertEqual(target_size, mask.shape[1:])
                self.assertEqual(1, mask.shape[0])
                if manipulated:
                    self.assertTrue(np.any(mask > 0))
                    manipulated_count += 1
                else:
                    self.assertTrue(np.all(mask == 0))
                self.assertGreaterEqual(mask.min(), 0.0)
                self.assertLessEqual(mask.max(), 1.0)

            # Check that detection labels has been loaded correctly.
            self.assertEqual(1, manipulated_count)
            # Check that all indices have been returned correctly.
            for i in range(0, len(dataset)):
                self.assertIn(i, returned_indices)


class TestForensicsDatasetWithAttentionMask(unittest.TestCase):

    def test_loading_dataset_with_csv_column_with_dataloader(self) -> None:
        test_csv: pathlib.Path = current_dir / "test_data" / "test_data.csv"
        assert test_csv.exists()
        splits: list[datasets.Split] = [datasets.Split.TRAIN_SPLIT,
                                        datasets.Split.EVAL_SPLIT,
                                        datasets.Split.TEST_SPLIT]
        input_signals: list[str] = ["image", "segmentation", "segmentation_raw"]
        input_channels: list[int] = [3, 3, 0]
        target_size: tuple[int, int] = (224, 224)
        batch_size: int = 2

        for split in splits:
            dataset: datasets.ForensicsDataset = datasets.ForensicsDatasetWithAttentionMask(
                test_csv, current_dir / "test_data",
                input_signals,
                input_channels,
                split,
                target_size
            )
            dataloader: data.DataLoader = data.DataLoader(dataset,
                                                          batch_size=batch_size,
                                                          num_workers=2)
            returned_indices: Set[int] = set()
            for batch in dataloader:
                inputs, attention_mask, mask, manipulated, indices = batch

                self.assertIsInstance(indices, torch.Tensor)
                returned_indices.update(indices.numpy().tolist())

                self.assertIsInstance(inputs, torch.Tensor)
                self.assertIsInstance(attention_mask, torch.Tensor)
                self.assertIsInstance(mask, torch.Tensor)
                self.assertEqual(4, len(inputs.size()))  # (B, C, H, W)
                self.assertEqual(3, len(attention_mask.size()))  # (B, H*W/16^2, H*W/16^2)
                self.assertEqual(4, len(mask.size()))  # (B, 1, H, W)
                self.assertEqual(
                    torch.Size((batch_size, 6, target_size[0], target_size[1])), inputs.size())
                self.assertEqual(
                    torch.Size((batch_size,
                                target_size[0]*target_size[1]//16**2,
                                target_size[0]*target_size[1]//16**2)),
                    attention_mask.size()
                )
                self.assertEqual(
                    torch.Size((batch_size, 1, target_size[0], target_size[1])), mask.size())
                self.assertEqual(torch.Size((batch_size,)), manipulated.size())
                for i in range(batch_size):
                    if manipulated[i] == 1:
                        self.assertTrue(np.any(mask[i, :, :, :].detach().cpu().numpy() > .0))
                    else:
                        self.assertTrue(np.all(mask[i, :, :, :].detach().cpu().numpy() == .0))

            # Check that all indices have been returned correctly.
            for i in range(0, len(dataset)):
                self.assertIn(i, returned_indices)

    def test_loading_dataset_with_csv_column_with_dataloader_from_lmdb_storage(self) -> None:
        test_csv: pathlib.Path = current_dir / "test_data" / "test_data.csv"
        assert test_csv.exists()
        splits: list[datasets.Split] = [datasets.Split.TRAIN_SPLIT,
                                        datasets.Split.EVAL_SPLIT,
                                        datasets.Split.TEST_SPLIT]
        input_signals: list[str] = ["image", "segmentation", "segmentation_raw"]
        input_channels: list[int] = [3, 3, 0]
        target_size: tuple[int, int] = (224, 224)
        batch_size: int = 2

        for split in splits:
            dataset: datasets.ForensicsDataset = datasets.ForensicsDatasetWithAttentionMask(
                test_csv, current_dir / "test_data",
                input_signals,
                input_channels,
                split,
                target_size,
                lmdb_file_storage_path=current_dir/"test_data"/"test_data.lmdb"
            )
            dataloader: data.DataLoader = data.DataLoader(dataset,
                                                          batch_size=batch_size,
                                                          num_workers=2)
            returned_indices: Set[int] = set()
            for batch in dataloader:
                inputs, attention_mask, mask, manipulated, indices = batch

                self.assertIsInstance(indices, torch.Tensor)
                returned_indices.update(indices.numpy().tolist())

                self.assertIsInstance(inputs, torch.Tensor)
                self.assertIsInstance(attention_mask, torch.Tensor)
                self.assertIsInstance(mask, torch.Tensor)
                self.assertEqual(4, len(inputs.size()))  # (B, C, H, W)
                self.assertEqual(3, len(attention_mask.size()))  # (B, H*W/16^2, H*W/16^2)
                self.assertEqual(4, len(mask.size()))  # (B, 1, H, W)
                self.assertEqual(
                    torch.Size((batch_size, 6, target_size[0], target_size[1])), inputs.size())
                self.assertEqual(
                    torch.Size((batch_size,
                                target_size[0]*target_size[1]//16**2,
                                target_size[0]*target_size[1]//16**2)),
                    attention_mask.size()
                )
                self.assertEqual(
                    torch.Size((batch_size, 1, target_size[0], target_size[1])), mask.size())
                self.assertEqual(torch.Size((batch_size,)), manipulated.size())
                for i in range(batch_size):
                    if manipulated[i] == 1:
                        self.assertTrue(np.any(mask[i, :, :, :].detach().cpu().numpy() > .0))
                    else:
                        self.assertTrue(np.all(mask[i, :, :, :].detach().cpu().numpy() == .0))

            # Check that all indices have been returned correctly.
            for i in range(0, len(dataset)):
                self.assertIn(i, returned_indices)


class TestHandcraftedForensicsSignalsSample(unittest.TestCase):
    def test_augment(self) -> None:
        image_sizes: list[tuple] = [
            (123, 211),
            (224, 224),
            (1024, 768),
            (768, 1024),
        ]

        for img_sz in image_sizes:
            # Create random data.
            image: np.ndarray = np.random.randint(0, 256, (*img_sz, 3)).astype(np.uint8)
            mask: np.ndarray = (np.random.randint(0, 2, (*img_sz, 1)) * 255).astype(np.uint8)
            instances: list[np.ndarray] = [
                (np.random.randint(0, 2, (*img_sz, 1)) * 255).astype(np.uint8) for _ in range(5)
            ]
            manipulated: bool = np.any(mask > 0)

            # Create a new sample and get the model input.
            sample = datasets.HandcraftedForensicsSignalsSample(image, instances, mask, manipulated)
            augmented_sample: datasets.HandcraftedForensicsSignalsSample = sample.augment()

            augmented_size: tuple[int, int] = (augmented_sample.image.shape[0],
                                               augmented_sample.image.shape[1])

            # Check augmented image.
            self.assertEqual(augmented_sample.image.shape[2], sample.image.shape[2])

            # Check augmented instances.
            for inst in augmented_sample.instances:
                self.assertEqual((*augmented_size, 1), inst.shape)

            # Check augmented mask.
            self.assertEqual((*augmented_size, 1), augmented_sample.mask.shape)

            # Check manipulated value.
            self.assertEqual(np.any(augmented_sample.mask > 0), augmented_sample.manipulated)

    def test_to_model_input(self) -> None:
        image_sizes: list[tuple] = [
            (123, 211),
            (224, 224),
            (1024, 768),
            (768, 1024),
            (2048, 2048)
        ]
        model_input_size: tuple[int, int] = (224, 224)

        for img_sz in image_sizes:
            # Create random data.
            image: np.ndarray = np.random.randint(0, 256, (*img_sz, 3)).astype(np.uint8)
            mask: np.ndarray = (np.random.randint(0, 2, (*img_sz, 1)) * 255).astype(np.uint8)
            instances: list[np.ndarray] = [
                (np.random.randint(0, 2, (*img_sz, 1)) * 255).astype(np.uint8) for _ in range(5)
            ]
            manipulated: bool = np.any(mask > 0)

            # Create a new sample and get the model input.
            sample = datasets.HandcraftedForensicsSignalsSample(image, instances, mask, manipulated)
            model_input: dict[str, Union[np.ndarray, int, tuple[int, int]]] = sample.to_model_input(
                model_input_size
            )

            # Validate model input contents.
            expected_keys: list[str] = ["image", "instances", "mask", "manipulated",
                                        "unpadded_size", "original_image"]
            for k in expected_keys:
                self.assertIn(k, model_input)

            # Validate image output.
            self.assertEqual((3, *model_input_size), model_input["image"].shape)

            # Validate instances output.
            self.assertEqual((5, *model_input_size), model_input["instances"].shape)
            self.assertTrue(np.all(
                np.logical_and(model_input["instances"] >= 0, model_input["instances"] <= 1)
            ))

            # Validate mask output.
            self.assertEqual((1, *model_input_size), model_input["mask"].shape)
            self.assertTrue(np.all(
                np.logical_and(model_input["mask"] >= 0, model_input["mask"] <= 1)
            ))

            # Validate manipulated output.
            self.assertEqual(manipulated, model_input["manipulated"])

            # Validate unpadded_size.
            self.assertEqual(model_input_size, model_input["unpadded_size"])

            # Validate original image.
            if img_sz[0] <= 1536 and img_sz[1] <= 1536:
                self.assertEqual((3, *img_sz), model_input["original_image"].shape)
            else:
                self.assertEqual(model_input["original_image"].shape[0], 3)
                self.assertLessEqual(model_input["original_image"].shape[1], 1536)
                self.assertLessEqual(model_input["original_image"].shape[2], 1536)


    def test_to_model_input_padded(self) -> None:
        image_sizes: list[tuple] = [
            (123, 211),
            (224, 224),
            (1024, 768),
            (768, 1024),
        ]
        model_input_size: tuple[int, int] = (224, 224)

        for img_sz in image_sizes:
            # Create random data.
            image: np.ndarray = np.random.randint(0, 256, (*img_sz, 3)).astype(np.uint8)
            mask: np.ndarray = (np.random.randint(0, 2, (*img_sz, 1)) * 255).astype(np.uint8)
            instances: list[np.ndarray] = [
                (np.random.randint(0, 2, (*img_sz, 1)) * 255).astype(np.uint8) for _ in range(5)
            ]
            manipulated: bool = np.any(mask > 0)

            # Create a new sample and get the model input.
            sample = datasets.HandcraftedForensicsSignalsSample(image, instances, mask, manipulated)
            model_input: dict[str, Union[np.ndarray, int, tuple[int, int]]] = sample.to_model_input(
                model_input_size, padded=True
            )

            # Validate model input contents.
            expected_keys: list[str] = ["image", "instances", "mask", "manipulated",
                                        "unpadded_size", "original_image"]
            for k in expected_keys:
                self.assertIn(k, model_input)

            # Validate image output.
            self.assertEqual((3, *model_input_size), model_input["image"].shape)

            # Validate instances output.
            self.assertEqual((5, *model_input_size), model_input["instances"].shape)
            self.assertTrue(np.all(
                np.logical_and(model_input["instances"] >= 0, model_input["instances"] <= 1)
            ))

            # Validate mask output.
            self.assertEqual((1, *model_input_size), model_input["mask"].shape)
            self.assertTrue(np.all(
                np.logical_and(model_input["mask"] >= 0, model_input["mask"] <= 1)
            ))

            # Validate manipulated output.
            self.assertEqual(manipulated, model_input["manipulated"])

            # Validate unpadded_size.
            if img_sz[0] == img_sz[1]:  # img is square
                self.assertEqual(model_input_size, model_input["unpadded_size"])
            elif img_sz[0] < img_sz[1]:  # Height < Width
                self.assertLess(model_input["unpadded_size"][0], model_input["unpadded_size"][1])
                self.assertEqual(model_input_size[1], model_input["unpadded_size"][1])
                self.assertTrue(
                    np.all(model_input["image"][:, model_input["unpadded_size"][0]:, :] == 0)
                )
            else:  # Height > Width
                self.assertLess(model_input["unpadded_size"][1], model_input["unpadded_size"][0])
                self.assertEqual(model_input_size[0], model_input["unpadded_size"][0])
                self.assertTrue(
                    np.all(model_input["image"][:, :, model_input["unpadded_size"][1]:] == 0)
                )

            # Validate original image.
            self.assertEqual((3, *img_sz), model_input["original_image"].shape)

    def test_to_model_input_without_groundtruth(self) -> None:
        image_sizes: list[tuple] = [
            (123, 211),
            (224, 224),
            (1024, 768),
            (768, 1024),
            (2048, 2048)
        ]
        model_input_size: tuple[int, int] = (224, 224)

        for img_sz in image_sizes:
            # Create random data.
            image: np.ndarray = np.random.randint(0, 256, (*img_sz, 3)).astype(np.uint8)
            instances: list[np.ndarray] = [
                (np.random.randint(0, 2, (*img_sz, 1)) * 255).astype(np.uint8) for _ in range(5)
            ]

            # Create a new sample and get the model input.
            sample = datasets.HandcraftedForensicsSignalsSample(image, instances)
            model_input: dict[str, Union[np.ndarray, int, tuple[int, int]]] = sample.to_model_input(
                model_input_size
            )

            # Validate model input contents.
            expected_keys: list[str] = ["image", "instances", "unpadded_size", "original_image"]
            for k in expected_keys:
                self.assertIn(k, model_input)

            # Validate image output.
            self.assertEqual((3, *model_input_size), model_input["image"].shape)

            # Validate instances output.
            self.assertEqual((5, *model_input_size), model_input["instances"].shape)
            self.assertTrue(np.all(
                np.logical_and(model_input["instances"] >= 0, model_input["instances"] <= 1)
            ))

            # Validate unpadded_size.
            self.assertEqual(model_input_size, model_input["unpadded_size"])

            # Validate original image.
            if img_sz[0] <= 1536 and img_sz[1] <= 1536:
                self.assertEqual((3, *img_sz), model_input["original_image"].shape)
            else:
                self.assertEqual(model_input["original_image"].shape[0], 3)
                self.assertLessEqual(model_input["original_image"].shape[1], 1536)
                self.assertLessEqual(model_input["original_image"].shape[2], 1536)


class TestHandcraftedForensicsSignalsDataset(unittest.TestCase):

    def test_loading_dataset_with_csv_column_with_dataloader(self) -> None:
        test_csv: pathlib.Path = current_dir / "test_data" / "test_data.csv"
        assert test_csv.exists()
        splits: list[datasets.Split] = [datasets.Split.TRAIN_SPLIT,
                                        datasets.Split.EVAL_SPLIT,
                                        datasets.Split.TEST_SPLIT]
        keep_aspect_ratio: list[bool] = [True, False]
        input_signals: list[str] = ["image", "segmentation_raw"]
        input_channels: list[int] = [3, 0]
        target_size: tuple[int, int] = (224, 224)
        batch_size: int = 2

        for split in splits:
            for kar in keep_aspect_ratio:
                dataset: datasets.HandcraftedForensicsSignalsDataset = \
                    datasets.HandcraftedForensicsSignalsDataset(
                        test_csv,
                        current_dir / "test_data",
                        input_signals,
                        input_channels,
                        split,
                        target_size,
                        keep_aspect_ratio=kar
                    )
                dataloader: data.DataLoader = data.DataLoader(dataset,
                                                              batch_size=batch_size,
                                                              num_workers=2,
                                                              collate_fn=dataset.build_collate_fn())
                returned_indices: Set[int] = set()
                for batch in dataloader:
                    model_input: dict[str, Any] = batch

                    # Extract the batched entries out of the dict.
                    image: torch.Tensor = model_input["image"]
                    mask: torch.Tensor = model_input["mask"]
                    manipulated: torch.Tensor = model_input["manipulated"]
                    attention_mask: torch.Tensor = model_input["attention_mask"]
                    unpadded_size: tuple[torch.Tensor, torch.Tensor] = model_input["unpadded_size"]
                    original_image: list[torch.Tensor] = model_input["original_image"]
                    indices: torch.Tensor = model_input["index"]

                    # Check the types of items in the mini-batch.
                    self.assertIsInstance(image, torch.Tensor)
                    self.assertIsInstance(mask, torch.Tensor)
                    self.assertIsInstance(manipulated, torch.Tensor)
                    self.assertIsInstance(attention_mask, torch.Tensor)
                    self.assertIsInstance(unpadded_size, list)
                    for v in unpadded_size:
                        self.assertIsInstance(v, torch.Tensor)
                    self.assertIsInstance(original_image, list)
                    for v in original_image:
                        self.assertIsInstance(v, torch.Tensor)
                    self.assertIsInstance(indices, torch.Tensor)

                    # Keep the indices of the items in the mini-batch for later checks.
                    returned_indices.update(indices.numpy().tolist())

                    self.assertEqual(  # (B, C, H, W)
                        torch.Size((batch_size, 3, *target_size)), image.size()
                    )
                    self.assertEqual(  # (B, 1, H, W)
                        torch.Size((batch_size, 1, *target_size)), mask.size()
                    )
                    self.assertEqual(  # (B, )
                        torch.Size((batch_size,)), manipulated.size()
                    )
                    self.assertEqual(  # (B, H*W/16^2, H*W/16^2)
                        torch.Size((batch_size,
                                    target_size[0]*target_size[1]//16**2,
                                    target_size[0]*target_size[1]//16**2)),
                        attention_mask.size()
                    )
                    for v1, v2 in zip(unpadded_size, target_size):
                        self.assertTrue(torch.all(torch.less_equal(v1, v2)))
                    for i in range(batch_size):
                        if manipulated[i] == 1:
                            self.assertTrue(np.any(mask[i, :, :, :].detach().cpu().numpy() > .0))
                        else:
                            self.assertTrue(np.all(mask[i, :, :, :].detach().cpu().numpy() == .0))
                    self.assertEqual(batch_size, len(original_image))
                    for img in original_image:
                        self.assertEqual(3, len(img.size()))
                        self.assertEqual(3, img.size(dim=0))
                        if split != datasets.Split.TRAIN_SPLIT:
                            self.assertEqual(torch.Size((3, 256, 384)), img.size())

                # Check that all indices have been returned correctly.
                for i in range(0, len(dataset)):
                    self.assertIn(i, returned_indices)

    def test_loading_dataset_with_csv_column_with_dataloader_from_lmdb_storage(self) -> None:
        test_csv: pathlib.Path = current_dir / "test_data" / "test_data.csv"
        assert test_csv.exists()
        splits: list[datasets.Split] = [datasets.Split.TRAIN_SPLIT,
                                        datasets.Split.EVAL_SPLIT,
                                        datasets.Split.TEST_SPLIT]
        input_signals: list[str] = ["image", "segmentation_raw"]
        input_channels: list[int] = [3, 0]
        target_size: tuple[int, int] = (224, 224)
        batch_size: int = 2

        for split in splits:
            dataset: datasets.HandcraftedForensicsSignalsDataset = \
                datasets.HandcraftedForensicsSignalsDataset(
                    test_csv,
                    current_dir / "test_data",
                    input_signals,
                    input_channels,
                    split,
                    target_size,
                    lmdb_file_storage_path=current_dir/"test_data"/"test_data.lmdb"
                )
            dataloader: data.DataLoader = data.DataLoader(dataset,
                                                          batch_size=batch_size,
                                                          num_workers=2,
                                                          collate_fn=dataset.build_collate_fn())
            returned_indices: Set[int] = set()
            for batch in dataloader:
                model_input: dict[str, Any] = batch

                # Extract the batched entries out of the dict.
                image: torch.Tensor = model_input["image"]
                mask: torch.Tensor = model_input["mask"]
                manipulated: torch.Tensor = model_input["manipulated"]
                attention_mask: torch.Tensor = model_input["attention_mask"]
                unpadded_size: tuple[torch.Tensor, torch.Tensor] = model_input["unpadded_size"]
                original_image: list[torch.Tensor] = model_input["original_image"]
                indices: torch.Tensor = model_input["index"]

                # Check the types of items in the mini-batch.
                self.assertIsInstance(image, torch.Tensor)
                self.assertIsInstance(mask, torch.Tensor)
                self.assertIsInstance(manipulated, torch.Tensor)
                self.assertIsInstance(attention_mask, torch.Tensor)
                self.assertIsInstance(unpadded_size, list)
                for v in unpadded_size:
                    self.assertIsInstance(v, torch.Tensor)
                self.assertIsInstance(original_image, list)
                for v in original_image:
                    self.assertIsInstance(v, torch.Tensor)
                self.assertIsInstance(indices, torch.Tensor)

                # Keep the indices of the items in the mini-batch for later checks.
                returned_indices.update(indices.numpy().tolist())

                self.assertEqual(  # (B, C, H, W)
                    torch.Size((batch_size, 3, *target_size)), image.size()
                )
                self.assertEqual(  # (B, 1, H, W)
                    torch.Size((batch_size, 1, *target_size)), mask.size()
                )
                self.assertEqual(  # (B, )
                    torch.Size((batch_size,)), manipulated.size()
                )
                self.assertEqual(  # (B, H*W/16^2, H*W/16^2)
                    torch.Size((batch_size,
                                target_size[0] * target_size[1] // 16 ** 2,
                                target_size[0] * target_size[1] // 16 ** 2)),
                    attention_mask.size()
                )
                for v1, v2 in zip(unpadded_size, target_size):
                    self.assertTrue(torch.all(torch.less_equal(v1, v2)))
                for i in range(batch_size):
                    if manipulated[i] == 1:
                        self.assertTrue(np.any(mask[i, :, :, :].detach().cpu().numpy() > .0))
                    else:
                        self.assertTrue(np.all(mask[i, :, :, :].detach().cpu().numpy() == .0))
                self.assertEqual(batch_size, len(original_image))
                for img in original_image:
                    self.assertEqual(3, len(img.size()))
                    self.assertEqual(3, img.size(dim=0))
                    if split != datasets.Split.TRAIN_SPLIT:
                        self.assertEqual(torch.Size((3, 256, 384)), img.size())

            # Check that all indices have been returned correctly.
            for i in range(0, len(dataset)):
                self.assertIn(i, returned_indices)

    def test_loading_large_number_of_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_dir: pathlib.Path = pathlib.Path(tmp_str)
            dataset_csv: pathlib.Path = create_random_dataset(8, tmp_dir)
            assert dataset_csv.exists()
            splits: list[datasets.Split] = [datasets.Split.TRAIN_SPLIT]
            input_signals: list[str] = ["image", "instances"]
            input_channels: list[int] = [3, 0]
            target_size: tuple[int, int] = (224, 224)
            batch_size: int = 2

            for split in splits:
                dataset: datasets.HandcraftedForensicsSignalsDataset = \
                    datasets.HandcraftedForensicsSignalsDataset(
                        dataset_csv,
                        tmp_dir,
                        input_signals,
                        input_channels,
                        split,
                        target_size,
                    )
                dataloader: data.DataLoader = data.DataLoader(dataset,
                                                              batch_size=batch_size,
                                                              num_workers=2,
                                                              collate_fn=dataset.build_collate_fn())
                returned_indices: Set[int] = set()
                for batch in dataloader:
                    pass

    def test_stratified_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_dir: pathlib.Path = pathlib.Path(tmp_str)
            dataset_csv: pathlib.Path = create_random_dataset(8, tmp_dir, manipulated_num=6)
            assert dataset_csv.exists()
            splits: list[datasets.Split] = [datasets.Split.TRAIN_SPLIT]
            input_signals: list[str] = ["image", "instances"]
            input_channels: list[int] = [3, 0]
            target_size: tuple[int, int] = (224, 224)
            batch_size: int = 2

            for split in splits:
                dataset: datasets.HandcraftedForensicsSignalsDataset = \
                    datasets.HandcraftedForensicsSignalsDataset(
                        dataset_csv,
                        tmp_dir,
                        input_signals,
                        input_channels,
                        split,
                        target_size,
                        stratify=True
                    )
                dataloader: data.DataLoader = data.DataLoader(dataset,
                                                              batch_size=batch_size,
                                                              num_workers=2,
                                                              collate_fn=dataset.build_collate_fn())
                authentic_count: int = 0
                manipulated_count: int = 0
                for batch in dataloader:
                    for i in range(batch_size):
                        if batch["manipulated"][i].item() == 1:
                            manipulated_count += 1
                        else:
                            authentic_count += 1
                self.assertEqual(authentic_count, manipulated_count)
                self.assertEqual(6, authentic_count)


def create_random_dataset(
    samples_num: int,
    out_dir: pathlib.Path,
    signals_num: int = 1,
    manipulated_num: Optional[int] = None
) -> pathlib.Path:
    if manipulated_num is None:
        entries: list[dict[str, str]] = [
            create_random_sample(out_dir, str(i), signals_num) for i in range(samples_num)
        ]
    else:
        authentic_num: int = samples_num - manipulated_num
        assert authentic_num >= 0
        entries: list[dict[str, str]] = [
            create_random_sample(out_dir, str(i), signals_num, manipulated=False)
            for i in range(authentic_num)
        ]
        entries.extend([
            create_random_sample(out_dir, str(i), signals_num, manipulated=True)
            for i in range(authentic_num, authentic_num+manipulated_num)
        ])

    csv_path: pathlib.Path = out_dir / "random_dataset.csv"
    utils.write_csv_file(entries, csv_path)
    return csv_path


def create_random_sample(
    out_dir: pathlib.Path,
    name: str,
    signals_num: int,
    manipulated: Optional[bool] = None
) -> dict[str, str]:
    sample_dir: pathlib.Path = out_dir / name
    sample_dir.mkdir()

    sample_height: int = random.randint(1024, 3092)
    sample_width: int = random.randint(1024, 3092)

    # Create random image
    image: np.ndarray = np.random.randint(0, 256, (sample_height, sample_width, 3), dtype=np.uint8)
    image_path: pathlib.Path = sample_dir / "image.jpg"
    Image.fromarray(image).save(image_path)

    if manipulated is None:
        # Decide whether image is manipulated or not.
        manipulated: bool = bool(random.randint(0, 1))

    # Create random mask
    if manipulated:
        mask: np.ndarray = np.random.randint(
            0, 2, (sample_height, sample_width), dtype=np.uint8
        ) * 255
    else:
        mask: np.ndarray = np.zeros((sample_height, sample_width), dtype=np.uint8)
    mask_path: pathlib.Path = sample_dir / "mask.png"
    Image.fromarray(mask).save(mask_path)

    # Create random instances
    instances_num: int = random.randint(1, 120)
    instances_entries: list[dict[str, str]] = []
    instances_dir: pathlib.Path = sample_dir / "instances"
    instances_dir.mkdir()
    for i in range(instances_num):
        instance_mask: np.ndarray = np.random.randint(
            0, 2, (sample_height, sample_width), dtype=np.uint8
        ) * 255
        instance_path: pathlib.Path = instances_dir / f"{i}.png"
        Image.fromarray(instance_mask).save(instance_path)
        instances_entries.append({
            "seg_model": "random",
            "seg_map": instance_path.name,
            "seg_score": 1.0,
            "class_id": 0
        })
    instances_csv_path: pathlib.Path = instances_dir / "instances.csv"
    utils.write_csv_file(instances_entries, instances_csv_path)

    # Create random signals
    signal_names: list[str] = [f"signal_{i}" for i in range(signals_num)]
    signal_paths: list[pathlib.Path] = [sample_dir / f"signal_{i}.png" for i in range(signals_num)]
    for s_path in signal_paths:
        signal: np.ndarray = np.random.randint(
            0, 256, (sample_height, sample_width), dtype=np.uint8
        )
        Image.fromarray(signal).save(s_path)

    return {
        "image": str(image_path.relative_to(out_dir)),
        "mask": str(mask_path.relative_to(out_dir)),
        "split": "train",
        "detection": "TRUE" if manipulated else "FALSE",
        "instances": str(instances_csv_path.relative_to(out_dir)),
        **{s_name: s_path.relative_to(out_dir)
           for s_name, s_path in zip(signal_names, signal_paths)}
    }
