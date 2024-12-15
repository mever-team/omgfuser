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

import csv
import functools
import pathlib
import random
import timeit
from enum import Enum
from typing import Any, Optional, Callable, Union, Iterable
import logging

import cv2
import numpy as np
import albumentations as A
import torch
from torch.utils.data import Dataset

from omgfuser.models import attention_mask
from omgfuser.data import filestorage, readers, augmentations, processors

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ForensicsDatasetSample:
    def __init__(self, inputs: list[np.ndarray], mask: np.ndarray, manipulated: bool):
        self.inputs: list[np.ndarray] = inputs
        self.mask: np.ndarray = mask
        self.manipulated: bool = manipulated

    def transform(self,
                  transformations: A.Compose,
                  additional_targets: dict[str, str],
                  transform_mask: bool) -> 'ForensicsDatasetSample':
        assert len(additional_targets) == len(self.inputs) - 1

        transorm_targets: dict[str, np.ndarray] = {
            "image": self.inputs[0],
            "mask": self.mask
        }
        additional_keys: list[str] = list(additional_targets.keys())
        for k, inp in zip(additional_keys, self.inputs[1:]):
            transorm_targets[k] = inp

        transformed: dict[str, np.ndarray] = transformations(**transorm_targets)
        transformed_mask: np.ndarray = transformed["mask"] if transform_mask else self.mask
        transformed_inputs: list[np.ndarray] = [transformed["image"]]
        for k in additional_keys:
            transformed_inputs.append(transformed[k])

        return ForensicsDatasetSample(transformed_inputs, transformed_mask, self.manipulated)

    def to_model_sample(self) -> tuple[np.ndarray, np.ndarray, int]:
        cat_input: np.ndarray = np.concatenate(self.inputs, axis=2)  # Cat over channels axis.
        cat_input = cat_input.transpose((2, 0, 1))
        mask: np.ndarray = self.mask.transpose((2, 0, 1)) / 255.0
        return cat_input, mask, int(self.manipulated)


class HandcraftedForensicsSignalsSample:
    def __init__(
        self,
        image: np.ndarray,
        instances: list[np.ndarray],
        mask: Optional[np.ndarray] = None,
        manipulated: Optional[bool] = None
    ):
        self.image: np.ndarray = image
        self.instances: list[np.ndarray] = instances
        self.mask: Optional[np.ndarray] = mask
        self.manipulated: Optional[bool] = manipulated

    def augment(
        self,
        max_size: int = 1536
    ) -> 'HandcraftedForensicsSignalsSample':
        # Image, instance maps and ground-truth mask should get the same geometric transformations.
        additional_targets: dict[str, str] = {
            f"instance_{i + 1}": "image" for i in range(len(self.instances))
        }
        transform_targets: dict[str, np.ndarray] = {
            "image": self.image,
            "mask": self.mask
        }
        additional_keys: list[str] = list(additional_targets.keys())
        for k, inp in zip(additional_keys, self.instances):
            transform_targets[k] = inp

        original_height: int = self.image.shape[0]
        original_width: int = self.image.shape[1]

        # Compute random scale ratio and crop factors.
        scale_ratio: float = random.uniform(0.5, 1.1)
        scale_ratio = min(scale_ratio, max_size/original_height, max_size/original_width)
        # Perform resizing and flip.
        geom_transformations = augmentations.train_augmentations_geometric_only(
            (int(original_height*scale_ratio), int(original_width*scale_ratio)), additional_targets
        )
        transform_targets = geom_transformations(**transform_targets)

        # Perform random cropping.
        if original_height > 2048 and original_width > 2048:
            # On big images increase the crop range a bit.
            crop_range_start: float = 0.65
            crop_range_stop: float = 1.0
        else:
            crop_range_start: float = 0.75
            crop_range_stop: float = 1.0
        scaled_height: int = transform_targets["image"].shape[0]
        scaled_width: int = transform_targets["image"].shape[1]
        crop_height: int = random.randint(int(crop_range_start * scaled_height),
                                          int(crop_range_stop * scaled_height))
        crop_width: int = random.randint(int(crop_range_start * scaled_width),
                                         int(crop_range_stop * scaled_width))
        crop_transformation = augmentations.random_crop_augmentation(
            (crop_height, crop_width), additional_targets
        )
        transform_targets: dict[str, np.ndarray] = crop_transformation(**transform_targets)

        transformed_image: np.ndarray = transform_targets["image"]
        transformed_instances: list[np.ndarray] = [
            transform_targets[k] for k in additional_targets.keys()
        ]
        transformed_mask: np.ndarray = transform_targets["mask"]
        transformed_manipulated: bool = np.any(transformed_mask > 0.1)

        return HandcraftedForensicsSignalsSample(
            transformed_image,
            transformed_instances,
            transformed_mask,
            transformed_manipulated
        )

    def to_model_input(
        self,
        target_size: tuple[int, int],
        padded: bool = False,
        max_size: int = 1536
    ) -> dict[str, Union[np.ndarray, int, tuple[int, int]]]:
        # Resize image, instance maps and ground-truth mask.
        additional_targets: dict[str, str] = {
            f"instance_{i + 1}": "image" for i in range(len(self.instances))
        }
        transorm_targets: dict[str, np.ndarray] = {"image": self.image}
        if self.mask is not None:
            transorm_targets["mask"] = self.mask
        additional_keys: list[str] = list(additional_targets.keys())
        for k, inp in zip(additional_keys, self.instances):
            transorm_targets[k] = inp
        unpadded_size: tuple[int, int] = target_size
        if padded:
            if self.image.shape[0] > self.image.shape[1]:  # Height > Width
                scale_ratio: float = target_size[0] / self.image.shape[0]
                resized_width: int = int(round(self.image.shape[1] * scale_ratio))
                unpadded_size = (target_size[0], resized_width)
            else:  # Width > Height
                scale_ratio: float = target_size[1] / self.image.shape[1]
                resized_height: int = int(round(self.image.shape[0] * scale_ratio))
                unpadded_size = (resized_height, target_size[1])
        resize_transformation = augmentations.resize_transformation(
            unpadded_size, additional_targets
        )
        resized: dict[str, np.ndarray] = resize_transformation(**transorm_targets)

        instances: Optional[np.ndarray] = None
        if len(self.instances) > 0:
            instances = np.concatenate([resized[k] for k in additional_targets], axis=2)

        # instances: Optional[np.ndarray] = None
        # if len(self.instances) > 0:
        #     instances = np.concatenate(self.instances, axis=2)
        #     additional_targets: dict[str, str] = {"instances": "image"}
        #     transform_targets: dict[str, np.ndarray] = {
        #         "image": self.image,
        #         "mask": self.mask,
        #         "instances": instances
        #     }
        # else:
        #     additional_targets: dict[str, str] = {}
        #     transform_targets: dict[str, np.ndarray] = {
        #         "image": self.image,
        #         "mask": self.mask
        #     }
        # if not padded:
        #     resize_transformation = augmentations.resize_transformation(
        #         target_size, additional_targets
        #     )
        #     resized: dict[str, np.ndarray] = resize_transformation(**transform_targets)
        #     unpadded_size: tuple[int, int] = target_size
        # else:
        #     resize_transformation = augmentations.resize_keep_aspect_ratio_transformation(
        #         max(target_size), additional_targets
        #     )
        #     resized: dict[str, np.ndarray] = resize_transformation(**transform_targets)
        #     unpadded_size: tuple[int, int] = (resized["image"].shape[0], resized["image"].shape[1])
        # if instances is not None:
        #     instances = resized["instances"]

        # Normalize image with ImageNet normalization.
        imagenet_norm_transform = augmentations.imagenet_normalization_transformation({})
        image: np.ndarray = imagenet_norm_transform(image=resized["image"])["image"]

        # Normalize instance maps.
        if instances is not None:
            instances: np.ndarray = instances / 255.0

        # Normalize ground-truth mask.
        mask: Optional[np.ndarray] = None
        if self.mask is not None:
            mask = resized["mask"] / 255.0

        # Constrain original image.
        constrained_original_image: np.ndarray = self.image
        if (constrained_original_image.shape[0] > max_size
                or constrained_original_image.shape[1] > max_size):
            resize_transform = augmentations.resize_keep_aspect_ratio_transformation(max_size, {})
            constrained_original_image = resize_transform(image=self.image)["image"]

        if padded:
            if instances is not None:
                padding_transformation = augmentations.pad_if_needed_transformation(
                    target_size, {"instances": "image"}
                )
                padding_targets = {"image": image, "instances": instances}
                if self.mask is not None:
                    padding_targets["mask"] = mask
                padded_maps = padding_transformation(**padding_targets)
                image = padded_maps["image"]
                if self.mask is not None:
                    mask = padded_maps["mask"]
                instances = padded_maps["instances"]
            else:
                padding_transformation = augmentations.pad_if_needed_transformation(
                    target_size, {}
                )
                padding_targets = {"image": image}
                if self.mask is not None:
                    padding_targets["mask"] = mask
                padded_maps = padding_transformation(**padding_targets)
                image = padded_maps["image"]
                if self.mask is not None:
                    mask = padded_maps["mask"]

        model_input: dict[str, Union[np.ndarray, int, tuple[int, int]]] = {
            "image": image.transpose((2, 0, 1)),
            "instances": instances.transpose((2, 0, 1)) if instances is not None else None,
            "unpadded_size": unpadded_size,
            "original_image": constrained_original_image.transpose((2, 0, 1))
        }
        if self.mask is not None:
            model_input["mask"] = mask.transpose((2, 0, 1))
        if self.manipulated is not None:
            model_input["manipulated"] = self.manipulated

        return model_input


class Split(Enum):
    TRAIN_SPLIT: str = "train"
    EVAL_SPLIT: str = "eval"
    TEST_SPLIT: str = "test"


class ForensicsDataset(Dataset):
    """Dataset for image manipulation detection and localization."""
    def __init__(self,
                 csv_file: pathlib.Path,
                 root_dir: pathlib.Path,
                 signals_columns: list[str],
                 signals_channels: list[int],
                 split: Split,
                 target_image_size: tuple[int, int],
                 mask_column: str = "mask",
                 detection_column: str = "detection",
                 split_column: str = "split",
                 resize_mask: bool = True,
                 transforms_generator: Optional[
                     Callable[[tuple[int, int], dict[str, str]], A.Compose]] = None,
                 imagenet_image_normalization: bool = False,
                 lmdb_file_storage_path: Optional[pathlib.Path] = None,
                 stratify: bool = False):
        """
        :param csv_file:
        :param root_dir:
        :param signals_columns:
        :param signals_channels:
        :param split:
        :param target_image_size:
        :param mask_column:
        :param detection_column:
        :param split_column:
        :param resize_mask: If set to False, mask is not resized into target_image_size.
        :param transforms_generator: Custom augmentations generator to apply, instead of
            the normal train or test ones.
        :param imagenet_image_normalization: When set to True, image is normalized with
            the mean and std of ImageNet. Otherwise, it is normalized with mean 0.5 and
            std 0.5. Defaults to False.
        :param lmdb_file_storage_path: Path to an LMDB File Storage containing the paths
            described in the csv file. If such a path is provided, files are not read
            from the filesystem but from the LMDB File Storage.
        :param stratify: A flag that when provided balances the manipulated and authentic
            samples in the dataset, by copying samples from the smallest distribution.
        """
        super().__init__()

        self.csv_file: pathlib.Path = csv_file
        self.root_dir: pathlib.Path = root_dir
        self.signals_columns: list[str] = signals_columns
        self.signals_channels: list[int] = signals_channels
        self.mask_column: str = mask_column
        self.detection_column: str = detection_column
        self.split_column: str = split_column
        self.split: Split = split
        self.target_image_size: tuple[int, int] = target_image_size  # (H, W)
        self.resize_mask: bool = resize_mask
        self.transforms_generator: Optional[
            Callable[[tuple[int, int], dict[str, str]], A.Compose]] = transforms_generator
        self.imagenet_image_normalization: bool = imagenet_image_normalization
        self.lmdb_file_storage_path: Optional[pathlib.Path] = lmdb_file_storage_path
        if self.lmdb_file_storage_path is not None:
            logger.info(
                f"Loading data from LMDB File Storage: {str(self.lmdb_file_storage_path)} | "
                f"Split: {self.split.value}"
            )

        self.data_specifiers: list[Optional[dict[str, str]]] = []
        for i in range(len(self.signals_columns)):
            self.signals_columns[i], specifier = _parse_data_specifier(self.signals_columns[i])
            self.data_specifiers.append(specifier)

        self.data_reader: Optional[readers.DataReader] = None  # Created on 1st __getitem__().
        self.data: list[dict[str, Any]] = []
        self._load_data(stratify)

    def __getitem__(self, item: int) -> tuple[np.ndarray, np.ndarray, int, int]:
        # Defer the creation of the data reader until the first read operation, in order to
        # properly handle the spawning of multiple processes by DataLoader, where each one
        # should contain a separate reader object.
        if self.data_reader is None:
            self._create_data_reader()

        sample: dict[str, Any] = self.data[item]

        # Load all the requested signals from the paths defined in csv.
        signals: list[np.ndarray] = []
        for i, (s_col, s_channels) in enumerate(zip(self.signals_columns, self.signals_channels)):
            signal_path: str = sample[s_col]
            try:
                if signal_path.endswith(".csv"):
                    # Signals are defined in another csv file (used for
                    # instance segmentation masks).
                    assert s_channels == 0  # Special channels value used for csv files.
                    assert i+1 == len(self.signals_columns)  # Signals from CSV file should be last.
                    signals_batch: list[np.ndarray] = self.data_reader.load_signals_from_csv(
                        signal_path, data_specifier=self.data_specifiers[i]
                    )
                    signals.extend(signals_batch)
                else:
                    # Signals are provided as image files.
                    s: np.ndarray = self.data_reader.load_image(signal_path, channels=s_channels)
                    signals.append(s)
            except Exception as e:
                logger.error(f"Failed to load {signal_path}")
                raise e

        # Load the detection ground-truth.
        assert sample[self.detection_column] in ["True", "False", "TRUE", "FALSE"]
        detection: bool = sample[self.detection_column] in ["True", "TRUE"]

        if detection:
            # Load the ground-truth mask.
            mask: np.ndarray = self.data_reader.load_image(sample[self.mask_column], channels=1)
        else:
            # Generate an empty map (authentic sample).
            if "image" in sample:
                # When the original image is accessible, generate the empty mask to its size.
                width, height = self.data_reader.get_image_size(sample["image"])
                mask_shape: tuple[int, int, int] = (height, width, 1)
            else:
                # When original image is not accessible, use the size of the first signal.
                first_signal_shape: tuple[int, int, int] = signals[0].shape
                mask_shape: tuple[int, int, int] = (first_signal_shape[0], first_signal_shape[1], 1)
            mask: np.ndarray = np.zeros(mask_shape, dtype=np.uint8)

        additional_targets, transforms = self._generate_inputs_transforms(
            len(signals), self.target_image_size
        )
        loaded_sample = ForensicsDatasetSample(signals, mask, detection)
        loaded_sample = loaded_sample.transform(transforms,
                                                additional_targets,
                                                self.resize_mask)
        return (*loaded_sample.to_model_sample(), item)

    def __len__(self) -> int:
        return len(self.data)

    def _create_data_reader(self) -> None:
        # Limit the number of OpenCV threads to 2 to utilize multiple processes. Otherwise,
        # each process spawns a number of threads equal to the number of logical cores and
        # the overall performance gets worse due to threads congestion.
        cv2.setNumThreads(2)

        if self.lmdb_file_storage_path is not None:
            self.data_reader = readers.LMDBFileStorageReader(
                filestorage.LMDBFileStorage(self.lmdb_file_storage_path)
            )
        else:
            self.data_reader = readers.FileSystemReader(self.root_dir)

    def _load_data(self, stratify: bool) -> None:
        self.data = _read_csv_file(self.csv_file)
        # Keep only the data in the requested split.
        self.data = [d for d in self.data if d[self.split_column] == self.split.value]
        logger.info(f"Total image samples: {len(self.data)}  |  Split: {self.split.value}")
        # Keep only the data whose requested columns are non-empty.
        self.data = [d for d in self.data if all([d[col] != "" for col in self.signals_columns])]
        logger.info(f"Valid image samples: {len(self.data)}  |  Split: {self.split.value}")

        if stratify:
            manipulated_samples: list[dict[str, str]] = []
            authentic_samples: list[dict[str, str]] = []
            for d in self.data:
                assert d[self.detection_column] in ["True", "False", "TRUE", "FALSE"]
                if d[self.detection_column] in ["True", "TRUE"]:
                    manipulated_samples.append(d)
                else:
                    authentic_samples.append(d)
            logger.info(f"Dataset: Authentic Samples: {len(authentic_samples)} | "
                        f"Manipulated Samples: {len(manipulated_samples)}")
            if len(authentic_samples) < len(manipulated_samples):
                authentic_samples = random.choices(authentic_samples, k=len(manipulated_samples))
            else:
                manipulated_samples = random.choices(manipulated_samples, k=len(authentic_samples))
            logger.info(f"Stratified: Authentic Samples: {len(authentic_samples)} | "
                        f"Manipulated Samples: {len(manipulated_samples)}")
            manipulated_samples.extend(authentic_samples)
            self.data = manipulated_samples

    def _generate_inputs_transforms(
        self,
        signals_num: int,
        target_image_size: tuple[int, int]
    ) -> tuple[dict[str, str], Union[A.Compose, augmentations.MultiTargetTransformation]]:
        additional_targets: dict[str, str] = {
            f"image_{i + 1}": "image" for i in range(signals_num - 1)
        }
        if self.transforms_generator is not None:
            transforms: A.Compose = self.transforms_generator(target_image_size,
                                                              additional_targets)
        elif self.split == Split.TRAIN_SPLIT:
            if self.imagenet_image_normalization:
                transforms: augmentations.MultiTargetTransformation = \
                    augmentations.train_augmentations_with_image(
                        target_shape=target_image_size, additional_targets=additional_targets
                    )
            else:
                transforms: A.Compose = augmentations.train_augmentations(
                    target_shape=target_image_size, additional_targets=additional_targets
                )
        else:
            if self.imagenet_image_normalization:
                transforms: augmentations.MultiTargetTransformation = \
                    augmentations.test_augmentations_with_image(
                        target_shape=target_image_size, additional_targets=additional_targets
                    )
            else:
                transforms: A.Compose = augmentations.test_augmentations(
                    target_shape=target_image_size, additional_targets=additional_targets
                )
        return additional_targets, transforms


class ForensicsDatasetWithAttentionMask(ForensicsDataset):

    def __init__(self,
                 csv_file: pathlib.Path,
                 root_dir: pathlib.Path,
                 signals_columns: list[str],
                 signals_channels: list[int],
                 split: Split,
                 target_image_size: tuple[int, int],
                 mask_column: str = "mask",
                 detection_column: str = "detection",
                 split_column: str = "split",
                 resize_mask: bool = True,
                 transforms_generator: Optional[
                     Callable[[tuple[int, int], dict[str, str]], A.Compose]] = None,
                 imagenet_image_normalization: bool = False,
                 lmdb_file_storage_path: Optional[pathlib.Path] = None,
                 stratify: bool = False):
        """
        :param csv_file:
        :param root_dir:
        :param signals_columns:
        :param signals_channels:
        :param split:
        :param target_image_size:
        :param mask_column:
        :param detection_column:
        :param split_column:
        :param resize_mask: If set to False, mask is not resized into target_image_size.
        :param transforms_generator: Custom augmentations generator to apply, instead of
            the normal train or test ones.
        :param imagenet_image_normalization: When set to True, image is normalized with
            the mean and std of ImageNet. Otherwise, it is normalized with mean 0.5 and
            std 0.5. Defaults to False.
        :param lmdb_file_storage_path: Path to an LMDB File Storage containing the paths
            described in the csv file. If such a path is provided, files are not read
            from the filesystem but from the LMDB File Storage.
        :param stratify: A flag that when provided balances the manipulated and authentic
            samples in the dataset, by copying samples from the smallest distribution.
        """
        super().__init__(
            csv_file,
            root_dir,
            signals_columns,
            signals_channels,
            split,
            target_image_size,
            mask_column=mask_column,
            detection_column=detection_column,
            split_column=split_column,
            resize_mask=resize_mask,
            transforms_generator=transforms_generator,
            imagenet_image_normalization=imagenet_image_normalization,
            lmdb_file_storage_path=lmdb_file_storage_path,
            stratify=stratify
        )

    def __getitem__(self, item: int) -> tuple[np.ndarray, torch.Tensor, np.ndarray, int, int]:
        packed_signals: np.ndarray
        mask: np.ndarray
        detection: bool
        index: int
        packed_signals, mask, detection, index = super().__getitem__(item)

        # Unpack signals and instance segmentation maps into two different arrays.
        assert self.signals_channels[-1] == 0
        total_signals_channels: int = sum(self.signals_channels)
        instance_masks: Optional[np.ndarray]
        if total_signals_channels == packed_signals.shape[0]:
            # No instance mask included in that sample
            signals: np.ndarray = packed_signals
            instance_masks = None
        else:
            signals: np.ndarray = packed_signals[:total_signals_channels, :, :]
            instance_masks = packed_signals[total_signals_channels:, :, :]

        instance_masks_tensor: Optional[torch.Tensor] = None
        if instance_masks is not None:
            instance_masks_tensor = torch.from_numpy(instance_masks)
        att_mask: torch.Tensor = attention_mask.instances_to_model_ready_attention_mask(
            instance_masks_tensor, self.target_image_size
        )

        return signals, att_mask, mask, detection, index


class HandcraftedForensicsSignalsDataset(Dataset):

    def __init__(self,
                 csv_file: pathlib.Path,
                 root_dir: pathlib.Path,
                 signals_columns: list[str],
                 signals_channels: list[int],
                 split: Split,
                 target_image_size: tuple[int, int],
                 mask_column: str = "mask",
                 detection_column: str = "detection",
                 split_column: str = "split",
                 resize_mask: bool = True,
                 # transforms_generator: Optional[
                 #     Callable[[tuple[int, int], dict[str, str]], A.Compose]] = None,
                 imagenet_image_normalization: bool = True,
                 lmdb_file_storage_path: Optional[pathlib.Path] = None,
                 # custom_signals: tuple[str] = tuple(),
                 # custom_signals_max_size: Optional[tuple[int, int]] = None,
                 image_column: str = "image",
                 stratify: bool = False,
                 keep_aspect_ratio: bool = False):
        """
        :param csv_file:
        :param root_dir:
        :param signals_columns:
        :param signals_channels:
        :param split:
        :param target_image_size:
        :param mask_column:
        :param detection_column:
        :param split_column:
        :param resize_mask: If set to False, mask is not resized into target_image_size.
        :param transforms_generator: Custom augmentations generator to apply, instead of
            the normal train or test ones.
        :param imagenet_image_normalization: When set to True, image is normalized with
            the mean and std of ImageNet. Otherwise, it is normalized with mean 0.5 and
            std 0.5. Defaults to False.
        :param lmdb_file_storage_path: Path to an LMDB File Storage containing the paths
            described in the csv file. If such a path is provided, files are not read
            from the filesystem but from the LMDB File Storage.
        :param custom_signals: Names of custom signals to be loaded.
        :param stratify: A flag that when provided balances the manipulated and authentic
            samples in the dataset, by copying samples from the smallest distribution.
        :param keep_aspect_ratio: A flag that when set to True images are resized by
            retaining their aspect ratio and padded in order to fit into `target_image_size`.
        """
        super().__init__()

        self.csv_file: pathlib.Path = csv_file
        self.root_dir: pathlib.Path = root_dir
        self.signals_columns: list[str] = signals_columns
        self.signals_channels: list[int] = signals_channels
        self.mask_column: str = mask_column
        self.detection_column: str = detection_column
        self.split_column: str = split_column
        self.split: Split = split
        self.target_image_size: tuple[int, int] = target_image_size  # (H, W)
        self.resize_mask: bool = resize_mask
        # self.transforms_generator: Optional[
        #     Callable[[tuple[int, int], dict[str, str]], A.Compose]] = transforms_generator
        self.imagenet_image_normalization: bool = imagenet_image_normalization
        self.lmdb_file_storage_path: Optional[pathlib.Path] = lmdb_file_storage_path
        if self.lmdb_file_storage_path is not None:
            logger.info(
                f"Loading data from LMDB File Storage: {str(self.lmdb_file_storage_path)} | "
                f"Split: {self.split.value}"
            )
        self.image_column: str = image_column
        self.keep_aspect_ratio: bool = keep_aspect_ratio

        assert image_column in self.signals_columns, \
            f"{self.image_column} column should be included into signals_columns"

        assert imagenet_image_normalization, "Only imagenet normalization currently supported."

        self.signals_columns = self.signals_columns.copy()
        self.signals_channels = self.signals_channels.copy()
        self.processor: Optional[processors.OnlinePreprocessor] = None
        if "npp" in self.signals_columns:
            npp_index: int = self.signals_columns.index("npp")
            self.signals_columns.pop(npp_index)
            self.signals_channels.pop(npp_index)
            self.processor = processors.NoiseprintProcessor()

        self.compute_dct: bool = False
        if "dct" in self.signals_columns:
            dct_index: int = self.signals_columns.index("dct")
            self.signals_columns.pop(dct_index)
            self.signals_channels.pop(dct_index)
            self.compute_dct = True
            if self.processor is not None:
                self.processor = processors.CombinedProcessor([
                    self.processor,
                    processors.DCTProcessor()
                ])
            else:
                self.processor = processors.DCTProcessor()

        self.data_specifiers: list[Optional[dict[str, str]]] = []
        for i in range(len(self.signals_columns)):
            self.signals_columns[i], specifier = _parse_data_specifier(self.signals_columns[i])
            self.data_specifiers.append(specifier)

        self.data_reader: Optional[readers.DataReader] = None  # Created on 1st __getitem__().
        self.data: list[dict[str, Any]] = []
        self._load_data(stratify)

    def __getitem__(self, item: int) -> dict[str, Union[np.ndarray, int, tuple[int, int]]]:
        # Defer the creation of the data reader until the first read operation, in order to
        # properly handle the spawning of multiple processes by DataLoader, where each one
        # should contain a separate reader object.
        if self.data_reader is None:
            self._create_data_reader()

        sample: dict[str, Any] = self.data[item]

        load_time_start: float = timeit.default_timer()

        # Load all the requested signals from the paths defined in csv.
        image: Optional[np.ndarray] = None
        instances: list[np.ndarray] = []
        for i, (s_col, s_channels) in enumerate(zip(self.signals_columns, self.signals_channels)):
            signal_path: str = sample[s_col]
            try:
                if signal_path.endswith(".csv"):
                    # Signals are defined in another csv file (used for
                    # instance segmentation masks).
                    assert s_channels == 0  # Special channels value used for csv files.
                    assert i + 1 == len(
                        self.signals_columns)  # Signals from CSV file should be last.
                    signals_batch: list[np.ndarray] = self.data_reader.load_signals_from_csv(
                        signal_path, data_specifier=self.data_specifiers[i]
                    )
                    instances.extend(signals_batch)
                else:
                    # Signals are provided as image files.
                    s: np.ndarray = self.data_reader.load_image(signal_path, channels=s_channels)
                    if s_col == self.image_column:
                        image = s
            except Exception as e:
                logger.error(f"Failed to load {signal_path}")
                raise e

        # Load the detection ground-truth.
        assert sample[self.detection_column] in ["True", "False", "TRUE", "FALSE"]
        detection: bool = sample[self.detection_column] in ["True", "TRUE"]

        if detection:
            # Load the ground-truth mask.
            mask: np.ndarray = self.data_reader.load_image(sample[self.mask_column], channels=1)
        else:
            # Generate an empty map (authentic sample).
            width, height = self.data_reader.get_image_size(sample[self.image_column])
            mask_shape: tuple[int, int, int] = (height, width, 1)
            mask: np.ndarray = np.zeros(mask_shape, dtype=np.uint8)

        load_time_stop: float = timeit.default_timer()

        augment_time_start: float = timeit.default_timer()

        loaded_sample = HandcraftedForensicsSignalsSample(image, instances, mask, detection)
        if self.split == Split.TRAIN_SPLIT:
            loaded_sample = loaded_sample.augment()

        augment_time_stop: float = timeit.default_timer()

        conversion_time_start: float = timeit.default_timer()

        model_input: dict[str, Any] = loaded_sample.to_model_input(self.target_image_size,
                                                                   padded=self.keep_aspect_ratio)

        conversion_time_stop: float = timeit.default_timer()

        attention_mask_time_start: float = timeit.default_timer()

        # Compute attention mask.
        instance_masks_tensor = model_input["instances"]
        if instance_masks_tensor is not None:
            instance_masks_tensor = torch.from_numpy(instance_masks_tensor)
        att_mask: torch.Tensor = attention_mask.instances_to_model_ready_attention_mask(
            instance_masks_tensor, self.target_image_size
        )

        attention_mask_time_stop: float = timeit.default_timer()

        model_input["attention_mask"] = att_mask
        del model_input["instances"]

        # Compute DCT volume.
        if self.compute_dct:
            from .data import dct_loader
            dct_vol, qtable = dct_loader.create_dct_volume_from_augmented(
                self.data_reader.load_file_path_or_stream(sample[self.image_column]),
                model_input["original_image"].transpose((1, 2, 0))
            )
            model_input["dct_vol"] = dct_vol
            model_input["qtable"] = qtable

        # Add dataset index to the returned dict.
        model_input["index"] = item

        model_input["load_time"] = load_time_stop - load_time_start
        model_input["augment_time"] = augment_time_stop - augment_time_start
        model_input["conversion_time"] = conversion_time_stop - conversion_time_start
        model_input["attention_mask_time"] = attention_mask_time_stop - attention_mask_time_start

        return model_input

    def __len__(self) -> int:
        return len(self.data)

    def get_data_processor(self) -> Optional[processors.OnlinePreprocessor]:
        return self.processor

    def build_collate_fn(self) -> Callable[[Iterable[dict[str, Any]]], dict[str, Any]]:
        if self.compute_dct:
            return functools.partial(dict_enlisting_collate_fn,
                                     keys_to_enlist={"original_image", "dct_vol"})
        else:
            return functools.partial(dict_enlisting_collate_fn, keys_to_enlist={"original_image"})

    def _create_data_reader(self) -> None:
        # Limit the number of OpenCV threads to 2 to utilize multiple processes. Otherwise,
        # each process spawns a number of threads equal to the number of logical cores and
        # the overall performance gets worse due to threads congestion.
        cv2.setNumThreads(2)

        if self.lmdb_file_storage_path is not None:
            self.data_reader = readers.LMDBFileStorageReader(
                filestorage.LMDBFileStorage(self.lmdb_file_storage_path)
            )
        else:
            self.data_reader = readers.FileSystemReader(self.root_dir)

    def _load_data(self, stratify: bool) -> None:
        self.data = _read_csv_file(
            self.csv_file,
            columns=[
                self.image_column,
                self.mask_column,
                self.detection_column,
                self.split_column,
                *self.signals_columns
            ]
        )
        # Keep only the data in the requested split.
        self.data = [d for d in self.data if d[self.split_column] == self.split.value]
        logger.info(f"Total image samples: {len(self.data)}  |  Split: {self.split.value}")
        # Keep only the data whose requested columns are non-empty.
        self.data = [d for d in self.data if all([d[col] != "" for col in self.signals_columns])]
        logger.info(f"Valid image samples: {len(self.data)}  |  Split: {self.split.value}")

        if stratify:
            manipulated_samples: list[dict[str, str]] = []
            authentic_samples: list[dict[str, str]] = []
            for d in self.data:
                assert d[self.detection_column] in ["True", "False", "TRUE", "FALSE"]
                if d[self.detection_column] in ["True", "TRUE"]:
                    manipulated_samples.append(d)
                else:
                    authentic_samples.append(d)
            logger.info(f"Dataset: Authentic Samples: {len(authentic_samples)} | "
                        f"Manipulated Samples: {len(manipulated_samples)}")
            if len(authentic_samples) < len(manipulated_samples):
                authentic_samples = random.choices(authentic_samples, k=len(manipulated_samples))
            else:
                manipulated_samples = random.choices(manipulated_samples, k=len(authentic_samples))
            logger.info(f"Stratified: Authentic Samples: {len(authentic_samples)} | "
                        f"Manipulated Samples: {len(manipulated_samples)}")
            manipulated_samples.extend(authentic_samples)
            self.data = manipulated_samples


def dict_enlisting_collate_fn(
    batch: Iterable[dict[str, Any]],
    keys_to_enlist: Iterable[str] = None
) -> dict[str, Any]:
    """Collate function that enlists non-collatable entries of a mini-batch with dicts."""
    # When no keys for non-collatable entries are provided, reside to the default collate_fn.
    if keys_to_enlist is None:
        return torch.utils.data.default_collate(batch)

    keys_to_enlist: set[str] = set(keys_to_enlist)  # Convert to set for amortized-time lookup.

    # Split the dicts in batch into two dicts, containing the collatable and non-collatable keys.
    collatable_batch: list[dict[str, Any]] = []
    non_collatable_batch: list[dict[str, Any]] = []
    for sample in batch:
        collatable_sample: dict[str, Any] = {}
        non_collatable_sample: dict[str, Any] = {}
        for k, v in sample.items():
            if k in keys_to_enlist:
                non_collatable_sample[k] = v
            else:
                collatable_sample[k] = v
        collatable_batch.append(collatable_sample)
        non_collatable_batch.append(non_collatable_sample)

    # The collatable entries are collated normally. The rest are enlisted.
    collated: dict[str, Any] = torch.utils.data.default_collate(collatable_batch)
    for k in keys_to_enlist:
        col: list[Any] = []
        for s in non_collatable_batch:
            if isinstance(s[k], torch.Tensor):
                col.append(torch.utils.data.default_collate([s[k]]).squeeze(dim=0))
            else:
                col.append(torch.utils.data.default_collate(s[k]))
        collated[k] = col
    return collated


def _read_csv_file(
    csv_file: pathlib.Path,
    columns: Optional[Iterable[str]] = None
) -> list[dict[str, Any]]:

    with csv_file.open("r") as f:
        reader = csv.DictReader(f, delimiter=",")

        if columns is None:
            contents: list[dict[str, Any]] = [row for row in reader]
        else:
            contents: list[dict[str, Any]] = []
            for row in reader:
                contents.append({k: row[k] for k in columns})

    return contents


def _parse_data_specifier(column_name: str) -> tuple[str, Optional[dict[str, str]]]:
    specifier: Optional[dict[str, str]] = None
    clean_column_name: str = column_name
    if "(" in column_name and ")" in column_name:
        clean_column_name = column_name.split("(")[0]
        specifier_text: str = column_name.split("(")[1].split(")")[0]
        specifier_key, specifier_value = specifier_text.split(":")
        specifier = {specifier_key: specifier_value}
    return clean_column_name, specifier
