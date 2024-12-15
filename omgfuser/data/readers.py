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
import io
import pathlib
from typing import Any, Union, Optional

import numpy as np
from PIL import Image

from . import filestorage


class DataReader:

    def read_csv_file(self, path: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    def load_image(self, path: str, channels: int) -> np.ndarray:
        raise NotImplementedError

    def get_image_size(self, path: str) -> tuple[int, int]:
        """Returns the size of an image as a (width, height) tuple."""
        raise NotImplementedError

    def load_signals_from_csv(
        self,
        csv_path: str,
        column_name: str = "seg_map",
        channels: int = 1,
        data_specifier: Optional[dict[str, str]] = None
    ) -> list[np.ndarray]:
        """Loads all the signals specified in a column of a CSV file.

        The default values for the column name and the number of channels have
        been specified for the CSV containing the instance segmentation maps of
        an image."""
        raise NotImplementedError

    def load_file_path_or_stream(self, path: str) -> Union[pathlib.Path, io.FileIO, io.BytesIO]:
        raise NotImplementedError


class FileSystemReader(DataReader):
    """Reader that maps relative paths to absolute paths of the filesystem."""
    def __init__(self, root_path: pathlib.Path):
        super().__init__()
        self.root_path: pathlib.Path = root_path

    def read_csv_file(self, path: str) -> list[dict[str, Any]]:
        with (self.root_path/path).open("r") as f:
            reader = csv.DictReader(f, delimiter=",")
            contents: list[dict[str, Any]] = [row for row in reader]
        return contents

    def get_image_size(self, path: str) -> tuple[int, int]:
        with Image.open(self.root_path/path) as image:
            image_size: tuple[int, int] = image.size
        return image_size

    def load_image(self, path: str, channels: int) -> np.ndarray:
        with Image.open(self.root_path/path) as image:
            if channels == 1:
                image = image.convert("L")
            else:
                image = image.convert("RGB")
            image = np.array(image)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        return image

    def load_signals_from_csv(
        self,
        csv_path: str,
        column_name: str = "seg_map",
        channels: int = 1,
        data_specifier: Optional[dict[str, str]] = None
    ) -> list[np.ndarray]:
        csv_data: list[dict[str, Any]] = self.read_csv_file(csv_path)

        signals: list[np.ndarray] = []
        for row in csv_data:
            # Ignore entries that do not match with the given data specifier.
            if data_specifier is not None and not data_specifier_matches_entry(row, data_specifier):
                continue

            signal_path: pathlib.Path = (self.root_path / csv_path).parent / row[column_name]
            signal: np.ndarray = self.load_image(str(signal_path.relative_to(self.root_path)),
                                                 channels=channels)
            signals.append(signal)

        return signals

    def load_file_path_or_stream(self, path: str) -> Union[pathlib.Path, io.FileIO, io.BytesIO]:
        return self.root_path / path


class LMDBFileStorageReader(DataReader):
    """Reader that maps relative paths into an LMDBFileStorage."""
    def __init__(self, storage: filestorage.LMDBFileStorage):
        super().__init__()
        self.storage: filestorage.LMDBFileStorage = storage

    def read_csv_file(self, path: str) -> list[dict[str, Any]]:
        stream = self.storage.open_file(path)
        reader = csv.DictReader(stream, delimiter=",")
        contents: list[dict[str, Any]] = [row for row in reader]
        return contents

    def get_image_size(self, path: str) -> tuple[int, int]:
        stream = self.storage.open_file(path, mode="b")
        with Image.open(stream) as image:
            image_size: tuple[int, int] = image.size
        return image_size

    def load_image(self, path: str, channels: int) -> np.ndarray:
        stream = self.storage.open_file(path, mode="b")
        with Image.open(stream) as image:
            if channels == 1:
                image = image.convert("L")
            else:
                image = image.convert("RGB")
            image = np.array(image)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        return image

    def load_signals_from_csv(
        self,
        csv_path: str,
        column_name: str = "seg_map",
        channels: int = 1,
        data_specifier: Optional[dict[str, str]] = None
    ) -> list[np.ndarray]:
        csv_data: list[dict[str, Any]] = self.read_csv_file(csv_path)

        signals: list[np.ndarray] = []
        for row in csv_data:
            # Ignore entries that do not match with the given data specifier.
            if data_specifier is not None and not data_specifier_matches_entry(row, data_specifier):
                continue

            signal_path: str = str(pathlib.Path(csv_path).parent / row[column_name])
            signal: np.ndarray = self.load_image(signal_path, channels=channels)
            signals.append(signal)

        return signals

    def load_file_path_or_stream(self, path: str) -> Union[pathlib.Path, io.FileIO, io.BytesIO]:
        return self.storage.open_file(path, mode="b")


def data_specifier_matches_entry(entry: dict[str, str], specifier: dict[str, str]) -> bool:
    """Checks whether a CSV entry matches a data specifier."""
    for k, v in specifier.items():
        if k not in entry or entry[k] != v:
            return False
    return True
