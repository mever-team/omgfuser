"""Tool for generating an augmented version of a given dataset.

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

import pathlib
from typing import Optional

import click
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from omgfuser import utils
from omgfuser.data.filestorage import read_csv_file


__version__: str = "1.0.0"
__revision__: int = 1
__author__: str = "Dimitrios S. Karageorgiou"
__email__: str = "dkarageo@iti.gr"


AUGMENTATIONS: list[str] = [
    "jpeg-compression",
    "rescaling",
    "gauss-noise",
    "gauss-blur"
]


@click.command()
@click.option("-c", "--dataset_csv",
              type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
              required=True,
              show_default=True)
@click.option("--dataset_root",
              type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path))
@click.option("-a", "--augmentation",
              type=click.Choice(AUGMENTATIONS, case_sensitive=False),
              required=True)
@click.option("-v", "--value", type=float, required=True)
@click.option("--image_column", type=str, default="image")
@click.option("--mask_column", type=str, default="mask")
@click.option("--copy_columns", type=str, default="detection,split")
@click.option("-o", "--output_dir",
              type=click.Path(file_okay=False, path_type=pathlib.Path),
              required=True,
              show_default=True)
def augment(
    dataset_csv: pathlib.Path,
    dataset_root: Optional[pathlib.Path],
    augmentation: str,
    value: float,
    image_column: str,
    mask_column: str,
    copy_columns: str,
    output_dir: pathlib.Path,
) -> None:
    csv_entries: list[dict[str, str]] = read_csv_file(dataset_csv)

    if not dataset_root:
        dataset_root = dataset_csv.absolute().parent

    augmented_csv_entries: list[dict[str, str]] = []

    txt_value: str = str(int(value) if value > 1 else value)

    for e in tqdm(csv_entries, desc="Augmenting dataset", unit="image"):
        rel_img_path: pathlib.Path = pathlib.Path(e[image_column])
        img_path: pathlib.Path = dataset_root / rel_img_path

        augmented_top_dir: str = f"{rel_img_path.parts[0]}_{augmentation}_{txt_value}"
        out_dir: pathlib.Path = output_dir.joinpath(
            augmented_top_dir, *rel_img_path.parent.parts[1:]
        )
        out_dir.mkdir(exist_ok=True, parents=True)

        out_path: pathlib.Path = augment_image(
            img_path, augmentation, value, out_dir, img_path.stem
        )

        # Create the sample entry into the CSV file of the augmented dataset.
        augmented_entry: dict[str, str] = {
            image_column: out_path.relative_to(output_dir),
            mask_column: e[mask_column],
        }
        for c in copy_columns.split(","):
            augmented_entry[c] = e[c]
        augmented_csv_entries.append(augmented_entry)

        augmented_csv_path: pathlib.Path = (
                output_dir /
                f"{dataset_csv.stem}_{augmentation}_{txt_value}.csv"
        )
        utils.write_csv_file(augmented_csv_entries, augmented_csv_path)


def augment_image(
    img_path: pathlib.Path,
    augmentation: str,
    value: float,
    out_dir: pathlib.Path,
    out_name: str
) -> pathlib.Path:

    out_filename: str = f"{out_name}.jpg"
    out_path: pathlib.Path = out_dir / out_filename

    if augmentation == "jpeg-compression":
        img: np.ndarray = load_image(img_path, 3)
        quality: int = int(value)
        Image.fromarray(img).save(out_path, quality=quality)
    elif augmentation == "rescaling":
        with Image.open(img_path) as imgp:
            imgp = imgp.resize((int(imgp.width*value), int(imgp.height*value)))
            imgp.save(out_path, quality=100)
    elif augmentation == "gauss-noise":
        img: np.ndarray = load_image(img_path, 3).astype(np.float32)
        gauss_noise: np.ndarray = np.clip(
            np.random.normal(0, int(value), img.shape),
            0, 255
        )
        img += gauss_noise
        img = np.clip(img, 0, 255)
        Image.fromarray(img.astype(np.uint8)).save(out_path, quality=100)
    elif augmentation == "gauss-blur":
        img: np.ndarray = load_image(img_path, 3).astype(np.float32)
        r: int = int((value-1)/2)
        sigma: float = r / 4.0
        img = gaussian_filter(img, sigma=sigma)
        Image.fromarray(img.astype(np.uint8)).save(out_path, quality=100)
    else:
        raise RuntimeError(f"Unsupported augmentation: {augmentation}")

    return out_path


def load_image(path: pathlib.Path, channels: int) -> np.ndarray:
    with Image.open(path) as image:
        if channels == 1:
            image = image.convert("L")
        else:
            image = image.convert("RGB")
        image = np.array(image)

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    return image


if __name__ == "__main__":
    augment()
