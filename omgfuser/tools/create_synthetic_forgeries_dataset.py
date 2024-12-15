"""Script for generating a new dataset out of five synthetic datasets including image forgeries.

The datasets that are utilized for generating the final dataset are the following ones:
    - SP-COCO (199.999 spliced images from COCO)
    - CM-COCO (199.424 copy-move images from COCO)
    - CM-RAISE (199.443 copy-move images from RAISE by utilizing instance masks from COCO)
    - CM-JPEG-RAISE (199.443 copy-move image, like CM-RAISE,
      compressed with many quantization tables)
    - JPEG-RAISE (25.462 authentic images from RAISE)

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
import logging
import pathlib
import shutil
import random
from typing import Optional, Any, List, Dict

import click
from tqdm import tqdm


__version__: str = "1.0.0"
__author__: str = "Dimitris Karageorgiou"
__email__: str = "dkarageo@iti.gr"


@click.command()
@click.option("--spcoco_path",
              type=click.Path(file_okay=False, exists=True, path_type=pathlib.Path),
              default="./SP-COCO",
              show_default=True)
@click.option("--cmcoco_path",
              type=click.Path(file_okay=False, exists=True, path_type=pathlib.Path),
              default="./CM-COCO",
              show_default=True)
@click.option("--cmraise_path",
              type=click.Path(file_okay=False, exists=True, path_type=pathlib.Path),
              default="./CM-RAISE",
              show_default=True)
@click.option("--cmjpegraise_path",
              type=click.Path(file_okay=False, exists=True, path_type=pathlib.Path),
              default="./CM-JPEG-RAISE",
              show_default=True)
@click.option("--jpegraise_path",
              type=click.Path(file_okay=False, exists=True, path_type=pathlib.Path),
              default="./JPEG-RAISE",
              show_default=True)
@click.option("--spcoco_num", type=int, default=10000, show_default=True)
@click.option("--cmcoco_num", type=int, default=5000, show_default=True)
@click.option("--cmraise_num", type=int, default=5000, show_default=True)
@click.option("--cmjpegraise_num", type=int, default=5000, show_default=True)
@click.option("--jpegraise_num", type=int, default=None, show_default=True)
@click.option("--output_dir",
              type=click.Path(file_okay=False, path_type=pathlib.Path))
def cli(
    spcoco_path: pathlib.Path,
    cmcoco_path: pathlib.Path,
    cmraise_path: pathlib.Path,
    cmjpegraise_path: pathlib.Path,
    jpegraise_path: pathlib.Path,
    spcoco_num: Optional[int],
    cmcoco_num: Optional[int],
    cmraise_num: Optional[int],
    cmjpegraise_num: Optional[int],
    jpegraise_num: Optional[int],
    output_dir: pathlib.Path
) -> None:
    manipulated_images_out_dir: pathlib.Path = output_dir / "synthetic_dataset" / "manipulated"
    masks_out_dir: pathlib.Path = output_dir / "synthetic_dataset" / "masks"
    authentic_images_out_dir: pathlib.Path = output_dir / "synthetic_dataset" / "authentic"

    manipulated_images_out_dir.mkdir(exist_ok=True, parents=True)
    masks_out_dir.mkdir(exist_ok=True, parents=True)
    authentic_images_out_dir.mkdir(exist_ok=True, parents=True)

    sp_coco_samples: list[dict[str, str]] = copy_random_samples(
        spcoco_path,
        spcoco_path / "sp_COCO_list.csv",
        manipulated_images_out_dir,
        masks_out_dir,
        output_dir,
        spcoco_num,
    )
    add_dataset_info(sp_coco_samples, True, "SP-COCO")

    cm_coco_samples: list[dict[str, str]] = copy_random_samples(
        cmcoco_path,
        cmcoco_path / "cm_COCO_list.csv",
        manipulated_images_out_dir,
        masks_out_dir,
        output_dir,
        cmcoco_num
    )
    add_dataset_info(cm_coco_samples, True, "CM-COCO")

    cm_raise_samples: list[dict[str, str]] = copy_random_samples(
        cmraise_path,
        cmraise_path / "bcm_COCO_list.csv",
        manipulated_images_out_dir,
        masks_out_dir,
        output_dir,
        cmraise_num
    )
    add_dataset_info(cm_raise_samples, True, "CM-RAISE")

    cm_jpeg_raise_samples: list[dict[str, str]] = copy_random_samples(
        cmjpegraise_path,
        cmjpegraise_path / "bcmc_COCO_list.csv",
        manipulated_images_out_dir,
        masks_out_dir,
        output_dir,
        cmjpegraise_num
    )
    add_dataset_info(cm_jpeg_raise_samples, True, "CM-JPEG-RAISE")

    jpeg_raise_samples: list[dict[str, str]] = copy_random_samples(
        jpegraise_path,
        jpegraise_path / "jpeg_raise.csv",
        authentic_images_out_dir,
        None,
        output_dir,
        jpegraise_num
    )
    add_dataset_info(jpeg_raise_samples, False, "JPEG-RAISE")

    total_csv_data: list[dict[str, str]] = []
    total_csv_data.extend(sp_coco_samples)
    total_csv_data.extend(cm_coco_samples)
    total_csv_data.extend(cm_raise_samples)
    total_csv_data.extend(cm_jpeg_raise_samples)
    total_csv_data.extend(jpeg_raise_samples)

    write_csv_file(total_csv_data,
                   output_dir/"synthetic_dataset_samples.csv",
                   fieldnames=["image", "mask", "detection", "dataset",
                               "original_image", "original_mask"])


def copy_random_samples(
    source_images_base: pathlib.Path,
    source_images_csv: pathlib.Path,
    target_images_dir: pathlib.Path,
    target_masks_dir: Optional[pathlib.Path],
    base_dir: pathlib.Path,
    num: Optional[int]
) -> list[dict[str, str]]:

    samples: list[dict[str, str]] = read_csv_file(source_images_csv)
    if num is not None:
        samples = random.sample(samples, num)

    samples_data: list[dict[str, str]] = []

    for s in tqdm(samples, "Copying samples", unit="sample"):
        # Copy the image from the source to target dir.
        src_img: pathlib.Path = source_images_base / s["image"]
        tgt_img: pathlib.Path = target_images_dir / src_img.name
        shutil.copy(src_img, tgt_img)

        data: dict[str, str] = {
            "image": str(tgt_img.relative_to(base_dir)),
            "original_image": str(src_img.relative_to(base_dir)),
            "mask": "",
            "original_mask": ""
        }

        if target_masks_dir is not None:
            src_mask: pathlib.Path = source_images_base / s["mask"]
            tgt_mask: pathlib.Path = target_masks_dir / src_mask.name
            shutil.copy(src_mask, tgt_mask)

            data["mask"] = str(tgt_mask.relative_to(base_dir))
            data["original_mask"] = str(src_mask.relative_to(base_dir))

        samples_data.append(data)

    return samples_data


def add_dataset_info(
    samples: list[dict[str, str]],
    manipulated: bool,
    name: str
) -> None:
    samples = samples.copy()
    for s in samples:
        s["dataset"] = name
        s["detection"] = "TRUE" if manipulated else "FALSE"


def read_csv_file(csv_file: pathlib.Path) -> List[Dict[str, str]]:
    # Read the whole csv file.
    logging.info(f"READING CSV: {str(csv_file)}")
    entries: List[Dict[str, str]] = []
    with csv_file.open() as f:
        reader: csv.DictReader = csv.DictReader(f, delimiter=",")
        for row in tqdm(reader, desc="Reading CSV entries", unit="entry"):
            entries.append(row)
    logging.info(f"TOTAL ENTRIES: {len(entries)}")
    return entries


def write_csv_file(
    data: list[dict[str, Any]],
    output_file: pathlib.Path,
    fieldnames: Optional[list[str]] = None
) -> None:
    with output_file.open("w") as f:
        if not fieldnames:
            fieldnames = data[0].keys()
        writer: csv.DictWriter = csv.DictWriter(f,
                                                fieldnames=fieldnames,
                                                delimiter=",",
                                                extrasaction="ignore")
        writer.writeheader()
        for r in data:
            writer.writerow(r)


if __name__ == "__main__":
    cli()
