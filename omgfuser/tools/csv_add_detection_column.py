"""Script for adding precomputed tampering detection results as a column in a dataset CSV file.

The precomputed detection results should be included into another CSV file that contains
at least the `image` column and a column with the detection score per image ranging in [0, 1].

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

import logging
import pathlib
from typing import Optional, Any

import click

from omgfuser.data.filestorage import read_csv_file
from omgfuser import utils


__version__: str = "1.1.0"
__revision__: int = 2
__author__: str = "Dimitrios Karageorgiou"
__email__: str = "dkarageo@iti.gr"

logging.getLogger().setLevel(logging.INFO)


@click.command()
@click.option("--dataset_csv",
              type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
              required=True,
              show_default=True)
@click.option("--detection_csv",
              type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
              required=True,
              show_default=True)
@click.option("--detection_column", type=str, required=True)
@click.option("--image_column", type=str, default="image", show_default=True)
@click.option("--detection_groundtruth", type=str,
              help="Limits the addition of precomputed results only to the samples whose "
                   "`detection` column matches the given value. Providing that option "
                   "is necessary when the precomputed results for the manipulated and the "
                   "authentic samples are included into different CSV files. When that option "
                   "is absent, the manipulation detection scores for every sample in "
                   "`dataset_csv` should be included in `detection_csv`.")
@click.option("--output_filename", type=str, required=True)
@click.option("--ignore_nonexisting", is_flag=True,
              help="When this flag is provided, images in `dataset_csv` without a corresponding "
                   "entry in `detection_csv` do not cause an error. Instead, their detection "
                   "column is left empty.")
@click.option("--simplify_paths", is_flag=True,
              help="When this flag is provided, image paths in `dataset_csv` are converted in "
                   "simple paths. A simple path is one that has flattened the directory "
                   "hierarchy by integrating directory levels into filename. Ultimately, "
                   "the path separators are converted into underscores.")
def cli(
    dataset_csv: pathlib.Path,
    detection_csv: pathlib.Path,
    detection_column: str,
    image_column: str,
    detection_groundtruth: Optional[str],
    output_filename: str,
    ignore_nonexisting: bool,
    simplify_paths: bool
) -> None:
    dataset_entries: list[dict[str, Any]] = read_csv_file(dataset_csv)
    logging.info(f"DATASET ENTRIES: {len(dataset_entries)}")

    detection_entries: list[dict[str, str]] = read_csv_file(detection_csv)
    detection_per_image: dict[str, float] = {
        e[image_column]: float(e[detection_column]) for e in detection_entries
    }

    if detection_groundtruth is not None:
        logging.info(
            f"ADDING DETECTION RESULTS TO SAMPLES WITH GROUND-TRUTH: {detection_groundtruth}"
        )

    detection_entries_added: int = 0
    for e in dataset_entries:
        previous_detection_value: str = e.get(detection_column, "").strip()
        if (detection_groundtruth is not None) and (e["detection"] != detection_groundtruth):
            if previous_detection_value == "":  # Initialize only empty cells.
                e[detection_column] = ""
        else:
            if simplify_paths:
                path_parts: tuple[str] = pathlib.Path(e[image_column]).parts
                image_name: str = "_".join(path_parts)
            else:
                image_name: str = pathlib.Path(e[image_column]).name

            if ignore_nonexisting:
                if image_name in detection_per_image:
                    e[detection_column] = detection_per_image[image_name]
                elif previous_detection_value == "":
                    e[detection_column] = ""
            else:
                e[detection_column] = detection_per_image[image_name]

            if previous_detection_value == "" and e[detection_column] != "":
                detection_entries_added += 1

    logging.info(f"ENTRIES AUGMENTED WITH DETECTION RESULTS: {detection_entries_added}")

    output_path: pathlib.Path = dataset_csv.parent / output_filename
    utils.write_csv_file(dataset_entries, output_path)


if __name__ == "__main__":
    cli()
