"""Script for generating a CSV file for the JPEG-RAISE dataset.

The generated CSV includes the following columns:
    - image: Path to the image samples of the dataset.
    - mask: Empty column, since JPEG-RAISE does not contain any manipulated samples.
    - detection: A column filled with FALSE, since JPEG-RAISE does not contain any manipulated
        samples.

Version: 1.0

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
from typing import Any, Optional

import click
import tqdm


AUTHENTIC_PATH: str = "compRAISE"  # Path of the authentic samples.

logging.getLogger().setLevel(logging.INFO)


@click.command()
@click.option("-d", "--dataset_dir", required=True,
              type=click.Path(file_okay=False, exists=True, path_type=pathlib.Path),
              help="Base directory of the JPEG-RAISE dataset.")
@click.option("-o", "--output_dir",
              type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path))
@click.option("--output_filename", type=str, default="jpeg_raise.csv")
def cli(
    dataset_dir: pathlib.Path,
    output_dir: Optional[pathlib.Path],
    output_filename: str
) -> None:
    authentic_path: pathlib.Path = pathlib.Path(dataset_dir / AUTHENTIC_PATH)

    entries: list[dict[str, str]] = []

    for p in tqdm.tqdm(authentic_path.iterdir(), "Finding authentic samples", unit="sample"):
        e: dict[str, str] = {
            "image": str(p.relative_to(dataset_dir)),
            "mask": "",
            "detection": "FALSE"
        }
        entries.append(e)

    if output_dir is None:
        output_dir = dataset_dir
    csv_path: pathlib.Path = output_dir / output_filename
    write_csv_file(entries, csv_path)
    logging.info(f"Exported CSV at: {str(csv_path)}")


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
