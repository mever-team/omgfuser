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
import logging
import pathlib
import shutil
from typing import List, Any, Optional

import click
import tqdm


@click.command()
@click.option("-i", "--base_dir", required=True,
              type=click.Path(file_okay=False, exists=True, path_type=pathlib.Path))
@click.option("-c", "--dataset_csv", required=True,
              type=click.Path(dir_okay=False, exists=True, path_type=pathlib.Path))
@click.option("-o", "--output_dir", required=True,
              type=click.Path(file_okay=False, path_type=pathlib.Path))
def cli(
    base_dir: pathlib.Path,
    dataset_csv: pathlib.Path,
    output_dir: pathlib.Path
) -> None:
    entries: list[dict[str, str]] = read_csv_file(dataset_csv)
    out_data: list[dict[str, str]] = []

    output_dir.mkdir(exist_ok=True, parents=True)
    for e in entries:
        relative_path: pathlib.Path = pathlib.Path(e["image"])
        src_path: pathlib.Path = base_dir / relative_path
        path_parts: list[str] = str(relative_path.parent).split("/")
        name_prefix: str = "_".join(path_parts)
        target_path: pathlib.Path = output_dir / f"{name_prefix}_{relative_path.name}"
        shutil.copyfile(src_path, target_path)
        out_data.append({
            "image": str(relative_path),
            "mask": "",
            "image_simple": str(target_path.relative_to(output_dir)),
            "mask_simple": ""
        })

    write_csv_file(out_data, output_file=output_dir/"dataset_files.csv")


def read_csv_file(csv_file: pathlib.Path) -> list[dict[str, str]]:
    # Read the whole csv file.
    logging.info(f"READING CSV: {str(csv_file)}")
    entries: List[dict[str, str]] = []
    with csv_file.open() as f:
        reader: csv.DictReader = csv.DictReader(f, delimiter=",")
        for row in tqdm.tqdm(reader, desc="Reading CSV entries", unit="entry"):
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
