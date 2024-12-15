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
from typing import List, Dict, Any, Optional

import tqdm
from PIL import Image, ImageOps

DATASET_ROOT = "/fssd3/user-data/dkarageo"
CSV_PATH = "/fssd3/user-data/dkarageo/nist16_test.csv"


def cli():
    csv_path: pathlib.Path = pathlib.Path(CSV_PATH)
    entries: list[dict[str, str]] = read_csv_file(csv_path)

    for e in tqdm.tqdm(entries, "Creating inverted masks", unit="sample"):
        if e["mask"] != "":
            mask_path: pathlib.Path = pathlib.Path(DATASET_ROOT) / e["mask"]
            dir_name: str = f"{mask_path.parent.name}_inverted"
            fixed_mask_dir: pathlib.Path = mask_path.parent.parent / dir_name
            fixed_mask_dir.mkdir(exist_ok=True, parents=True)
            fixed_mask_path: pathlib.Path = fixed_mask_dir / mask_path.name
            create_fixed_mask(mask_path, fixed_mask_path)
            e["mask"] = str(fixed_mask_path.relative_to(pathlib.Path(DATASET_ROOT)))

    write_csv_file(entries, csv_path)


def read_csv_file(csv_file: pathlib.Path) -> List[Dict[str, str]]:
    # Read the whole csv file.
    logging.info(f"READING CSV: {str(csv_file)}")
    entries: List[Dict[str, str]] = []
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


def create_fixed_mask(mask_path: pathlib.Path,
                      output_path: pathlib.Path) -> None:
    with Image.open(mask_path) as mask:
        # Mask should be an 8-bit single-channel image.
        if mask.mode != "L":
            mask = mask.convert("L")

        # Invert the mask. Input images are expected to contain high values for the manipulated
        # regions and small values for the pristine ones.
        mask = ImageOps.invert(mask)

        mask.save(output_path)


if __name__ == "__main__":
    cli()
