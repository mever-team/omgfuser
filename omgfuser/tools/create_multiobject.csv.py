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

from pathlib import Path
from typing import Any

import click
import numpy as np
from PIL import Image
from skimage import measure
from tqdm import tqdm

from omgfuser.data.filestorage import read_csv_file
from omgfuser.utils import write_csv_file


@click.command()
@click.option("-d", "--dataset_csv", required=True,
              type=click.Path(dir_okay=False, exists=True, path_type=Path),
              help="CSV of the dataset.")
@click.option("-o", "--output_dir", required=True,
              type=click.Path(file_okay=False, path_type=Path),
              help="Directory where the filtered csv will be written")
def cli(
    dataset_csv: Path,
    output_dir: Path
) -> None:
    entries: list[dict[str, Any]] = read_csv_file(dataset_csv)
    base_dir: Path = dataset_csv.parent
    multiobject_entries: list[dict[str, Any]] = []
    for e in tqdm(entries):
        mask_file: str = e["mask"]
        if not mask_file:
            continue
        mask_path: Path = base_dir / mask_file
        with Image.open(mask_path) as mask:
            mask = np.asarray(mask)
            mask = mask / 255
            segmented = measure.label(mask)
            regions = measure.regionprops(segmented)
            if len(regions) > 2:
                multiobject_entries.append(e)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_csv: Path = output_dir / f"{dataset_csv.stem}_many_objects.csv"
    write_csv_file(multiobject_entries, output_csv)


if __name__ == "__main__":
    cli()
