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

import pathlib
import shutil
from typing import Optional

import click

from omgfuser import utils
from omgfuser.data.filestorage import read_csv_file


@click.command()
@click.option("-c", "--dataset_csv",
              type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
              required=True,
              show_default=True)
@click.option("--dataset_root",
              type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path))
@click.option("-o", "--output_dir",
              type=click.Path(file_okay=False, path_type=pathlib.Path),
              required=True,
              show_default=True)
@click.option("--image_column", type=str, default="image")
@click.option("--simple_image_column", type=str, default="image_simple")
@click.option("--rule", type=str)
def simplify(
    dataset_csv: pathlib.Path,
    dataset_root: Optional[pathlib.Path],
    output_dir: pathlib.Path,
    image_column: str,
    simple_image_column: str,
    rule: Optional[str]
):
    if not dataset_root:
        dataset_root = dataset_csv.absolute().parent

    csv_entries: list[dict[str, str]] = read_csv_file(dataset_csv)

    output_dir.mkdir(exist_ok=True, parents=True)

    for e in csv_entries:
        if rule is not None and rule not in set(pathlib.Path(e[image_column]).parts):
            continue

        src_path: pathlib.Path = dataset_root / e[image_column]
        filename: str = "_".join([p for p in pathlib.Path(e[image_column]).parts])
        target_dir: pathlib.Path = output_dir / f"{pathlib.Path(e[image_column]).parts[0]}_simple"
        target_dir.mkdir(exist_ok=True, parents=True)
        target_path: pathlib.Path = target_dir / filename
        shutil.copy(src_path, target_path)

        e[simple_image_column] = str(target_path.relative_to(output_dir))

    simple_dataset_csv: pathlib.Path = output_dir / f"{dataset_csv.stem}_simple.csv"
    utils.write_csv_file(csv_entries, simple_dataset_csv)


if __name__ == "__main__":
    simplify()
