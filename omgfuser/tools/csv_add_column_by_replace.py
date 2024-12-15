"""Adds a new column to a CSV file by copying an existing one and replacing a string occurrence.

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

import click
from tqdm import tqdm

from omgfuser import utils
from omgfuser import datasets


@click.command()
@click.option("-c", "--csv_path", required=True,
              type=click.Path(dir_okay=False, exists=True, path_type=pathlib.Path))
@click.option("-s", "--src_column", type=str, required=True)
@click.option("-t", "--tgt_column", type=str, required=True)
@click.option("-k", "--src_str", type=str, required=True)
@click.option("-l", "--tgt_str", type=str, required=True)
@click.option("-o", "--outfile", type=str, required=True)
def cli(
    csv_path: pathlib.Path,
    src_column: str,
    tgt_column: str,
    src_str: str,
    tgt_str: str,
    outfile: str
) -> None:
    entries: list[dict[str, str]] = datasets._read_csv_file(csv_path)

    for e in tqdm(entries, desc=f"Adding {tgt_column} column", unit="entry"):
        e[tgt_column] = e[src_column].replace(src_str, tgt_str)

    utils.write_csv_file(entries, pathlib.Path(outfile))


if __name__ == "__main__":
    cli()
