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

import click
from tqdm import tqdm

from omgfuser import utils
from omgfuser import datasets


@click.command()
@click.option("-a", "--csv_a", required=True,
              type=click.Path(dir_okay=False, exists=True, path_type=pathlib.Path))
@click.option("-b", "--csv_b", required=True,
              type=click.Path(dir_okay=False, exists=True, path_type=pathlib.Path))
@click.option("-o", "--outfile", type=str, required=True)
def cli(
    csv_a: pathlib.Path,
    csv_b: pathlib.Path,
    outfile: str,
) -> None:
    csv_a_entries: list[dict[str, str]] = datasets._read_csv_file(csv_a)
    csv_b_entries: list[dict[str, str]] = datasets._read_csv_file(csv_b)

    csv_a_entries.extend(csv_b_entries)
    utils.write_csv_file(csv_a_entries, pathlib.Path(outfile))


if __name__ == "__main__":
    cli()
