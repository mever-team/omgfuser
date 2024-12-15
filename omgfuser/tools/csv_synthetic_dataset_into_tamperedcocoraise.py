"""Script intended to generate a csv from the synthetic dataset into the TamperedCOCORAISE.

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
import logging

import click
from tqdm import tqdm

from omgfuser import utils
from omgfuser import datasets


logging.getLogger().setLevel(logging.INFO)


@click.command()
@click.option("-c", "--data_csv", required=True,
              type=click.Path(dir_okay=False, exists=True, path_type=pathlib.Path))
@click.option("-g", "--guide_csv", required=True,
              type=click.Path(dir_okay=False, exists=True, path_type=pathlib.Path))
@click.option("-m", "--guide_column", type=str, default="image")
@click.option("-o", "--outfile", type=str, required=True)
def cli(
    data_csv: pathlib.Path,
    guide_csv: pathlib.Path,
    guide_column: str,
    outfile: str
) -> None:
    data_entries: list[dict[str, str]] = datasets._read_csv_file(data_csv)
    guide_entries: list[dict[str, str]] = datasets._read_csv_file(guide_csv)

    logging.info(f"Entries in data csv: {len(data_entries)}")
    logging.info(f"Entries in guidance csv: {len(guide_entries)}")

    data_entries_by_filename: dict[str, dict[str, str]] = {
        pathlib.Path(e[guide_column]).name: e for e in data_entries
    }

    selected_entries: list[dict[str, str]] = []
    error_keys: list[str] = []
    for e in tqdm(guide_entries, desc=f"Finding matching samples", unit="entry"):
        k: str = pathlib.Path(e[guide_column]).name
        try:
            selected_entries.append(data_entries_by_filename[k])
        except KeyError:
            error_keys.append(k)

    for k in error_keys:
        logging.error(f"Failed to find: {k}")

    logging.info(f"Entries in final csv: {len(selected_entries)}")

    utils.write_csv_file(selected_entries, pathlib.Path(outfile))


if __name__ == "__main__":
    cli()
