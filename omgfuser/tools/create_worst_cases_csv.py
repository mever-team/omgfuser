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
import logging
from typing import Any, Optional

import click

from omgfuser.data.filestorage import read_csv_file
from omgfuser.utils import write_csv_file


__version__: str = "1.0.0"
__revision__: int = 1
__author__: str = "Dimitrios S. Karageorgiou"
__email__: str = "dkarageo@iti.gr"

logging.getLogger().setLevel(logging.INFO)


@click.command()
@click.option('-m', '--metric', type=str, required=True,
              help="Metric according which dataset samples will be filtered.")
@click.option('-a', '--algorithm', type=str, required=True, multiple=True,
              help="Algorithms for which the provided metric should be below the given threshold.")
@click.option('-t', '--threshold', type=float,
              help="Threshold under which the provided metric should be for all the provided "
                   "algorithms, in order for a sample to be included into the filtered dataset.")
@click.option('-p', '--percentage', type=float)
@click.option('--dataset_csv',
              type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
              required=True,
              show_default=True)
@click.option('--output_dir',
              type=click.Path(file_okay=False, path_type=pathlib.Path),
              help="Path to a directory where the filtered dataset csv will be written.")
def cli(
    metric: str,
    algorithm: list[str],
    threshold: Optional[float],
    percentage: Optional[float],
    dataset_csv: pathlib.Path,
    output_dir: pathlib.Path
) -> None:
    assert percentage is not None or threshold is not None
    assert percentage is None or threshold is None

    entries: list[dict[str, Any]] = read_csv_file(dataset_csv)

    def filter_samples(th: float) -> list[dict[str, Any]]:
        filt_entries: list[dict[str, Any]] = []
        for e in entries:
            for alg in algorithm:
                column_name: str = f"{alg}_{metric}"
                if float(e[column_name]) >= th:
                    break
            else:
                filt_entries.append(e)
        return filt_entries

    if threshold is not None:
        filtered_entries: list[dict[str, Any]] = filter_samples(threshold)
    else:
        threshold = 0.01
        filtered_entries: list[dict[str, Any]] = filter_samples(threshold)
        while (len(filtered_entries) / len(entries)) < percentage and threshold < 1.0:
            threshold += 0.01
            filtered_entries = filter_samples(threshold)

    logging.info(f"Filtered samples: {len(filtered_entries)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file: pathlib.Path = output_dir / f"{dataset_csv.stem}_filtered_{metric}_{threshold}.csv"
    write_csv_file(filtered_entries, output_file)
    logging.info(f"Exported filtered CSV to {output_file}")


if __name__ == "__main__":
    cli()
