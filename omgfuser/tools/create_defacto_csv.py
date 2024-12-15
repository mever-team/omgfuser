"""Script for generating train-validation CSV for the DEFACTO dataset.

The generated CSV contains the paths to the outputs of multiple image forensics algorithms.

The dataset is split as 90% training data - 10% validation data.

In order to use that script, a base CSV file for the DEFACTO dataset is required. That
base CSV file should contain at least the columns `image`, `image_simple` and `mask`.

Version: 2.0

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
import copy
import hashlib
import pathlib
import logging
from typing import List, Dict, Any, Optional

import click
import tqdm
from sklearn.model_selection import train_test_split


logging.getLogger().setLevel(logging.INFO)


ALGORITHMS: dict[str, str] = {
    "eva": "EVA",
    "eva_coco": "EVA",
    "eva_elvis": "EVA",
    "eva_raw": "EVA",
    "adq1": "ADQ1",
    "adq2": "ADQ2",
    "blk": "BLK",
    "cagi": "CAGI",
    "catnetv2": "CATNetv2",
    "cfa": "CFA",
    "cmfd": "CMFD",
    "dct": "DCT",
    "fusion": "Fusion",
    "ifosn": "IFOSN",
    "mantranet": "Mantranet",
    "mvssnetplus": "MVSSNetPlus",
    "noiseprint": "Noiseprint",
    "psccnet": "PSCCNet",
    "span": "SPAN",
    "splicebuster": "Splicebuster_gmm_num=10",
    "wavelet": "Wavelet",
    "zero": "Zero",
}

COLUMNS_OUTPUT_ORDER: list[str] = [
    "image",
    "mask",
    "detection",
    "split",
    "eva",
    "eva_coco",
    "eva_elvis",
    "eva_raw",
    "adq1",
    "blk",
    "cagi",
    "dct",
    "splicebuster",
    "noiseprint",
    "mantranet",
    "span",
    "fusion",
    "adq2",
    "cfa",
    "cmfd",
    "wavelet",
    "zero",
    "catnetv2",
    "psccnet",
    "mvssnetplus",
    "ifosn"
]


@click.command()
@click.option("-i", "--dataset_dir", required=True,
              type=click.Path(file_okay=False, exists=True, path_type=pathlib.Path),
              help="Base directory of the DEFACTO dataset. Paths inside the `dataset_csv` "
                   "should be relative to that directory.")
@click.option("-c", "--dataset_csv", required=True,
              type=click.Path(dir_okay=False, exists=True, path_type=pathlib.Path),
              help="A base CSV file defining the image samples of the dataset. It should contain "
                   "at least the columns `image`, `image_simple` and `mask`.")
@click.option("-o", "--output_dir", required=True,
              type=click.Path(file_okay=False, exists=True, path_type=pathlib.Path),
              help="Directory where the generated CSV will be exported. The paths in the CSV "
                   "will be relative to that directory. For that reason, the output directory "
                   "should either match the `dataset_dir` or should be a parent directory. "
                   "The intended usage of that argument is to make the paths included into "
                   "the generated CSV relative to the outer directory of the DEFACTO dataset.")
def cli(
    dataset_dir: pathlib.Path,
    dataset_csv: pathlib.Path,
    output_dir: pathlib.Path
) -> None:
    assert dataset_dir.is_relative_to(output_dir), \
        "Dataset dir should be located under output dir."

    entries: list[Dict[str, str]] = read_csv_file(dataset_csv)

    non_duplicate_entries: list[Dict[str, str]] = find_non_duplicate_entries(entries, dataset_dir)
    logging.info(
        f"TOTAL ENTRIES: {len(entries)}  |  NON-DUPLICATE ENTRIES: {len(non_duplicate_entries)}"
    )

    non_duplicate_entries = find_algorithms_outputs(non_duplicate_entries,
                                                    dataset_dir/"Predictions")
    non_duplicate_entries = create_train_val_test_splits(non_duplicate_entries)
    non_duplicate_entries = make_paths_relative_to_output_dir(non_duplicate_entries,
                                                              dataset_dir,
                                                              output_dir)

    # Add detection column. Currently, all items in DEFACTO are manipulated.
    for e in non_duplicate_entries:
        e["detection"] = "TRUE"

    csv_output_file: pathlib.Path = output_dir / "dataset_algorithms.csv"
    write_csv_file(non_duplicate_entries, csv_output_file, COLUMNS_OUTPUT_ORDER)
    logging.info(f"Exported CSV to: {str(csv_output_file)}")


def find_non_duplicate_entries(
    entries: list[Dict[str, str]],
    base_dir: pathlib.Path = pathlib.Path("../../scripts"),
    file_column: str = "image"
) -> list[Dict[str, str]]:
    redundant_files: list[pathlib.Path] = find_redundant_files(
        [base_dir/e[file_column] for e in entries]
    )
    non_duplicate_entries: list[Dict[str, str]] = [
        e for e in entries if (base_dir/e[file_column]) not in redundant_files
    ]
    return non_duplicate_entries


def find_algorithms_outputs(
    entries: list[Dict[str, str]],
    algorithms_dir: pathlib.Path = pathlib.Path("./Predictions"),
    simple_file_column: str = "image_simple"
) -> list[Dict[str, str]]:
    entries = copy.deepcopy(entries)

    for e in tqdm.tqdm(entries, desc="Finding algorithms outputs", unit="sample"):
        for a_col, a_dir in ALGORITHMS.items():
            if "eva" in a_col:
                eva_dir: pathlib.Path = algorithms_dir / a_dir / pathlib.Path(e["image"]).parent
                sample_name: str = pathlib.Path(e["image"]).stem
                algorithm_out_path = find_eva_maps(a_col, eva_dir, sample_name)
            else:
                algorithm_out_name: str = f"{pathlib.Path(e[simple_file_column]).stem}.png"
                algorithm_out_path: pathlib.Path = (
                    algorithms_dir / a_dir / "manipulated_masks" / algorithm_out_name
                )

            if algorithm_out_path.exists():
                e[a_col] = str(algorithm_out_path)
            else:
                e[a_col] = ""

    return entries


def find_eva_maps(
    map_type: str,
    maps_directory: pathlib.Path,
    base_name: str
) -> pathlib.Path:
    p: pathlib.Path
    if map_type == "eva":
        p = maps_directory / f"{base_name}_map.png"
    elif map_type == "eva_elvis":
        p = maps_directory / f"{base_name}_elvis_map.png"
    elif map_type == "eva_coco":
        p = maps_directory / f"{base_name}_coco_map.png"
    elif map_type == "eva_raw":
        p = maps_directory / base_name / "segmentation_instances.csv"
    else:
        raise RuntimeError(f"Unrecognized type of EVA map: {map_type}")
    return p


def create_train_val_test_splits(
    entries: list[Dict[str, str]]
) -> list[Dict[str, str]]:
    train_entries: list[Dict[str, str]]
    val_entries: list[Dict[str, str]]
    train_entries, val_entries = train_test_split(entries, train_size=0.9, random_state=123)
    updated_entries: list[dict[str, str]] = []
    logging.info(f"TRAINING SAMPLES: {len(train_entries)}")
    for e in train_entries:
        e = e.copy()
        e["split"] = "train"
        updated_entries.append(e)
    logging.info(f"VALIDATION SAMPLES: {len(val_entries)}")
    for e in val_entries:
        e = e.copy()
        e["split"] = "val"
        updated_entries.append(e)
    return updated_entries


def make_paths_relative_to_output_dir(
    entries: list[dict[str, str]],
    dataset_root: pathlib.Path,
    output_dir: pathlib.Path
) -> list[dict[str, str]]:
    entries = copy.deepcopy(entries)
    for e in entries:
        for alg in ["image", "mask", *ALGORITHMS.keys()]:
            if e[alg] != "":
                p: pathlib.Path = pathlib.Path(e[alg])
                if not p.is_absolute():
                    p = dataset_root / p
                assert p.exists()
                p = p.relative_to(output_dir)
                e[alg] = str(p)
    return entries


def find_redundant_files(files: list[pathlib.Path]) -> list[pathlib.Path]:
    duplicate_groups: list[list[pathlib.Path]] = find_duplicate_files(files)
    files_to_remove: list[pathlib.Path] = []
    for g in duplicate_groups:
        files_to_remove.extend(g[1:])
    return files_to_remove


def find_duplicate_files(files: list[pathlib.Path]) -> list[list[pathlib.Path]]:
    files_by_hash: dict[str, list[pathlib.Path]] = {}
    for f in tqdm.tqdm(files, desc="Calculating MD5 hashes", unit="file"):
        retries: int = 5
        while retries > 0:
            try:
                file_md5: str = md5(f)
                break
            except OSError as e:
                if retries > 0:
                    retries -= 1
                else:
                    raise e
        md5_entries: list[pathlib.Path] = files_by_hash.get(file_md5, [])
        md5_entries.append(f)
        files_by_hash[file_md5] = md5_entries
    duplicate_files: list[list[pathlib.Path]] = [
        g for g in files_by_hash.values() if len(g) > 1
    ]
    return duplicate_files


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


def md5(file: pathlib.Path) -> str:
    """Calculates md5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


if __name__ == "__main__":
    cli()
