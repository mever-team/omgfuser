"""Script for generating CSV file for a dataset, given a simple-structure CSV already exists.

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
    "cfa": "CFA",
    "cmfd": "CMFD",
    "dct": "DCT",
    "fusion": "Fusion",
    "mantranet": "Mantranet",
    "noiseprint": "Noiseprint",
    "span": "SPAN",
    "splicebuster": "Splicebuster_gmm_num=10",
    "wavelet": "Wavelet",
    "zero": "Zero",
    "catnetv2": "CATNetv2",
    "psccnet": "PSCCNet",
    "mvssnetplus": "MVSSNetPlus",
    "ifosn": "IFOSN",
    "sam_raw": "SAM",
    "trufor": "TruFor"
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
    "ifosn",
    "sam_raw",
    "trufor"
]


@click.command()
@click.option("-i", "--dataset_dir", required=True,
              type=click.Path(file_okay=False, exists=True, path_type=pathlib.Path),
              help="Base path of the dataset. Paths in `images_column` should be relative "
                   "to that path.")
@click.option("--dataset_simple_dir",
              type=click.Path(file_okay=False, exists=True, path_type=pathlib.Path),
              help="Base path of the dataset transformed to a simple path structure. "
                   "Paths in `images_column_simple` should be relative to that path.")
@click.option("-c", "--dataset_csv", required=True,
              type=click.Path(dir_okay=False, exists=True, path_type=pathlib.Path),
              help="Path to a CSV file containing at least the columns defined in "
                   "`images_column` and `images_column_simple`. Also, the CSV should"
                   "contain an additional `mask` column.")
@click.option("--images_column", type=str, default="image")
@click.option("--images_column_simple", type=str, default="image_simple")
@click.option("--detection", type=bool)
@click.option("--split", type=str)
@click.option("-o", "--output_dir", required=True,
              type=click.Path(path_type=pathlib.Path))
@click.option("--output_filename", type=str, default="dataset_algorithms.csv")
def cli(
    dataset_dir: pathlib.Path,
    dataset_simple_dir: Optional[pathlib.Path],
    dataset_csv: pathlib.Path,
    images_column: str,
    images_column_simple: str,
    detection: Optional[bool],
    split: Optional[str],
    output_dir: pathlib.Path,
    output_filename: str,
) -> None:
    assert dataset_dir.is_relative_to(output_dir), \
        "Dataset dir should be located under output dir."

    entries: list[Dict[str, str]] = read_csv_file(dataset_csv)

    non_duplicate_entries: list[Dict[str, str]] = find_non_duplicate_entries(
        entries, dataset_dir, file_column=images_column
    )
    logging.info(
        f"TOTAL ENTRIES: {len(entries)}  |  NON-DUPLICATE ENTRIES: {len(non_duplicate_entries)}"
    )

    # Set the columns containing the outputs of each algorithm.
    if dataset_simple_dir is None:
        dataset_simple_dir = dataset_dir
    non_duplicate_entries = find_algorithms_outputs(
        non_duplicate_entries, dataset_simple_dir/"Predictions",
        simple_file_column=images_column_simple
    )

    # Set the split column.
    if split is not None:
        for e in non_duplicate_entries:
            e["split"] = split
    else:
        non_duplicate_entries = create_train_val_test_splits(non_duplicate_entries)

    non_duplicate_entries = make_paths_relative_to_output_dir(non_duplicate_entries,
                                                              dataset_dir,
                                                              output_dir)

    # Set the detection column.
    for e in non_duplicate_entries:
        if detection is not None:
            e["detection"] = "TRUE" if detection else "FALSE"
        else:
            assert "detection" in e

    csv_output_file: pathlib.Path = output_dir / output_filename
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
            if "eva" in a_col or "sam" in a_col:
                seg_dir: pathlib.Path = (
                        algorithms_dir / a_dir / pathlib.Path(e[simple_file_column]).parent)
                # seg_dir: pathlib.Path = (
                #         algorithms_dir / a_dir / pathlib.Path(e[simple_file_column]).parent.parent)
                sample_name: str = pathlib.Path(e[simple_file_column]).stem
                algorithm_out_path = find_segmentation_maps(a_col, seg_dir, sample_name)
            else:
                algorithm_out_name: str = f"{pathlib.Path(e[simple_file_column]).stem}.png"
                if e.get("detection", "TRUE") in ["TRUE", "True"]:
                    algorithm_out_path: pathlib.Path = (
                        algorithms_dir / a_dir / "manipulated_masks" / algorithm_out_name
                    )
                else:
                    algorithm_out_path: pathlib.Path = (
                            algorithms_dir / a_dir / "authentic_masks" / algorithm_out_name
                    )
                    # algorithm_out_path: pathlib.Path = (
                    #         algorithms_dir / a_dir / "manipulated_masks" / algorithm_out_name
                    # )

            e[a_col] = ""
            if algorithm_out_path.exists():
                e[a_col] = str(algorithm_out_path)
            elif "." in algorithm_out_path.stem:  # Fix for MVSSNet
                undoted_name: str = \
                    f"{pathlib.Path(algorithm_out_path.stem).stem}{algorithm_out_path.suffix}"
                undotted_file: pathlib.Path = algorithm_out_path.parent / undoted_name
                if undotted_file.exists():
                    e[a_col] = str(undotted_file)

    return entries


def find_segmentation_maps(
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
    elif map_type == "eva_raw" or map_type == "sam_raw":
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
