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

import functools
import random
import pathlib
import tempfile
import logging
import timeit
import multiprocessing
from typing import Optional

import click
import numpy as np
from tqdm import tqdm
from torch.utils import data
from PIL import Image

from omgfuser import datasets
from omgfuser import utils


logging.getLogger().setLevel(logging.INFO)


@click.command()
@click.option("--samples", type=int, default=4, show_default=True)
@click.option("--workers", type=int, default=2, show_default=True)
@click.option("--dataset_dir", type=click.Path(file_okay=False, path_type=pathlib.Path))
@click.option("--dataset_csv", type=click.Path(dir_okay=False, exists=True, path_type=pathlib.Path))
@click.option("--lmdb_storage",
              type=click.Path(dir_okay=False, exists=True, path_type=pathlib.Path))
@click.option("--batch_size", type=int, default=40, show_default=True)
def cli(
    samples: int,
    workers: int,
    dataset_dir: Optional[pathlib.Path],
    dataset_csv: Optional[pathlib.Path],
    lmdb_storage: Optional[pathlib.Path],
    batch_size: int
) -> None:
    if dataset_csv is not None:
        if dataset_dir is None:
            dataset_dir = dataset_csv.parent
        benchmark_dataloading(dataset_csv, dataset_dir, workers, lmdb_storage, batch_size)
    elif dataset_dir is None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_dir: pathlib.Path = pathlib.Path(tmp_str)

            start: float = timeit.default_timer()
            dataset_csv: pathlib.Path = create_random_dataset(samples, tmp_dir)
            stop: float = timeit.default_timer()
            logging.info(f"Dataset generation time: {stop-start:.3f} secs")

            benchmark_dataloading(dataset_csv, tmp_dir, workers, lmdb_storage, batch_size)
    elif not dataset_dir.exists():
        dataset_dir.mkdir(parents=True)

        start: float = timeit.default_timer()
        dataset_csv: pathlib.Path = create_random_dataset(samples, dataset_dir)
        stop: float = timeit.default_timer()
        logging.info(f"Dataset generation time: {stop - start:.3f} secs")

        benchmark_dataloading(dataset_csv, dataset_dir, workers, lmdb_storage, batch_size)
    else:
        benchmark_dataloading(dataset_dir/"random_dataset.csv", dataset_dir, workers, lmdb_storage,
                              batch_size)


def benchmark_dataloading(
    dataset_csv: pathlib.Path,
    dataset_dir: pathlib.Path,
    workers: int,
    lmdb_storage: Optional[pathlib.Path],
    batch_size: int
) -> None:
    assert dataset_csv.exists()
    splits: list[datasets.Split] = [datasets.Split.TRAIN_SPLIT]
    input_signals: list[str] = ["image", "instances"]
    input_channels: list[int] = [3, 0]
    target_size: tuple[int, int] = (224, 224)

    for split in splits:
        dataset: datasets.HandcraftedForensicsSignalsDataset = \
            datasets.HandcraftedForensicsSignalsDataset(
                dataset_csv,
                dataset_dir,
                input_signals,
                input_channels,
                split,
                target_size,
                lmdb_file_storage_path=lmdb_storage
            )
        dataloader: data.DataLoader = data.DataLoader(dataset,
                                                      batch_size=batch_size,
                                                      num_workers=workers,
                                                      collate_fn=dataset.build_collate_fn())

        total_loading_time: float = 0.
        start: float = timeit.default_timer()
        for batch in dataloader:

            load_time: float = batch["load_time"].sum().item()
            augment_time: float = batch["augment_time"].sum().item()
            conversion_time: float = batch["conversion_time"].sum().item()
            attention_mask_time: float = batch["attention_mask_time"].sum().item()

            stop: float = timeit.default_timer()
            elapsed_time: float = stop - start
            logging.info(f"\tSample loading time: {elapsed_time:.3f} secs | "
                         f"Load Time: {load_time:.3f} s | "
                         f"Augment Time: {augment_time:.3f} s | "
                         f"Conv. Time: {conversion_time:.3f} s | "
                         f"Att. Mask Time: {attention_mask_time:.3f} s")
            total_loading_time += elapsed_time
            start: float = timeit.default_timer()

        logging.info(f"Total loading time: {total_loading_time:.3f} secs")


def create_random_dataset(
    samples_num: int,
    out_dir: pathlib.Path,
    signals_num: int = 1
) -> pathlib.Path:
    if samples_num < 16:
        entries: list[dict[str, str]] = []
        for i in tqdm(range(samples_num), desc="Generating random samples", unit="sample"):
            entries.append(create_random_sample(str(i), out_dir, signals_num))
    else:
        logging.info(f"Generating {samples_num} samples in parallel.")
        with multiprocessing.Pool() as pool:
            entries: list[dict[str, str]] = pool.map(
                functools.partial(create_random_sample, out_dir=out_dir, signals_num=signals_num),
                [str(i) for i in range(samples_num)]
            )

    csv_path: pathlib.Path = out_dir / "random_dataset.csv"
    utils.write_csv_file(entries, csv_path)
    return csv_path


def create_random_sample(
    name: str,
    out_dir: pathlib.Path,
    signals_num: int
) -> dict[str, str]:
    sample_dir: pathlib.Path = out_dir / name
    sample_dir.mkdir()

    sample_height: int = random.randint(1024, 3092)
    sample_width: int = random.randint(1024, 3092)

    # Create random image
    image: np.ndarray = np.random.randint(0, 256, (sample_height, sample_width, 3), dtype=np.uint8)
    image_path: pathlib.Path = sample_dir / "image.jpg"
    Image.fromarray(image).save(image_path)

    # Decide whether image is manipulated or not.
    manipulated: bool = bool(random.randint(0, 1))

    # Create random mask
    if manipulated:
        mask: np.ndarray = np.random.randint(
            0, 2, (sample_height, sample_width), dtype=np.uint8
        ) * 255
    else:
        mask: np.ndarray = np.zeros((sample_height, sample_width), dtype=np.uint8)
    mask_path: pathlib.Path = sample_dir / "mask.png"
    Image.fromarray(mask).save(mask_path)

    # Create random instances
    instances_num: int = random.randint(1, 120)
    instances_entries: list[dict[str, str]] = []
    instances_dir: pathlib.Path = sample_dir / "instances"
    instances_dir.mkdir()
    for i in range(instances_num):
        instance_mask: np.ndarray = np.random.randint(
            0, 2, (sample_height, sample_width), dtype=np.uint8
        ) * 255
        instance_path: pathlib.Path = instances_dir / f"{i}.png"
        Image.fromarray(instance_mask).save(instance_path)
        instances_entries.append({
            "seg_model": "random",
            "seg_map": instance_path.name,
            "seg_score": 1.0,
            "class_id": 0
        })
    instances_csv_path: pathlib.Path = instances_dir / "instances.csv"
    utils.write_csv_file(instances_entries, instances_csv_path)

    # Create random signals
    signal_names: list[str] = [f"signal_{i}" for i in range(signals_num)]
    signal_paths: list[pathlib.Path] = [sample_dir / f"signal_{i}.png" for i in range(signals_num)]
    for s_path in signal_paths:
        signal: np.ndarray = np.random.randint(
            0, 256, (sample_height, sample_width), dtype=np.uint8
        )
        Image.fromarray(signal).save(s_path)

    return {
        "image": str(image_path.relative_to(out_dir)),
        "mask": str(mask_path.relative_to(out_dir)),
        "split": "train",
        "detection": "TRUE" if manipulated else "FALSE",
        "instances": str(instances_csv_path.relative_to(out_dir)),
        **{s_name: s_path.relative_to(out_dir)
           for s_name, s_path in zip(signal_names, signal_paths)}
    }


if __name__ == "__main__":
    cli()
