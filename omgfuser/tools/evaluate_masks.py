"""Tool for evaluating precomputed forgery localization masks and detection results of a dataset.

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

import logging
import pathlib
from typing import Optional, Any

import click
import numpy as np
import albumentations as A
from tqdm import tqdm
from torch.nn import functional
from torch.utils.data import DataLoader
import torch.multiprocessing

from omgfuser import datasets
from omgfuser import metrics
from omgfuser import training
from omgfuser.data.filestorage import read_csv_file
from omgfuser.utils import write_csv_file


torch.multiprocessing.set_sharing_strategy('file_system')
logging.getLogger().setLevel(logging.INFO)


__version__: str = "1.3.0"
__revision__: int = 4
__author__: str = "Dimitrios S. Karageorgiou"
__email__: str = "dkarageo@iti.gr"


@click.command()
@click.option('-m', '--metrics_names', type=str,
              default="auc,iou,precision,recall,f1,accuracy,f1-best",
              show_default=True,
              help="A comma separated list of the metrics to be computed.")
@click.option('--dataset_csv',
              type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
              required=True,
              show_default=True)
@click.option('--dataset_root',
              type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path))
@click.option('--data_workers', type=int, default=1)
@click.option('--input_signals', required=True, type=str)
@click.option('--signals_channels', required=True, type=str)
@click.option('--mask_signal', type=str, default="mask")
@click.option('--precomputed_detections', type=str,
              help="A comma-separated list of signals for which detection results are included "
                   "in the provided CSV file. The column containing these results should be named "
                   "as `<signal>_detection`. The value for each sample should be a floating point "
                   "number in the range of [0, 1]. Lower values denote that a sample is authentic, "
                   "while higher values denote that a sample has been tampered."
              )
# Data loader loads train augmentations when provided with a train split.
@click.option('--split', type=click.Choice(["train", "test", "eval"]), default="test")
@click.option('--approximate_auc', is_flag=True, default=False,
              show_default=True,
              help="Flag for switching to an approximate computation of pixel-level AUC. "
                   "That computation does not require the whole dataset to fit in memory. "
                   "In big datasets, or datasets with very large images, providing "
                   "that flag is necessary in order not to run out of memory."
              )
@click.option('--output_dir',
              type=click.Path(file_okay=False, path_type=pathlib.Path),
              help="Path to a directory where the results csv will be written. The results csv "
                   "will be the dataset_csv augmented with a new column for each signal-metric "
                   "pair. When this option is omitted, no results csv will be written.")
def evaluate(metrics_names: str,
             dataset_csv: pathlib.Path,
             dataset_root: Optional[pathlib.Path],
             data_workers: int,
             input_signals: str,
             signals_channels: str,
             mask_signal: str,
             precomputed_detections: Optional[str],
             split: str,
             approximate_auc: bool,
             output_dir: Optional[pathlib.Path]) -> None:

    metrics_names_separate: list[str] = metrics_names.split(",")

    input_signals_separate: list[str] = input_signals.split(",")
    signals_channels_separate: list[int] = [int(c) for c in signals_channels.split(",")]
    signals_with_precomputed_detections: set[str] = set()
    if precomputed_detections is not None:
        signals_with_precomputed_detections.update(precomputed_detections.split(","))

    if not dataset_root:
        dataset_root = dataset_csv.absolute().parent

    dataset_entries: list[dict[str, Any]] = read_csv_file(dataset_csv)

    for signal, channels in zip(input_signals_separate, signals_channels_separate):
        target_image_size: tuple[int, int] = (224, 224)
        data: datasets.ForensicsDataset = datasets.ForensicsDataset(
            csv_file=dataset_csv,
            root_dir=dataset_root,
            signals_columns=[signal],
            signals_channels=signals_channels_separate,
            split=datasets.Split(split),
            target_image_size=target_image_size,
            mask_column=mask_signal,
            resize_mask=False,
            transforms_generator=masks_augmentations
        )
        data_loader = DataLoader(dataset=data,
                                 shuffle=False,
                                 batch_size=1,
                                 num_workers=data_workers)

        precomputed_detection_results: Optional[dict[str, np.ndarray]] = None
        if signal in signals_with_precomputed_detections:
            precomputed_detection_results = evaluate_precomputed_detection_results(
                dataset_csv,
                signal,
                split
            )

        results: DatasetEvaluationResults = evaluate_signal(
            data_loader,
            signal_name=signal,
            approximate_auc=approximate_auc,
            metrics_names=tuple(metrics_names_separate),
            precomputed_detection_results=precomputed_detection_results
        )
        results.add_to_dataset_entries(dataset_entries, signal)

    if output_dir is not None:
        # Export the augmented dataset entries to a CSV file into the output directory.
        output_dir.mkdir(parents=True, exist_ok=True)
        file_name: str = f"{dataset_csv.stem}_with_results.csv"
        output_file: pathlib.Path = output_dir / file_name
        write_csv_file(dataset_entries, output_file)


def evaluate_signal(
    data_loader: DataLoader,
    signal_name: str,
    approximate_auc: bool,
    metrics_names: tuple[str],
    precomputed_detection_results: Optional[dict[str, np.ndarray]],
) -> 'DatasetEvaluationResults':
    if approximate_auc:
        logging.info("Utilizing approximate pixel-level AUC computation.")
    localization_metrics: metrics.Metrics = metrics.Metrics(metrics=metrics_names,
                                                            approximate_auc=approximate_auc)
    detection_metrics: metrics.Metrics = metrics.Metrics(metrics=metrics_names)

    per_sample_results: list[tuple[int, dict[str, np.ndarray]]] = []

    with tqdm(total=len(data_loader),
              unit="batch",
              desc=f"Validating {signal_name}",
              postfix={metric: np.nan
                       for metric in localization_metrics.get_metric_names()}) as progress_bar:

        for batch in data_loader:
            progress_bar.update(1)
            signal_batch, mask_batch, detection_ground_truth, index = batch

            signal_batch = signal_batch.detach().cpu()
            mask_batch = mask_batch.detach().cpu()
            if not signal_batch.size() == mask_batch.size():
                signal_batch = functional.interpolate(
                    signal_batch, (mask_batch.size(dim=2), mask_batch.size(dim=3))
                )
            loc_results: dict[str, np.ndarray] = localization_metrics.update(
                signal_batch, mask_batch
            )
            per_sample_results.append((index, loc_results))

            if precomputed_detection_results is None:
                # Detection scores are generated from the signals through max pooling.
                signal_detection_score, _ = torch.max(
                    torch.flatten(signal_batch, start_dim=1), dim=1
                )
                det_results = detection_metrics.update(signal_detection_score,
                                                       detection_ground_truth)
                results = training._combine_localization_detection_results(loc_results, det_results)
            else:
                results = loc_results

            progress_bar.set_postfix(**results)

        loc_results = localization_metrics.compute()

        if precomputed_detection_results is None:
            det_results = detection_metrics.compute()
        else:
            logging.info(f"Using precomputed detection results for {signal_name}")
            det_results = precomputed_detection_results

        results = training._combine_localization_detection_results(loc_results, det_results)
        progress_bar.set_postfix(**results)
        progress_bar.close()

        training.log_localization_detection_results(loc_results, det_results, signal_name)

    return DatasetEvaluationResults(
        results,
        per_sample_results
    )


def evaluate_precomputed_detection_results(
    dataset_csv: pathlib.Path,
    signal: str,
    split: str
) -> dict[str, np.ndarray]:
    """Computes evaluation metrics out of the precomputed forgery detection results in a CSV.

    :param dataset_csv: Path to the CSV file.
    :param signal: Name of the signal to be evaluated for the task of image tampering detection.
        The results of that signal should be included into a column named as `<signal>_detection`.
    :param split: The split of the dataset for which the evaluation metrics will be computed.
        Expected names are `train`, `eval`, `test`.

    :return: A dictionary of metrics like AUC and F1.
    """
    entries: list[dict[str, str]] = read_csv_file(dataset_csv)
    entries = [e for e in entries if e["split"] == split]

    detection_column: str = f"{signal}_detection"
    detection_metrics: metrics.Metrics = metrics.Metrics()

    with tqdm(total=len(entries),
              desc=f"Evaluating precomputed detection results for {signal}",
              unit="sample",
              postfix={metric: np.nan for metric in detection_metrics.get_metric_names()}
              ) as progress_bar:

        for e in entries:
            progress_bar.update(1)
            detection_score: torch.Tensor = torch.Tensor([float(e[detection_column])])
            assert e["detection"] in ["True", "False", "TRUE", "FALSE"]
            detection: bool = e["detection"] in ["TRUE", "True", "true"]
            groundtruth_score: torch.Tensor = torch.Tensor([int(detection)])

            det_results: dict[str, np.ndarray] = detection_metrics.update(
                detection_score, groundtruth_score
            )
            progress_bar.set_postfix(**det_results)

    det_results = detection_metrics.compute()
    progress_bar.set_postfix(**det_results)
    progress_bar.close()

    return det_results


def masks_augmentations(target_shape: tuple[int, int],
                        additional_targets: dict[str, str]) -> A.Compose:
    """Generate augmentations that do not resize input signals, as train or test ones."""
    return A.Compose([A.Normalize(mean=0., std=1.0, max_pixel_value=255.0)],
                     is_check_shapes=False)


class DatasetEvaluationResults:
    def __init__(
        self,
        overall_results: dict[str, np.ndarray],
        per_sample_results: list[tuple[int, dict[str, np.ndarray]]]
    ) -> None:
        self.overall_results: dict[str, np.ndarray] = overall_results
        self.per_sample_results: list[tuple[int, dict[str, np.ndarray]]] = per_sample_results

    def add_to_dataset_entries(
        self,
        dataset_entries: list[dict[str, Any]],
        prefix: str,
        detection_column: str = "detection"
    ) -> None:
        """Adds the evaluation results to the CSV entries of a dataset."""
        for r in self.per_sample_results:
            index: int = r[0]
            sample_results: dict[str, np.ndarray] = r[1]

            for metric, value in sample_results.items():
                prefixed_metric_name: str = f"{prefix}_{metric}"
                if np.size(value) > 1:
                    # For metrics that the definition of positive class matters, consider
                    # as positive the manipulated class for manipulated samples and the
                    # authentic class for authentic class. Otherwise
                    if dataset_entries[index][detection_column] in {"TRUE", "True", "true"}:
                        dataset_entries[index][prefixed_metric_name] = value[1]
                    else:
                        dataset_entries[index][prefixed_metric_name] = value[0]
                else:
                    dataset_entries[index][prefixed_metric_name] = value.item()


if __name__ == "__main__":
    evaluate()
