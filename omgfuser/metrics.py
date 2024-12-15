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

from typing import Optional

import numpy as np
import torch
import torchmetrics
import torchmetrics.functional.classification as class_metrics


class Metrics:

    def __init__(self,
                 metrics: tuple[str] = ('auc', 'iou', 'precision', 'recall', 'f1', 'accuracy',
                                        'f1-best'),
                 threshold: float = 0.5,
                 average=None,
                 return_on_update: bool = True,
                 sync_on_compute: bool = False,
                 approximate_auc: bool = False):

        self.metrics: tuple[str] = metrics
        self.threshold: float = threshold
        self.average = average
        self.return_on_update: bool = return_on_update

        # Map of metrics' methods implemented internally to their names.
        self.builtin_metric_functions = {'iou': calculate_iou,
                                         'precision': calculate_precision,
                                         'recall': calculate_recall,
                                         'f1': calculate_f1,
                                         'accuracy': calculate_accuracy}

        # Internal state is a Confusion Matrix and a ROC calculator.
        self.confusion_matrix: torchmetrics.classification.BinaryConfusionMatrix = \
            torchmetrics.classification.BinaryConfusionMatrix(threshold=self.threshold,
                                                              sync_on_compute=sync_on_compute)
        self.auroc: Optional[torchmetrics.classification.BinaryAUROC] = None
        self.mean_f1best: Optional[torchmetrics.aggregation.MeanMetric] = \
            torchmetrics.aggregation.MeanMetric(sync_on_compute=sync_on_compute)
        if "auc" in self.metrics:
            thresholds: Optional[int] = 100 if approximate_auc else None
            self.auroc = torchmetrics.classification.BinaryAUROC(sync_on_compute=sync_on_compute,
                                                                 thresholds=thresholds)

    def get_metric_names(self) -> tuple[str]:
        return self.metrics

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> dict[str, np.ndarray]:
        """Updates the metrics with the data of a batch.

        :param preds: A tensor of shape (N, ...) containing the predicted values.
            Values should be in [0, 1].
        :param targets: A tensor of shape (N, ...) containing the ground-truth values.
            Values should be in [0, 1].
        :return: A dictionary with the partial metrics computed on the given batch.
        """
        # Binarize ground-truth map to be 100% sure that it contains exactly two values.
        targets = torch.where(targets > self.threshold, 1.0, 0.0)
        # Convert the targets to classes.
        targets = targets.long()

        res: dict[str, np.ndarray] = {}

        if len(self.metrics) > 0:
            # Update confusion matrix and AUC.
            conf_matrix: torch.Tensor = self.confusion_matrix(preds, targets)
            if 'auc' in self.metrics:
                auc: torch.Tensor = self.auroc(preds, targets)

            # Compute the results for each metric.
            for metric in self.metrics:
                if metric == 'auc':
                    res[metric] = auc.numpy()
                elif metric == 'f1-best':
                    if torch.max(targets).numpy().item() > 0:
                        batched_bestf1: torch.Tensor = calculate_f1best(preds, targets)
                    else:
                        batched_bestf1: torch.Tensor = calculate_f1best(-preds+1, -targets+1)
                    self.mean_f1best.update(batched_bestf1)
                    res[metric] = torch.mean(batched_bestf1).detach().numpy()
                else:
                    res[metric] = self.builtin_metric_functions[metric](
                        conf_matrix, average=self.average
                    ).numpy()

        return res

    def compute(self) -> dict[str, np.ndarray]:
        """Computes the metrics for all the accumulated data.

        :return: A dictionary that maps the names of the computed metrics to their values.
        """
        res: dict[str, np.ndarray] = {}

        if len(self.metrics) > 0:
            conf_matrix: torch.Tensor = self.confusion_matrix.compute()
            if 'auc' in self.metrics:
                auc: torch.Tensor = self.auroc.compute()

            # Compute the results for each metric.
            for metric in self.metrics:
                if metric == 'auc':
                    res[metric] = auc.numpy()
                elif metric == 'f1-best':
                    res[metric] = self.mean_f1best.compute().detach().numpy()
                else:
                    res[metric] = self.builtin_metric_functions[metric](
                        conf_matrix, average=self.average
                    ).numpy()

        return res

    def reset(self) -> None:
        """Resets the internal state of the metrics.

        Should be called upon each new epoch.
        """
        if self.confusion_matrix:
            self.confusion_matrix.reset()
        if self.auroc:
            self.auroc.reset()
        if self.mean_f1best:
            self.mean_f1best.reset()


def calculate_accuracy(conf_matrix: torch.tensor, average=None) -> torch.tensor:
    return conf_matrix.diag().sum() / conf_matrix.sum()


def calculate_iou(conf_matrix: torch.tensor, average=None) -> torch.tensor:
    true_positive = torch.diag(conf_matrix)
    false_positive = conf_matrix.sum(0) - true_positive
    false_negative = conf_matrix.sum(1) - true_positive
    iou = true_positive / (true_positive + false_positive + false_negative)
    if average == 'macro':
        return iou.mean()
    else:
        return iou


def calculate_precision(conf_matrix: torch.tensor, average=None) -> torch.tensor:
    true_positive = torch.diag(conf_matrix)
    false_positive = conf_matrix.sum(0) - true_positive
    precision = true_positive / (true_positive + false_positive)
    # When no positive predictions exist, precision is set 1.0
    precision = torch.where(torch.isnan(precision), torch.ones_like(precision), precision)
    if average == 'macro':
        return precision.mean()
    else:
        return precision


def calculate_recall(conf_matrix: torch.tensor, average=None) -> torch.tensor:
    true_positive = torch.diag(conf_matrix)
    false_negative = conf_matrix.sum(1) - true_positive
    recall = true_positive / (true_positive + false_negative)
    if average == 'macro':
        return recall.mean()
    else:
        return recall


def calculate_f1(conf_matrix: torch.tensor, average=None) -> torch.tensor:
    true_positive = torch.diag(conf_matrix)
    false_negative = conf_matrix.sum(1) - true_positive
    false_positive = conf_matrix.sum(0) - true_positive
    precision = true_positive / (true_positive + false_positive)
    # When no positive predictions exist, precision is set 1.0
    precision = torch.where(torch.isnan(precision), torch.ones_like(precision), precision)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)
    # If after correction of precision f1 is NaN, then precision == 0 and recall == 0, meaning
    # that true_positives == 0 and false_positives > 0. So, set f1 to 0.
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    if average == 'macro':
        return f1.mean()
    else:
        return f1


def calculate_f1best(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes the F1 score for the best threshold on each entry in the mini-batch.

    :param preds: A tensor of shape (N, ...).
    :param target: A tensor of shape (N, ...).
    :returns: A tensor of shape (N,).
    """
    assert preds.size(dim=0) == target.size(dim=0)
    batched_bestf1: torch.Tensor = torch.zeros((preds.size(dim=0),))
    for i in range(preds.size(dim=0)):
        precision, recall, _ = class_metrics.binary_precision_recall_curve(preds, target)
        precision = torch.where(torch.isnan(precision), torch.ones_like(precision), precision)
        f1: torch.Tensor = 2 * recall * precision / (recall + precision)
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
        batched_bestf1[i] = torch.max(f1)
    return batched_bestf1
