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
# import timeit
from enum import Enum
from typing import Optional, Union, Any

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional
from torch.nn.parallel import DistributedDataParallel
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
from torch.utils import tensorboard
from tqdm import tqdm
from warmup_scheduler_pytorch import WarmUpScheduler
from matplotlib import pyplot as plt

from . import metrics
from . import losses
from . import data_utils
from .data import processors


localization_metrics: metrics.Metrics = metrics.Metrics()
detection_metrics: metrics.Metrics = metrics.Metrics()


class ForwardPassType(Enum):
    SIMPLE_FORWARD_PASS = "simple"
    ATTENTION_MASK_PASS = "attention_mask"
    PREPROCESSOR_ATTENTION_MASK_PASS = "preprocessor_attention_mask"


def train_model(model: nn.Module,
                epochs: int,
                steps_per_epoch: int,
                device: torch.device,
                train_loader: data.DataLoader,
                val_loader: Union[data.DataLoader, dict[str, data.DataLoader]],
                optimizer: optim.Optimizer,
                scheduler,
                scheduler_step_per_epoch: bool,
                criterion: nn.Module,
                checkpoint_path: pathlib.Path,
                step: int = 0,
                rank: int = 0,
                writer: tensorboard.SummaryWriter = None,
                monitor_value=None,
                early_stop_patience: int = 5,
                save_all: bool = False,
                initial_epoch: int = 1,
                convert_detection_to_one_hot: bool = True,
                convert_localization_to_one_hot: bool = False,
                requires_sigmoid: bool = False,
                autocast: bool = False,
                accumulation_steps: int = 1,
                forward_pass_type: ForwardPassType = ForwardPassType.SIMPLE_FORWARD_PASS,
                train_processor: Optional[processors.OnlinePreprocessor] = None,
                val_processor: Optional[processors.OnlinePreprocessor] = None):

    model = model.to(device)
    criterion = criterion.to(device)
    if train_processor is not None:
        train_processor.to_device(device)

    optimizer.zero_grad()

    early_stop_patience_counter: int = early_stop_patience

    if autocast:
        logging.info("USING AUTOCAST - FP16 MIXED PRECISION TRAINING")
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(initial_epoch, epochs+1):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch-1)

        epoch_loss: float = .0
        model.train()

        # start_time: float = timeit.default_timer()

        dataloader_iterator = iter(train_loader)
        # Train on train samples for one epoch.
        for i in tqdm(range(steps_per_epoch),
                      desc=f"Training | Epoch {epoch}",
                      unit="batch"):
            batch = next(dataloader_iterator)

            # stop_time: float = timeit.default_timer()
            # logging.info(f"Rank {rank}: Batch loading time: {stop_time - start_time:.3f} secs")

            # Forward pass.
            if forward_pass_type == ForwardPassType.SIMPLE_FORWARD_PASS:
                loss: torch.Tensor = train_forward_pass(
                    batch, model, device, criterion, step, convert_detection_to_one_hot,
                    autocast, accumulation_steps, writer=writer,
                    convert_localization_to_one_hot=convert_localization_to_one_hot,
                    rank=rank
                )
            elif forward_pass_type == ForwardPassType.ATTENTION_MASK_PASS:
                loss: torch.Tensor = train_forward_pass_masked_attention(
                    batch, model, device, criterion, step, convert_detection_to_one_hot,
                    autocast, accumulation_steps, writer=writer,
                    convert_localization_to_one_hot=convert_localization_to_one_hot,
                    rank=rank
                )
            elif forward_pass_type == ForwardPassType.PREPROCESSOR_ATTENTION_MASK_PASS:
                loss: torch.Tensor = train_forward_pass_processor_masked_attention(
                    batch, model, device, criterion, step, train_processor,
                    convert_detection_to_one_hot,
                    autocast, accumulation_steps, writer=writer,
                    convert_localization_to_one_hot=convert_localization_to_one_hot,
                    rank=rank
                )
            else:
                raise RuntimeError(f"Unsupported forward pass type: {forward_pass_type.name}")

            if autocast:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # When accumulation_steps > 1, the update of the model and the update of
            # the training steps is performed once per accumulation_steps. For example,
            # training with batch size of 8 and accumulation_steps == 1, should
            # yield the same results as training with batch size of 4 and
            # accumulation_steps == 2.
            if ((i+1) % accumulation_steps == 0) or ((i+1) == len(train_loader)):
                if autocast:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                if scheduler is not None and scheduler_step_per_epoch:
                    # Update the LR scheduler once per batch.
                    if isinstance(scheduler, WarmUpScheduler):
                        if writer is not None:
                            writer.add_scalar("lr",
                                              scheduler.get_last_lr()[0],
                                              step)
                        scheduler.step()
                    else:
                        scheduler.step((epoch-1)+((i+1)/len(train_loader)))

                step += 1

            epoch_loss += loss.detach().cpu().item() / steps_per_epoch

            # start_time: float = timeit.default_timer()

        if scheduler is not None and not scheduler_step_per_epoch:
            # Update the LR scheduler once per epoch.
            scheduler.step()

        if rank == 0:
            # Compute the total training loss of the epoch.
            writer.add_scalar('train/loss', epoch_loss, epoch)

            # Evaluate on validation samples.
            val_metric_results = evaluate_model(
                model.module if isinstance(model, DistributedDataParallel) else model,
                criterion,
                val_loader,
                device,
                epoch,
                writer,
                convert_detection_to_one_hot=convert_detection_to_one_hot,
                convert_localization_to_one_hot=convert_localization_to_one_hot,
                requires_sigmoid=requires_sigmoid,
                forward_pass_type=forward_pass_type,
                processor=val_processor
            )
            val_loss = val_metric_results['loss']

            if (monitor_value is None or monitor_value > val_loss) or save_all:
                # Save the model when validation loss has been reduced from the previous one.
                monitor_value = val_loss
                early_stop_patience_counter = early_stop_patience
                print('saving to ckpt_{}_{}.pth'.format(epoch, step))
                torch.save(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'loss': val_loss
                    },
                    checkpoint_path / 'ckpt_{}_{}.pth'.format(epoch, step)
                )
            else:
                early_stop_patience_counter -= 1
                if early_stop_patience_counter == 0:
                    print('early stopping')


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Union[data.DataLoader, dict[str, data.DataLoader]],
    device: torch.device,
    epoch: int,
    writer: Optional[tensorboard.SummaryWriter] = None,
    return_per_sample_metrics: bool = False,
    convert_detection_to_one_hot: bool = True,
    convert_localization_to_one_hot: bool = False,
    requires_sigmoid: bool = False,
    forward_pass_type: ForwardPassType = ForwardPassType.SIMPLE_FORWARD_PASS,
    discard_mask_of_negative_predictions: bool = False,
    processor: Optional[processors.OnlinePreprocessor] = None
) -> Union[dict[str, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    model.to(device)
    criterion.to(device)

    # Switch model and loss function to evaluation mode.
    model.eval()
    criterion.eval()

    if not isinstance(data_loader, dict):
        data_loader: dict[str, data.DataLoader] = {"": data_loader}

    per_dataset_results: list[dict[str, Union[float, np.ndarray]]] = []

    for d_name, d_loader in data_loader.items():
        batches_num: int = len(d_loader)
        val_loss: float = .0
        with tqdm(total=len(d_loader),
                  unit="batch",
                  desc=f"Validation {d_name} | Epoch {epoch}",
                  postfix={metric: np.nan
                           for metric in localization_metrics.get_metric_names()}) as progress_bar:

            per_sample_metrics: dict[int, dict[str, np.ndarray]] = {}

            for batch in d_loader:
                progress_bar.update(1)

                if forward_pass_type == forward_pass_type.SIMPLE_FORWARD_PASS:
                    loss, results, sample_index = evaluation_pass(
                        batch, model, criterion, device,
                        convert_detection_to_one_hot=convert_detection_to_one_hot,
                        requires_sigmoid=requires_sigmoid,
                        return_per_sample_metrics=return_per_sample_metrics,
                        convert_localization_to_one_hot=convert_localization_to_one_hot,
                        discard_mask_of_negative_predictions=discard_mask_of_negative_predictions
                    )
                elif forward_pass_type == forward_pass_type.ATTENTION_MASK_PASS:
                    loss, results, sample_index = evaluation_pass_masked_attention(
                        batch, model, criterion, device,
                        convert_detection_to_one_hot=convert_detection_to_one_hot,
                        requires_sigmoid=requires_sigmoid,
                        return_per_sample_metrics=return_per_sample_metrics,
                        convert_localization_to_one_hot=convert_localization_to_one_hot,
                        discard_mask_of_negative_predictions=discard_mask_of_negative_predictions
                    )
                elif forward_pass_type == forward_pass_type.PREPROCESSOR_ATTENTION_MASK_PASS:
                    loss, results, sample_index = evaluation_pass_processor_masked_attention(
                        batch, model, criterion, device, processor,
                        convert_detection_to_one_hot=convert_detection_to_one_hot,
                        requires_sigmoid=requires_sigmoid,
                        return_per_sample_metrics=return_per_sample_metrics,
                        convert_localization_to_one_hot=convert_localization_to_one_hot,
                        discard_mask_of_negative_predictions=discard_mask_of_negative_predictions
                    )
                else:
                    raise RuntimeError(f"Unsupported forward pass type: {forward_pass_type.name}")

                if return_per_sample_metrics:
                    per_sample_metrics[sample_index] = results
                val_loss += loss.detach().cpu().item() / batches_num
                progress_bar.set_postfix(**results)

            loc_results = localization_metrics.compute()
            det_results = detection_metrics.compute()
            log_localization_detection_results(loc_results, det_results, signal_name="OMGFuser")
            results = _combine_localization_detection_results(loc_results, det_results)
            results['loss'] = val_loss
            per_dataset_results.append(results)
            progress_bar.set_postfix(**results)
            progress_bar.close()

            if writer:
                metrics_d_name = f"{d_name}_" if d_name != "" else ""
                _write_results_to_writer(results, writer, f'{metrics_d_name}val/', epoch)

            localization_metrics.reset()
            detection_metrics.reset()

    if writer:
        _write_results_to_writer(_average_results(per_dataset_results), writer, 'val/', epoch)

    if return_per_sample_metrics:
        if len(data_loader) > 1:
            # TODO: Implement support for multiple datasets. Currently, only one supported.
            logging.warning("Only evaluation results of the last dataset are returned.")
        return per_sample_metrics
    return results


def train_forward_pass(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    model: nn.Module,
    device: torch.device,
    criterion: nn.Module,
    step: int,
    convert_detection_to_one_hot: bool = True,
    autocast: bool = False,
    accumulation_steps: int = 1,
    writer: Optional[tensorboard.SummaryWriter] = None,
    convert_localization_to_one_hot: bool = False,
    rank: int = 0
) -> torch.Tensor:
    """Training forward pass for models that utilize RGB segmentation maps."""
    inputs_batch, mask_batch, prediction_batch, _ = batch

    inputs_batch = inputs_batch.to(device)
    mask_batch = mask_batch.to(device)
    prediction_batch = prediction_batch.to(device)

    # The detection head of some models produces one-hot outputs, while the detection
    # head of others produces single valued outputs. So, the ground-truth values for
    # the detection task should be transformed accordingly.
    if convert_detection_to_one_hot:
        prediction_ground_truth: torch.Tensor = functional.one_hot(prediction_batch,
                                                                   num_classes=2)
    else:
        prediction_ground_truth: torch.Tensor = prediction_batch.unsqueeze(dim=1)

    if convert_localization_to_one_hot:
        mask_batch = data_utils.convert_2d_map_to_binary_one_hot(mask_batch)

    mask_batch = mask_batch.float()
    prediction_ground_truth = prediction_ground_truth.float()

    with torch.cuda.amp.autocast(autocast):
        logging.debug(f"Starting forward pass | Rank: {rank} | Step: {step}")
        outputs: dict[str, torch.Tensor] = model(inputs_batch)
        output_masks: torch.Tensor = outputs["localization"]
        output_predictions: torch.Tensor = outputs["detection"]
        logging.debug(f"Forward pass completed | Rank: {rank} | Step: {step}")

        logging.debug(f"Starting loss computation | Rank: {rank} | Step: {step}")
        if isinstance(criterion, losses.LocalizationDetectionBootstrappedBCE):
            loss, p = criterion(output_masks, output_predictions,
                                mask_batch, prediction_ground_truth,
                                it=step)
            if rank == 0 and writer is not None:
                writer.add_scalar("train/loss_p", p, step)
        else:
            loss: torch.Tensor = criterion(output_masks, output_predictions,
                                           mask_batch, prediction_ground_truth)
        loss = loss / accumulation_steps
        logging.debug(f"Loss computation completed | Rank: {rank} | Step: {step}")

    return loss


def train_forward_pass_masked_attention(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    model: nn.Module,
    device: torch.device,
    criterion: nn.Module,
    step: int,
    convert_detection_to_one_hot: bool = True,
    autocast: bool = False,
    accumulation_steps: int = 1,
    writer: Optional[tensorboard.SummaryWriter] = None,
    convert_localization_to_one_hot: bool = False,
    rank: int = 0
) -> torch.Tensor:
    """Training forward pass for models that utilize attention masks."""
    inputs_batch, attention_mask_batch, mask_batch, prediction_batch, _ = batch

    inputs_batch = inputs_batch.to(device)
    attention_mask_batch = attention_mask_batch.to(device)
    mask_batch = mask_batch.to(device)
    prediction_batch = prediction_batch.to(device)

    # The detection head of some models produces one-hot outputs, while the detection
    # head of others produces single valued outputs. So, the ground-truth values for
    # the detection task should be transformed accordingly.
    if convert_detection_to_one_hot:
        prediction_ground_truth: torch.Tensor = functional.one_hot(prediction_batch,
                                                                   num_classes=2)
    else:
        prediction_ground_truth: torch.Tensor = prediction_batch.unsqueeze(dim=1)

    if convert_localization_to_one_hot:
        mask_batch = data_utils.convert_2d_map_to_binary_one_hot(mask_batch)

    attention_mask_batch = attention_mask_batch.bool()
    mask_batch = mask_batch.float()
    prediction_ground_truth = prediction_ground_truth.float()

    with torch.cuda.amp.autocast(autocast):
        logging.debug(f"Starting forward pass | Rank: {rank} | Step: {step}")
        outputs: dict[str, torch.Tensor] = model(inputs_batch, attention_mask_batch)
        output_masks: torch.Tensor = outputs["localization"]
        output_predictions: torch.Tensor = outputs["detection"]
        logging.debug(f"Forward pass completed | Rank: {rank} | Step: {step}")

        logging.debug(f"Starting loss computation | Rank: {rank} | Step: {step}")
        if isinstance(criterion, losses.LocalizationDetectionBootstrappedBCE):
            loss, p = criterion(output_masks, output_predictions,
                                mask_batch, prediction_ground_truth,
                                it=step)
            if rank == 0 and writer is not None:
                writer.add_scalar("train/loss_p", p, step)
        else:
            loss: torch.Tensor = criterion(output_masks, output_predictions,
                                           mask_batch, prediction_ground_truth)
        loss = loss / accumulation_steps
        logging.debug(f"Loss computation completed | Rank: {rank} | Step: {step}")

    return loss


def train_forward_pass_processor_masked_attention(
    batch: dict[str, Any],
    model: nn.Module,
    device: torch.device,
    criterion: nn.Module,
    step: int,
    processor: processors.OnlinePreprocessor,
    convert_detection_to_one_hot: bool = True,
    autocast: bool = False,
    accumulation_steps: int = 1,
    writer: Optional[tensorboard.SummaryWriter] = None,
    convert_localization_to_one_hot: bool = False,
    rank: int = 0,
    verbose_visualization: bool = False
) -> torch.Tensor:
    """Training forward pass with custom online processor."""
    # start_time: float = timeit.default_timer()
    # Preprocess the batch using the provided processor.
    with torch.no_grad():
        batch = processor.preprocess(batch)
    # stop_time: float = timeit.default_timer()
    # logging.info(f"Rank {rank}: Preprocessing time: {stop_time-start_time:.3f} secs")

    # start_time: float = timeit.default_timer()
    # Retrieve the model inputs/targets from the batch container.
    inputs_batch: torch.Tensor = batch["input"]
    attention_mask_batch: torch.Tensor = batch["attention_mask"]
    mask_batch: torch.Tensor = batch["mask"]
    prediction_batch: torch.Tensor = batch["manipulated"]

    if verbose_visualization:
        for i in range(inputs_batch.size(dim=0)):
            _visualize_rgb_npp_minibatch(inputs_batch[i], attention_mask_batch[i],
                                         mask_batch[i], prediction_batch[i])

    inputs_batch = inputs_batch.to(device)
    attention_mask_batch = attention_mask_batch.to(device)
    mask_batch = mask_batch.to(device)
    prediction_batch = prediction_batch.to(device)

    # The detection head of some models produces one-hot outputs, while the detection
    # head of others produces single valued outputs. So, the ground-truth values for
    # the detection task should be transformed accordingly.
    if convert_detection_to_one_hot:
        prediction_ground_truth: torch.Tensor = functional.one_hot(prediction_batch,
                                                                   num_classes=2)
    else:
        prediction_ground_truth: torch.Tensor = prediction_batch.unsqueeze(dim=1)

    if convert_localization_to_one_hot:
        mask_batch = data_utils.convert_2d_map_to_binary_one_hot(mask_batch)

    attention_mask_batch = attention_mask_batch.bool()
    mask_batch = mask_batch.float()
    prediction_ground_truth = prediction_ground_truth.float()

    with torch.cuda.amp.autocast(autocast):
        logging.debug(f"Starting forward pass | Rank: {rank} | Step: {step}")
        outputs: dict[str, torch.Tensor] = model(inputs_batch, attention_mask_batch)
        output_masks: torch.Tensor = outputs["localization"]
        output_predictions: torch.Tensor = outputs["detection"]
        logging.debug(f"Forward pass completed | Rank: {rank} | Step: {step}")

        logging.debug(f"Starting loss computation | Rank: {rank} | Step: {step}")
        if isinstance(criterion, losses.LocalizationDetectionBootstrappedBCE):
            loss, p = criterion(output_masks, output_predictions,
                                mask_batch, prediction_ground_truth,
                                it=step)
            if rank == 0 and writer is not None:
                writer.add_scalar("train/loss_p", p, step)
        else:
            loss: torch.Tensor = criterion(output_masks, output_predictions,
                                           mask_batch, prediction_ground_truth)
        loss = loss / accumulation_steps
        logging.debug(f"Loss computation completed | Rank: {rank} | Step: {step}")
    # stop_time: float = timeit.default_timer()
    # logging.info(f"Rank {rank}: Computation time: {stop_time-start_time:.3f} secs")

    return loss


def evaluation_pass(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    return_per_sample_metrics: bool = False,
    convert_detection_to_one_hot: bool = True,
    requires_sigmoid: bool = False,
    convert_localization_to_one_hot: bool = False,
    discard_mask_of_negative_predictions: bool = False
) -> tuple[torch.Tensor, dict[str, np.ndarray], Optional[int]]:
    inputs_batch, mask_batch, prediction_batch, indices = batch

    inputs_batch = inputs_batch.to(device)
    mask_batch = mask_batch.to(device)
    prediction_batch = prediction_batch.to(device)

    if convert_detection_to_one_hot:
        # Convert prediction labels to one-hot.
        prediction_ground_truth: torch.Tensor = functional.one_hot(prediction_batch,
                                                                   num_classes=2)
    else:
        prediction_ground_truth: torch.Tensor = prediction_batch.unsqueeze(dim=1)

    outputs: dict[str, torch.Tensor] = model(inputs_batch)
    output_masks: torch.Tensor = outputs["localization"]
    output_predictions: torch.Tensor = outputs["detection"]

    # Compute evaluation loss.
    loss_mask_batch: torch.Tensor = mask_batch
    if not output_masks.size() == loss_mask_batch.size():
        # Convert ground truth masks to the size of output masks to compute loss.
        loss_mask_batch = functional.interpolate(
            loss_mask_batch, (output_masks.size(dim=2), output_masks.size(dim=3))
        )
    if convert_localization_to_one_hot:
        loss_mask_batch = data_utils.convert_2d_map_to_binary_one_hot(mask_batch)

    if isinstance(criterion, losses.LocalizationDetectionBootstrappedBCE):
        loss, p = criterion(output_masks, output_predictions,
                            loss_mask_batch.float(), prediction_ground_truth.float())
    else:
        loss: torch.Tensor = criterion(output_masks, output_predictions,
                                       loss_mask_batch.float(), prediction_ground_truth.float())

    if convert_localization_to_one_hot:
        output_masks = torch.argmax(output_masks, dim=1, keepdim=True).float()

    # When resizing of the masks to the output size of the model has been disabled,
    # the output of the model will have a different size compared to the actual
    # ground-truth masks in the dataset. In that case, the output should be rescaled
    # to the size of the ground-truth mask. Disabling masks resizing in the data
    # generator is necessary in order to compare model's output against the original
    # image size and not against whatever the model may output.
    if not output_masks.size() == mask_batch.size():
        output_masks = functional.interpolate(
            output_masks, (mask_batch.size(dim=2), mask_batch.size(dim=3))
        )

    if requires_sigmoid:
        output_masks = torch.nn.functional.sigmoid(output_masks)
        output_predictions = torch.nn.functional.sigmoid(output_predictions)

    if convert_detection_to_one_hot:
        # Convert the predicted one hot vector to class number.
        argmax_output_detection = torch.argmax(output_predictions, dim=1)
    else:
        argmax_output_detection = torch.squeeze(output_predictions, dim=1)

    if discard_mask_of_negative_predictions:
        # Zero-out the predicted localization mask when the detector predicts the image sample
        # to be non-manipulated.
        output_masks = (output_masks *
                        argmax_output_detection.float().unsqueeze(1).unsqueeze(2).unsqueeze(3))

    loc_results = localization_metrics.update(output_masks.detach().cpu(),
                                              mask_batch.detach().cpu())
    det_results = detection_metrics.update(argmax_output_detection.detach().cpu(),
                                           prediction_batch.detach().cpu())
    results = _combine_localization_detection_results(loc_results, det_results)
    results['loss'] = loss.item()

    sample_index: Optional[int] = None
    if return_per_sample_metrics:
        indices_l: list[int] = indices.numpy().tolist()
        # Batch size should be 1, in order for metrics to correspond to 1 sample.
        assert (len(indices_l) == 1)
        sample_index = indices_l[0]

    return loss, results, sample_index


def evaluation_pass_masked_attention(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    return_per_sample_metrics: bool = False,
    convert_detection_to_one_hot: bool = True,
    requires_sigmoid: bool = False,
    convert_localization_to_one_hot: bool = False,
    discard_mask_of_negative_predictions: bool = False
) -> tuple[torch.Tensor, dict[str, np.ndarray], Optional[int]]:
    inputs_batch, attention_mask, mask_batch, prediction_batch, indices = batch

    inputs_batch = inputs_batch.to(device)
    attention_mask = attention_mask.to(device)
    mask_batch = mask_batch.to(device)
    prediction_batch = prediction_batch.to(device)

    if convert_detection_to_one_hot:
        # Convert prediction labels to one-hot.
        prediction_ground_truth: torch.Tensor = functional.one_hot(prediction_batch,
                                                                   num_classes=2)
    else:
        prediction_ground_truth: torch.Tensor = prediction_batch.unsqueeze(dim=1)

    attention_mask = attention_mask.bool()

    outputs: dict[str, torch.Tensor] = model(inputs_batch, attention_mask)
    output_masks: torch.Tensor = outputs["localization"]
    output_predictions: torch.Tensor = outputs["detection"]

    # Compute evaluation loss.
    loss_mask_batch: torch.Tensor = mask_batch
    if not output_masks.size() == loss_mask_batch.size():
        # Convert ground truth masks to the size of output masks to compute loss.
        loss_mask_batch = functional.interpolate(
            loss_mask_batch, (output_masks.size(dim=2), output_masks.size(dim=3))
        )
    if convert_localization_to_one_hot:
        loss_mask_batch = data_utils.convert_2d_map_to_binary_one_hot(mask_batch)
    if isinstance(criterion, losses.LocalizationDetectionBootstrappedBCE):
        loss, p = criterion(output_masks, output_predictions,
                            loss_mask_batch.float(), prediction_ground_truth.float())
    else:
        loss: torch.Tensor = criterion(output_masks, output_predictions,
                                       loss_mask_batch.float(), prediction_ground_truth.float())

    if convert_localization_to_one_hot:
        output_masks = torch.argmax(output_masks, dim=1, keepdim=True).float()

    # When resizing of the masks to the output size of the model has been disabled,
    # the output of the model will have a different size compared to the actual
    # ground-truth masks in the dataset. In that case, the output should be rescaled
    # to the size of the ground-truth mask. Disabling masks resizing in the data
    # generator is necessary in order to compare model's output against the original
    # image size and not against whatever the model may output.
    if not output_masks.size() == mask_batch.size():
        output_masks = functional.interpolate(
            output_masks, (mask_batch.size(dim=2), mask_batch.size(dim=3))
        )

    if requires_sigmoid:
        output_masks = torch.nn.functional.sigmoid(output_masks)
        output_predictions = torch.nn.functional.sigmoid(output_predictions)

    if convert_detection_to_one_hot:
        # Convert the predicted one hot vector to class number.
        argmax_output_detection = torch.argmax(output_predictions, dim=1)
    else:
        argmax_output_detection = torch.squeeze(output_predictions, dim=1)

    if discard_mask_of_negative_predictions:
        # Zero-out the predicted localization mask when the detector predicts the image sample
        # to be non-manipulated.
        output_masks = (output_masks *
                        argmax_output_detection.float().unsqueeze(1).unsqueeze(2).unsqueeze(3))

    loc_results = localization_metrics.update(output_masks.detach().cpu(),
                                              mask_batch.detach().cpu())
    det_results = detection_metrics.update(argmax_output_detection.detach().cpu(),
                                           prediction_batch.detach().cpu())
    results = _combine_localization_detection_results(loc_results, det_results)
    results['loss'] = loss.item()

    sample_index: Optional[int] = None
    if return_per_sample_metrics:
        indices_l: list[int] = indices.numpy().tolist()
        # Batch size should be 1, in order for metrics to correspond to 1 sample.
        assert (len(indices_l) == 1)
        sample_index = indices_l[0]

    return loss, results, sample_index


def evaluation_pass_processor_masked_attention(
    batch: dict[str, Any],
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    processor: processors.OnlinePreprocessor,
    return_per_sample_metrics: bool = False,
    convert_detection_to_one_hot: bool = True,
    requires_sigmoid: bool = False,
    convert_localization_to_one_hot: bool = False,
    discard_mask_of_negative_predictions: bool = False
) -> tuple[torch.Tensor, dict[str, np.ndarray], Optional[int]]:

    # Preprocess the batch using the provided processor.
    processor.to_device(device)
    batch = processor.preprocess(batch)

    # Retrieve the model inputs/targets from the batch container.
    inputs_batch: torch.Tensor = batch["input"]
    attention_mask: torch.Tensor = batch["attention_mask"]
    mask_batch: torch.Tensor = batch["mask"]
    prediction_batch: torch.Tensor = batch["manipulated"]
    indices: torch.Tensor = batch["index"]
    unpadded_size: tuple[torch.Tensor, torch.Tensor] = batch["unpadded_size"]

    inputs_batch = inputs_batch.to(device)
    attention_mask = attention_mask.to(device)
    mask_batch = mask_batch.to(device)
    prediction_batch = prediction_batch.to(device)

    if convert_detection_to_one_hot:
        # Convert prediction labels to one-hot.
        prediction_ground_truth: torch.Tensor = functional.one_hot(prediction_batch,
                                                                   num_classes=2)
    else:
        prediction_ground_truth: torch.Tensor = prediction_batch.unsqueeze(dim=1)

    attention_mask = attention_mask.bool()

    outputs: dict[str, torch.Tensor] = model(inputs_batch, attention_mask)
    output_masks: torch.Tensor = outputs["localization"]
    output_predictions: torch.Tensor = outputs["detection"]

    # Compute evaluation loss.
    loss_mask_batch: torch.Tensor = mask_batch
    if not output_masks.size() == loss_mask_batch.size():
        # Convert ground truth masks to the size of output masks to compute loss.
        loss_mask_batch = functional.interpolate(
            loss_mask_batch, (output_masks.size(dim=2), output_masks.size(dim=3))
        )
    if convert_localization_to_one_hot:
        loss_mask_batch = data_utils.convert_2d_map_to_binary_one_hot(mask_batch)
    if isinstance(criterion, losses.LocalizationDetectionBootstrappedBCE):
        loss, p = criterion(output_masks, output_predictions,
                            loss_mask_batch.float(), prediction_ground_truth.float())
    else:
        loss: torch.Tensor = criterion(output_masks, output_predictions,
                                       loss_mask_batch.float(), prediction_ground_truth.float())

    if convert_localization_to_one_hot:
        output_masks = torch.argmax(output_masks, dim=1, keepdim=True).float()

    # When resizing of the masks to the output size of the model has been disabled,
    # the output of the model will have a different size compared to the actual
    # ground-truth masks in the dataset. In that case, the output should be rescaled
    # to the size of the ground-truth mask. Disabling masks resizing in the data
    # generator is necessary in order to compare model's output against the original
    # image size and not against whatever the model may output.
    if not output_masks.size() == mask_batch.size():
        output_masks = functional.interpolate(
            output_masks, (mask_batch.size(dim=2), mask_batch.size(dim=3))
        )

    if requires_sigmoid:
        output_masks = torch.nn.functional.sigmoid(output_masks)
        output_predictions = torch.nn.functional.sigmoid(output_predictions)

    if convert_detection_to_one_hot:
        # Convert the predicted one hot vector to class number.
        argmax_output_detection = torch.argmax(output_predictions, dim=1)
    else:
        argmax_output_detection = torch.squeeze(output_predictions, dim=1)

    if discard_mask_of_negative_predictions:
        # Zero-out the predicted localization mask when the detector predicts the image sample
        # to be non-manipulated.
        output_masks = (output_masks *
                        argmax_output_detection.float().unsqueeze(1).unsqueeze(2).unsqueeze(3))

    output_masks = output_masks.detach().cpu()
    mask_batch = mask_batch.detach().cpu()
    is_padded: bool = False
    for i in range(output_masks.size(dim=0)):
        if (unpadded_size[0][i].item() != output_masks.size(dim=2)
                and unpadded_size[1][i].item() != output_masks.size(dim=3)):
            is_padded = True
            break
    # When some samples in the batch are padded, then remove the padding of each sample
    # and update the metrics for each sample separately.
    if is_padded:
        for i in range(output_masks.size(dim=0)):
            # TODO: Instead of keeping only the results of the last sample in the batch,
            #  compute the average. However, those partial results are currently used only for
            #  updating the progress bar and they make no actual different to the final results.
            unpadded_height: int = unpadded_size[0][i].item()
            unpadded_width: int = unpadded_size[1][i].item()
            loc_results = localization_metrics.update(
                output_masks[i, :, :unpadded_height, :unpadded_width],
                mask_batch[i, :, :unpadded_height, :unpadded_width]
            )
    else:
        loc_results = localization_metrics.update(output_masks, mask_batch)

    det_results = detection_metrics.update(argmax_output_detection.detach().cpu(),
                                           prediction_batch.detach().cpu())
    results = _combine_localization_detection_results(loc_results, det_results)
    results['loss'] = loss.item()

    sample_index: Optional[int] = None
    if return_per_sample_metrics:
        indices_l: list[int] = indices.numpy().tolist()
        # Batch size should be 1, in order for metrics to correspond to 1 sample.
        assert (len(indices_l) == 1)
        sample_index = indices_l[0]

    return loss, results, sample_index


def _combine_localization_detection_results(localization_results: dict[str, np.ndarray],
                                            detection_results: dict[str, np.ndarray]
                                            ) -> dict[str, np.ndarray]:
    results: dict[str, np.ndarray] = {}
    results.update({f"{k}_loc": v for k, v in localization_results.items()})
    results.update({f"{k}_det": v for k, v in detection_results.items()})
    return results


def _average_results(
    results: list[dict[str, Union[float, np.ndarray]]]
) -> dict[str, Union[float, np.ndarray]]:

    average_results: dict[str, Union[float, np.ndarray]] = {}
    for k in results[0].keys():
        if isinstance(results[0][k], np.ndarray):
            r: np.ndarray = np.stack([results[i][k] for i in range(len(results))])
            r = np.mean(r, axis=0)
            average_results[k] = r
        else:
            average_results[k] = np.mean([results[i][k] for i in range(len(results))])
    return average_results


def _write_results_to_writer(
    results: dict[str, Union[float, np.ndarray]],
    writer,
    prefix: str,
    epoch: int
):
    for metric in results:
        if np.size(results[metric]) > 1:
            for i in range(len(results[metric])):
                writer.add_scalar(
                    prefix + metric + '_' + str(i),
                    results[metric][i], epoch
                )
            writer.add_scalar(
                prefix + metric, np.mean(results[metric]), epoch
            )
        else:
            writer.add_scalar(prefix + metric, results[metric], epoch)


def _visualize_rgb_npp_minibatch(
    model_input: torch.Tensor,
    attention_mask: torch.Tensor,
    ground_truth_mask: torch.Tensor,
    ground_truth_score: torch.Tensor
) -> None:
    from omgfuser.models.attention_mask import model_ready_attention_mask_to_attention_region

    image: np.ndarray = model_input[0:3, :, :].detach().cpu().numpy()
    noiseprint: np.ndarray = model_input[3, :, :].detach().cpu().numpy()
    dct = None
    if model_input.size(dim=0) > 4:
        dct: np.ndarray = model_input[4, :, :].detach().cpu().numpy()

    # Invert the normalization with ImageNet mean and std.
    mean: np.ndarray = np.array((0.485, 0.456, 0.406)).reshape((3, 1, 1))
    std: np.ndarray = np.array((0.229, 0.224, 0.225)).reshape((3, 1, 1))
    image = ((image * std + mean) * 255).astype(np.uint8)
    image = image.transpose((1, 2, 0))

    noiseprint = (noiseprint * 256.0)

    fig = plt.figure(figsize=(20, 20))

    plots_num: int = 3 if dct is None else 4

    # Display image.
    fig.add_subplot(3, plots_num, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Image ({image.shape[0]}, {image.shape[1]})")

    # Display noiseprint.
    fig.add_subplot(3, plots_num, 2)
    plt.imshow(noiseprint, cmap="gray")
    plt.axis("off")
    plt.title(f"Noiseprint++ ({noiseprint.shape[0]}, {noiseprint.shape[1]})")

    # Display groundtruth.
    ground_truth_mask_arr: np.ndarray = ground_truth_mask.detach().cpu().numpy()
    ground_truth_mask_arr = (ground_truth_mask_arr[0] * 255).astype(np.uint8)
    fig.add_subplot(3, plots_num, 3)
    plt.imshow(ground_truth_mask_arr, cmap="gray")
    plt.axis("off")
    plt.title(f"Ground-Truth ({ground_truth_mask_arr.shape[0]}, {ground_truth_mask_arr.shape[1]})")

    # Display DCT
    if dct is not None:
        fig.add_subplot(3, plots_num, 4)
        plt.imshow(dct, cmap="gray")
        plt.axis("off")
        plt.title(f"DCT ({dct.shape[0]}, {dct.shape[1]})")

    # Display some attention regions.
    instance_regions: list[np.ndarray] = model_ready_attention_mask_to_attention_region(
        attention_mask
    )
    for i in range(plots_num*2):
        if i < len(instance_regions):
            inst_reg: np.ndarray = instance_regions[i]
            fig.add_subplot(3, plots_num, plots_num+1+i)
            plt.imshow(inst_reg, cmap="gray")
            plt.axis("off")
            plt.title(
                f"Attention Region {i} ({inst_reg.shape[0]}, {inst_reg.shape[1]})")

    # Display ground-truth score
    is_manipulated: bool = bool(ground_truth_score.detach().cpu().item())
    fig.suptitle(f"Manipulated: {'TRUE' if is_manipulated else 'FALSE'}")

    fig.show()


def log_localization_detection_results(
    loc_results: dict[str, np.ndarray],
    det_results: dict[str, np.ndarray],
    signal_name: str
) -> None:
    with np.printoptions(suppress=True, precision=2):
        for task_results, task in zip([loc_results, det_results], ["Loc", "Det"]):
            for metric_name, metric_value in task_results.items():
                if metric_value.size == 1:
                    logging.info(
                        f"{signal_name} | {task} | {metric_name}: {metric_value.item() * 100:.2f}"
                    )
                else:
                    logging.info(
                        f"{signal_name} | {task} | {metric_name}: {metric_value * 100}"
                    )