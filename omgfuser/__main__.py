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
import logging
import pathlib
from math import ceil
from typing import Optional, Union

import cv2
import click
import torch
import torch.distributed as dist
import torch.multiprocessing
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from warmup_scheduler_pytorch import WarmUpScheduler

from .models import fusion
from .data import processors
from . import losses
from . import datasets
from . import training
from . import utils


__version__: str = "1.0.0"
__revision__: int = 1
__author__: str = "Dimitrios S. Karageorgiou"
__email__: str = "dkarageo@iti.gr"


AVAILABLE_LOSSES: list[str] = [
    "localization_bce",
    "localization_detection_bce",
    "bootstrapped_localization_bce",
    "bootstrapped_localization_bce_5_40_0.25",
    "class_aware_localization_detection_bce",
    "class_aware_localization_only_bce",
    "class_aware_localization_only_bce_with_logits",
    "localization_bce_dice",
    "localization_detection_bce_dice",
    "class_aware_localization_only_bce_dice",
    "class_aware_localization_detection_bce_dice",
    "class_aware_localization_detection_bce_dice_more_manipulated"
]

AVAILABLE_MODELS: list[str] = [
    "masked_attention_positional_fusion_double_conv_upscaler_transformer_single_mlp_detector_dinov2frozen_feat_int_drop_stream_drop_path_5_inputs",
    "masked_attention_positional_fusion_double_conv_upscaler_transformer_single_mlp_detector_dinov2patchembedfrozen_feat_int_bilinear_drop_stream_drop_path_2_inputs_448",
]

AVAILABLE_SCHEDULERS: list[str] = [
    "cosine_annealing_warm_restarts",
    "cosine_annealing_with_warmup"
]

AVAILABLE_OPTIMIZERS: list[str] = [
    "adamw",
    "sgd_momentum"
]

logging.getLogger().setLevel(logging.INFO)
# Limit the number of OpenCV threads to 2 to utilize multiple processes. Otherwise,
# each process spawns a number of threads equal to the number of logical cores and
# the overall performance gets worse due to threads congestion.
cv2.setNumThreads(2)
torch.multiprocessing.set_sharing_strategy('file_system')


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option('--experiment_name', type=str, default="", show_default=True)
@click.option('--resume', is_flag=True, show_default=True)
@click.option('--resume_epoch', type=int, default=None)
@click.option('--batch_size', required=True, type=int, default=16, show_default=True)
@click.option('--accumulation_steps', type=int, default=1, show_default=True)
@click.option('--gpu_id', type=str)
@click.option('--learning_rate', type=float, default=1e-3, show_default=True)
@click.option('--warmup_epochs', type=int, default=4, show_default=True)
@click.option('--min_lr', type=float, default=1e-6, show_default=True)
@click.option('--epochs', type=int, default=20, show_default=True)
@click.option('--optim', default="adamw",
              type=click.Choice(AVAILABLE_OPTIMIZERS, case_sensitive=False))
@click.option('--save_all', is_flag=True)
@click.option('--checkpoint_path',
              type=click.Path(file_okay=False, path_type=pathlib.Path),
              default='checkpoints/',
              show_default=True)
@click.option('--checkpoint_name', type=str)
@click.option('--dataset_csv',
              type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
              required=True,
              show_default=True)
@click.option('--eval_dataset_csv',
              type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path), multiple=True)
@click.option('--eval_dataset_name', type=str, multiple=True)
@click.option('--eval_batch_size', type=int, multiple=True)
@click.option('--lmdb_storage',
              type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path))
@click.option('--dataset_root',
              type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path))
@click.option('--data_workers', type=int, default=8)
@click.option('--model_name',
              type=click.Choice(AVAILABLE_MODELS, case_sensitive=False),
              default="fusion_model")
@click.option('--input_signals', required=True, type=str)
@click.option('--signals_channels', required=True, type=str)
@click.option('--first_stage_layers', type=int, default=6)
@click.option('--second_stage_layers', type=int, default=6)
@click.option('--loss_function',
              type=click.Choice(AVAILABLE_LOSSES, case_sensitive=False),
              default="localization_detection_bce")
@click.option('--lr_scheduler_name',
              type=click.Choice(AVAILABLE_SCHEDULERS, case_sensitive=False),
              default=None)
@click.option('--single_feature_extractors', is_flag=True)
@click.option('--autocast', is_flag=True)
@click.option('--distributed', is_flag=True)
@click.option('--stratify', is_flag=True)
@click.option('--max_steps_per_epoch', type=int)
@click.option('--keep_aspect_ratio', is_flag=True)
@click.option('--model_expansion', is_flag=True)
@click.option('--expansion_finetune', is_flag=True)
def train(
    experiment_name: str,
    resume: bool,
    resume_epoch: Optional[int],
    batch_size: int,
    accumulation_steps: int,
    gpu_id: Optional[str],
    learning_rate: float,
    warmup_epochs: int,
    min_lr: float,
    epochs: int,
    optim: str,
    save_all: bool,
    checkpoint_path: pathlib.Path,
    checkpoint_name: str,
    dataset_csv: pathlib.Path,
    eval_dataset_csv: list[pathlib.Path],
    eval_dataset_name: list[str],
    eval_batch_size: list[int],
    lmdb_storage: Optional[pathlib.Path],
    dataset_root: Optional[pathlib.Path],
    data_workers: int,
    model_name: str,
    input_signals: str,
    signals_channels: str,
    first_stage_layers: int,
    second_stage_layers: int,
    loss_function: str,
    lr_scheduler_name: Optional[str],
    single_feature_extractors: bool,
    autocast: bool,
    distributed: bool,
    stratify: bool,
    max_steps_per_epoch: Optional[int],
    keep_aspect_ratio: bool,
    model_expansion: bool,
    expansion_finetune: bool,
) -> None:
    if distributed:
        dist.init_process_group("nccl")
        rank: int = dist.get_rank()
        logging.info(f"Starting training worker: {rank}")
    else:
        rank: int = 0

    writer: Optional[SummaryWriter] = None
    if rank == 0:
        kwargs = locals()
        print(kwargs)
        writer = SummaryWriter('runs/train/' + experiment_name)

    # Select CPU or GPU device.
    if distributed:
        gpu_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    elif gpu_id is not None:
        device = torch.device('cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # Create the model
    model: fusion.FusionModel
    requires_one_hot: bool
    requires_one_hot_loc: bool
    requires_sigmoid: bool
    requires_imagenet_image_normalization: bool
    train_type: training.ForwardPassType
    image_size: tuple[int, int]
    (model, requires_one_hot, requires_one_hot_loc, requires_sigmoid,
     requires_imagenet_image_normalization, train_type, image_size) = _create_model(
        model_name=model_name,
        first_stage_layers=first_stage_layers,
        second_stage_layers=second_stage_layers,
        single_feature_extractors=single_feature_extractors
    )

    # Create train-val-test data generators.
    if not dataset_root:
        dataset_root = dataset_csv.absolute().parent
    input_signals_separate: list[str] = input_signals.split(",")
    signals_channels: list[int] = [int(c) for c in signals_channels.split(",")]
    eval_datasets: dict[str, pathlib.Path] = {
        k: v for k, v in zip(eval_dataset_name, eval_dataset_csv)
    }
    eval_batch_sizes: dict[str, int] = {
        k: v for k, v in zip(eval_dataset_name, eval_batch_size)
    }
    train_data, val_data, test_data = _create_data_generators(
        dataset_csv,
        eval_datasets,
        dataset_root,
        input_signals_separate,
        signals_channels,
        train_type,
        requires_imagenet_image_normalization,
        stratify,
        keep_aspect_ratio,
        lmdb_storage=lmdb_storage,
        target_image_size=image_size
    )
    train_loader, val_loader, test_loader = _create_data_loaders(
        train_data, val_data, test_data, batch_size, eval_batch_sizes, data_workers,
        distributed=distributed
    )
    train_processor, val_processor, test_processor = _create_data_processors(
        train_data,
        # TODO: Support multiple datasets with different processors.
        val_data[eval_dataset_name[0]] if isinstance(val_data, dict) else val_data,
        test_data
    )

    # Checkpoint path is characterized by the experiment name.
    experiment_checkpoints_path = checkpoint_path / experiment_name
    experiment_checkpoints_path.mkdir(exist_ok=True, parents=True)

    if model_expansion:
        # Expand a pretrained model with an additional stream.
        checkpoint_file: pathlib.Path = experiment_checkpoints_path / checkpoint_name
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        state_dict = fix_ddp_state_dict(checkpoint['model'])
        model.extend_pretrained_model(state_dict, expansion_finetune)

    steps_per_epoch: int = ceil(len(train_loader) / accumulation_steps)
    if max_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps_per_epoch)
    criterion = _create_loss(loss_function,
                             steps_per_epoch=steps_per_epoch)
    params = [p for p in model.parameters() if p.requires_grad]
    if optim == "adamw":
        optimizer = torch.optim.AdamW(params, lr=learning_rate)
    elif optim == "sgd_momentum":
        optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9)
    else:
        raise RuntimeError(f"Non-implemented optimizer: {optim}")
    scheduler, scheduler_step_per_epoch = _create_scheduler(
        optimizer, lr_scheduler_name, epochs, steps_per_epoch,
        warmup_epochs=warmup_epochs, min_lr=min_lr
    )

    initial_epoch: int = 1
    if resume:
        # Load previous checkpoint.
        checkpoint_file: pathlib.Path = experiment_checkpoints_path / checkpoint_name
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        model.load_state_dict(fix_ddp_state_dict(checkpoint['model']))
        model.to(device)  # Transfer the model to GPU to re-initialize optimizer in GPU!
        if resume_epoch is None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:  # Older checkpoints did not contain scheduler.
                scheduler.load_state_dict(checkpoint['scheduler'])
            initial_epoch = checkpoint["epoch"] + 1  # Resume after the last saved epoch.
        else:
            initial_epoch = resume_epoch
        if rank == 0:
            logging.info(f"RESUMING FROM: {checkpoint_file}")
            logging.info(f"STARTING AT EPOCH: {initial_epoch}")

    if distributed:
        # Create a DDP instance of the model.
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(device)
        model: DistributedDataParallel = DistributedDataParallel(
            model, device_ids=[device.index], output_device=device, find_unused_parameters=True
        )

    if rank == 0:
        logging.info(f"EFFECTIVE BATCH SIZE: {batch_size * accumulation_steps}")

    training.train_model(model=model,
                         epochs=epochs,
                         steps_per_epoch=steps_per_epoch,
                         device=device,
                         train_loader=train_loader,
                         val_loader=val_loader,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         scheduler_step_per_epoch=scheduler_step_per_epoch,
                         criterion=criterion,
                         checkpoint_path=experiment_checkpoints_path,
                         writer=writer,
                         rank=rank,
                         save_all=save_all,
                         initial_epoch=initial_epoch,
                         step=(initial_epoch-1)*steps_per_epoch,
                         convert_detection_to_one_hot=requires_one_hot,
                         convert_localization_to_one_hot=requires_one_hot_loc,
                         requires_sigmoid=requires_sigmoid,
                         autocast=autocast,
                         accumulation_steps=accumulation_steps,
                         forward_pass_type=train_type,
                         train_processor=train_processor,
                         val_processor=val_processor)


@cli.command()
@click.option('--experiment_name', type=str, default="", show_default=True)
# @click.option('--batch_size', required=True, type=int, default=16, show_default=True)
@click.option('--gpu_id', type=str)
@click.option('--checkpoint_path',
              type=click.Path(exists=True, path_type=pathlib.Path),
              default='checkpoints/',
              show_default=True)
@click.option('--dataset_csv',
              type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
              required=True,
              multiple=True,
              show_default=True)
@click.option('--dataset_name', type=str, multiple=True)
@click.option('--lmdb_storage',
              type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path))
@click.option('--dataset_root',
              type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path))
@click.option('--data_split',
              type=click.Choice(["train", "eval", "test"], case_sensitive=False),
              default="test")
@click.option('--resize_mask', is_flag=True)
@click.option('--data_workers', type=int, default=8)
@click.option('--model_name',
              type=click.Choice(AVAILABLE_MODELS, case_sensitive=False),
              default="fusion_model")
@click.option('--input_signals', required=True, type=str)
@click.option('--signals_channels', required=True, type=str)
@click.option('--first_stage_layers', type=int, default=6)
@click.option('--second_stage_layers', type=int, default=6)
@click.option('--loss_function',
              type=click.Choice(AVAILABLE_LOSSES, case_sensitive=False),
              default="localization_detection_bce")
@click.option('--single_feature_extractors', is_flag=True)
@click.option('--export_best_samples', is_flag=True)
@click.option('--discard_mask_of_negative_predictions', is_flag=True)
@click.option('--keep_aspect_ratio', is_flag=True)
def test(
    experiment_name: str,
    # batch_size: int,
    gpu_id: Optional[str],
    checkpoint_path: pathlib.Path,
    dataset_csv: list[pathlib.Path],
    dataset_name: list[str],
    lmdb_storage: Optional[pathlib.Path],
    dataset_root: Optional[pathlib.Path],
    data_split: str,
    resize_mask: bool,
    data_workers: int,
    model_name: str,
    input_signals: str,
    signals_channels: str,
    first_stage_layers: int,
    second_stage_layers: int,
    loss_function: str,
    single_feature_extractors: bool,
    export_best_samples: bool,
    discard_mask_of_negative_predictions: bool,
    keep_aspect_ratio: bool
) -> None:
    kwargs = locals()
    print(kwargs)

    run_dir: pathlib.Path = pathlib.Path("runs/tests") / experiment_name
    writer: SummaryWriter = SummaryWriter(str(run_dir))

    # Select CPU or GPU device.
    if gpu_id is not None:
        device = torch.device('cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # Create the model
    model: fusion.FusionModel
    requires_one_hot: bool
    requires_one_hot_loc: bool
    requires_sigmoid: bool
    requires_imagenet_image_normalization: bool
    train_type: training.ForwardPassType
    image_size: tuple[int, int]
    (model, requires_one_hot, requires_one_hot_loc, requires_sigmoid,
     requires_imagenet_image_normalization, train_type, image_size) = _create_model(
        model_name=model_name,
        first_stage_layers=first_stage_layers,
        second_stage_layers=second_stage_layers,
        single_feature_extractors=single_feature_extractors
    )

    # Create data loader.
    input_signals_separate: list[str] = input_signals.split(",")
    signals_channels: list[int] = [int(c) for c in signals_channels.split(",")]
    if len(dataset_csv) > 1:
        csv_per_dataset: dict[str, pathlib.Path] = {
            k: v for k, v in zip(dataset_name, dataset_csv)
        }
    else:
        csv_per_dataset: dict[str, pathlib] = {
            "" if len(dataset_name) == 0 else dataset_name[0]: dataset_csv[0]
        }
    data_loaders: dict[str, DataLoader] = {}
    for d_name, d_csv in csv_per_dataset.items():
        if not dataset_root:
            d_root = d_csv.absolute().parent
        else:
            d_root = dataset_root
        data: Union[datasets.ForensicsDataset,
                    datasets.HandcraftedForensicsSignalsDataset] = _create_data_generator_for_split(
            d_csv,
            d_root,
            input_signals_separate,
            signals_channels,
            split=datasets.Split(data_split),
            resize_mask=resize_mask,
            train_type=train_type,
            requires_imagenet_image_normalization=requires_imagenet_image_normalization,
            lmdb_storage=lmdb_storage,
            keep_aspect_ratio=keep_aspect_ratio,
            target_image_size=image_size
        )
        collate_fn = None
        if isinstance(data, datasets.HandcraftedForensicsSignalsDataset):
            collate_fn = data.build_collate_fn()
        data_loaders[d_name] = DataLoader(
            dataset=data,
            shuffle=True,
            batch_size=1,
            num_workers=data_workers,
            pin_memory=False,
            collate_fn=collate_fn
        )

    # TODO: Support multiple datasets with different processors.
    data_processor: Optional[processors.OnlinePreprocessor] = None
    if isinstance(data, datasets.HandcraftedForensicsSignalsDataset):
        data_processor = data.get_data_processor()

    # Create loss function.
    criterion = _create_loss(loss_function, steps_per_epoch=len(data_loaders[d_name]))

    # Load all the weight checkpoints that should be evaluated.
    checkpoint_files: list[pathlib.Path] = []
    if checkpoint_path.is_dir():
        # Load all the checkpoints in the directory (for evaluating all the trained epochs).
        checkpoint_files.extend([f for f in checkpoint_path.iterdir() if f.suffix == ".pth"])
        checkpoint_files.sort(key=utils.natural_keys)
    else:
        # Load just the specific checkpoint file.
        checkpoint_files.append(checkpoint_path)

    for ckpt_file in checkpoint_files:
        # Load previous checkpoint.
        checkpoint = torch.load(ckpt_file, map_location='cpu')
        model.load_state_dict(fix_ddp_state_dict(checkpoint['model']))
        # Evaluate the model.
        results = training.evaluate_model(
            model, criterion, data_loaders, device, checkpoint["epoch"], writer,
            return_per_sample_metrics=export_best_samples,
            convert_detection_to_one_hot=requires_one_hot,
            convert_localization_to_one_hot=requires_one_hot_loc,
            requires_sigmoid=requires_sigmoid,
            forward_pass_type=train_type,
            discard_mask_of_negative_predictions=discard_mask_of_negative_predictions,
            processor=data_processor
        )
        if export_best_samples:
            utils.export_best_samples(
                model=model,
                device=device,
                data=data,
                inputs_names=input_signals_separate,
                channels=signals_channels,
                results=results,
                output_dir=run_dir/"samples",
                requires_sigmoid=requires_sigmoid,
                forward_pass_type=train_type,
                processor=data_processor
            )


def _create_model(
    model_name: str,
    first_stage_layers: int,
    second_stage_layers: int,
    single_feature_extractors: bool
) -> tuple[nn.Module, bool, bool, bool, bool, training.ForwardPassType, tuple[int, int]]:
    logging.info(f"MODEL: {model_name}")
    logging.info(f"FIRST STAGE LAYERS: {first_stage_layers}")
    logging.info(f"SECOND STAGE LAYERS: {second_stage_layers}")

    if model_name == "masked_attention_positional_fusion_double_conv_upscaler_transformer_single_mlp_detector_dinov2frozen_feat_int_drop_stream_drop_path_5_inputs":
        model: nn.Module = fusion.MaskedAttentionFusionModelWithImage(
            inputs_num=5,
            first_stage_depth=first_stage_layers,
            first_stage_heads=12,
            second_stage_depth=second_stage_layers,
            second_stage_heads=12,
            height=224,
            width=224,
            mask_channels=1,
            dropout=.1,
            drop_stream_probability=0.2,
            drop_path_probability=0.1,
            upscaler_type=fusion.UpscalerType.DOUBLE_CONV_UPSCALER,
            second_stage_fusion_type=fusion.FusionType.POSITIONAL_FUSION,
            detection_head_type=fusion.DetectionHeadType.TRANSFORMER_ONE_LAYER_MLP_SINGLE_OUT_SIGMOID,
            image_backbone_type=fusion.BackboneType.DINOv2_FROZEN_FEATURE_INTERPOLATION,
            pass_dino_features_through_first_stage=False
        )
        requires_one_hot: bool = False
        requires_one_hot_localization: bool = False
        requires_sigmoid: bool = False
        requires_imagenet_image_normalization: bool = True
        training_type: training.ForwardPassType = training.ForwardPassType.ATTENTION_MASK_PASS
        image_size: tuple[int, int] = (224, 224)
    elif model_name == "masked_attention_positional_fusion_double_conv_upscaler_transformer_single_mlp_detector_dinov2patchembedfrozen_feat_int_bilinear_drop_stream_drop_path_2_inputs_448":
        model: nn.Module = fusion.MaskedAttentionFusionModelWithImage(
            inputs_num=2,
            first_stage_depth=first_stage_layers,
            first_stage_heads=12,
            second_stage_depth=second_stage_layers,
            second_stage_heads=12,
            height=448,
            width=448,
            mask_channels=1,
            dropout=.1,
            drop_stream_probability=0.2,
            drop_path_probability=0.1,
            upscaler_type=fusion.UpscalerType.DOUBLE_CONV_UPSCALER,
            second_stage_fusion_type=fusion.FusionType.POSITIONAL_FUSION,
            detection_head_type=fusion.DetectionHeadType.TRANSFORMER_ONE_LAYER_MLP_SINGLE_OUT_SIGMOID,
            image_backbone_type=fusion.BackboneType.DINOv2_PATCH_EMBED_FROZEN_FEATURE_INTERPOLATION,
            pass_dino_features_through_first_stage=False
        )
        requires_one_hot: bool = False
        requires_one_hot_localization: bool = False
        requires_sigmoid: bool = False
        requires_imagenet_image_normalization: bool = True
        training_type: training.ForwardPassType = training.ForwardPassType.PREPROCESSOR_ATTENTION_MASK_PASS
        image_size: tuple[int, int] = (448, 448)
    else:
        raise RuntimeError(f"{model_name} is not a valid model name.")

    return (
        model,
        requires_one_hot,
        requires_one_hot_localization,
        requires_sigmoid,
        requires_imagenet_image_normalization,
        training_type,
        image_size
    )


def _create_data_generators(
    dataset_csv: pathlib.Path,
    eval_datasets: dict[str, pathlib.Path],
    dataset_root: pathlib.Path,
    input_signals: list[str],
    signals_channels: list[int],
    train_type: training.ForwardPassType,
    requires_imagenet_image_normalization: bool,
    stratify: bool,
    keep_aspect_ratio: bool,
    lmdb_storage: Optional[pathlib.Path] = None,
    target_image_size: tuple[int, int] = (224, 224)
) -> tuple[
    Union[datasets.ForensicsDataset, datasets.HandcraftedForensicsSignalsDataset],
    Union[datasets.ForensicsDataset,
          datasets.HandcraftedForensicsSignalsDataset,
          dict[str, Union[datasets.ForensicsDataset, datasets.HandcraftedForensicsSignalsDataset]]],
    Union[datasets.ForensicsDataset, datasets.HandcraftedForensicsSignalsDataset]
]:
    """Creates train, validation and test data generators."""

    train_data: datasets.ForensicsDataset = _create_data_generator_for_split(
        dataset_csv, dataset_root, input_signals, signals_channels, datasets.Split.TRAIN_SPLIT,
        train_type, requires_imagenet_image_normalization=requires_imagenet_image_normalization,
        lmdb_storage=lmdb_storage,
        stratify=stratify,
        keep_aspect_ratio=keep_aspect_ratio,
        target_image_size=target_image_size
    )
    test_data: datasets.ForensicsDataset = _create_data_generator_for_split(
        dataset_csv, dataset_root, input_signals, signals_channels, datasets.Split.TEST_SPLIT,
        train_type, requires_imagenet_image_normalization=requires_imagenet_image_normalization,
        lmdb_storage=lmdb_storage,
        keep_aspect_ratio=keep_aspect_ratio,
        target_image_size=target_image_size
    )

    # Validation data can either be included into the training data CSV, or be provided into
    # separate CSVs, one for each validation dataset.
    if len(eval_datasets) > 0:
        val_data: dict[str, Union[datasets.ForensicsDataset,
                                  datasets.HandcraftedForensicsSignalsDataset]] = {}
        for d_name, d_csv in eval_datasets.items():
            val_data[d_name] = _create_data_generator_for_split(
                d_csv,
                dataset_root,
                input_signals,
                signals_channels,
                datasets.Split.EVAL_SPLIT,
                train_type,
                requires_imagenet_image_normalization=requires_imagenet_image_normalization,
                lmdb_storage=lmdb_storage,
                keep_aspect_ratio=keep_aspect_ratio,
                target_image_size=target_image_size
            )
    else:
        val_data: datasets.ForensicsDataset = _create_data_generator_for_split(
            dataset_csv, dataset_root, input_signals, signals_channels, datasets.Split.EVAL_SPLIT,
            train_type, requires_imagenet_image_normalization=requires_imagenet_image_normalization,
            lmdb_storage=lmdb_storage,
            keep_aspect_ratio=keep_aspect_ratio,
            target_image_size=target_image_size
        )

    return train_data, val_data, test_data


def _create_data_generator_for_split(
    dataset_csv: pathlib.Path,
    dataset_root: pathlib.Path,
    input_signals: list[str],
    signals_channels: list[int],
    split: datasets.Split,
    train_type: training.ForwardPassType,
    target_image_size: tuple[int, int] = (224, 224),
    resize_mask: bool = True,
    requires_imagenet_image_normalization: bool = False,
    lmdb_storage: Optional[pathlib.Path] = None,
    stratify: bool = False,
    keep_aspect_ratio: bool = False
) -> Union[datasets.ForensicsDataset, datasets.HandcraftedForensicsSignalsDataset]:
    if resize_mask:
        logging.info(f"EVALUATION ON MODEL OUTPUT SIZE: {target_image_size}")
    else:
        logging.info(f"EVALUATION ON ACTUAL MASKS SIZE")

    if train_type == training.ForwardPassType.SIMPLE_FORWARD_PASS:
        data: datasets.ForensicsDataset = datasets.ForensicsDataset(
            csv_file=dataset_csv,
            root_dir=dataset_root,
            signals_columns=input_signals,
            signals_channels=signals_channels,
            split=split,
            target_image_size=target_image_size,
            resize_mask=resize_mask,
            imagenet_image_normalization=requires_imagenet_image_normalization,
            lmdb_file_storage_path=lmdb_storage,
            stratify=stratify
        )
    elif train_type == training.ForwardPassType.ATTENTION_MASK_PASS:
        data = datasets.ForensicsDatasetWithAttentionMask(
            csv_file=dataset_csv,
            root_dir=dataset_root,
            signals_columns=input_signals,
            signals_channels=signals_channels,
            split=split,
            target_image_size=target_image_size,
            resize_mask=resize_mask,
            imagenet_image_normalization=requires_imagenet_image_normalization,
            lmdb_file_storage_path=lmdb_storage,
            stratify=stratify
        )
    elif train_type == training.ForwardPassType.PREPROCESSOR_ATTENTION_MASK_PASS:
        data: datasets.HandcraftedForensicsSignalsDataset = \
            datasets.HandcraftedForensicsSignalsDataset(
                csv_file=dataset_csv,
                root_dir=dataset_root,
                signals_columns=input_signals,
                signals_channels=signals_channels,
                split=split,
                target_image_size=target_image_size,
                resize_mask=resize_mask,
                imagenet_image_normalization=requires_imagenet_image_normalization,
                lmdb_file_storage_path=lmdb_storage,
                stratify=stratify,
                keep_aspect_ratio=keep_aspect_ratio
            )
    else:
        raise RuntimeError(f"Invalid Dataset Type: {train_type.value}")

    return data


def _create_data_loaders(
    train_data: Union[datasets.ForensicsDataset, datasets.HandcraftedForensicsSignalsDataset],
    val_data: Union[datasets.ForensicsDataset,
                    datasets.HandcraftedForensicsSignalsDataset,
                    dict[str, Union[datasets.ForensicsDataset,
                                    datasets.HandcraftedForensicsSignalsDataset]]],
    test_data: Union[datasets.ForensicsDataset, datasets.HandcraftedForensicsSignalsDataset],
    batch_size: int,
    val_batch_sizes: dict[str, int],
    workers: int = 8,
    distributed: bool = False
) -> tuple[
    DataLoader,
    Union[DataLoader, dict[str, DataLoader]],
    DataLoader
]:
    # When distributed training is enabled, create a sampler to split data across processes.
    train_data_sampler: Optional[DistributedSampler] = None
    if distributed:
        train_data_sampler = DistributedSampler(
            train_data,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank()
        )
        if dist.get_rank() == 0:
            logging.info("USING DISTRIBUTED DATA SAMPLER")

    # When using HandcraftedForensicsSignalDataset the collate function is provided by the
    # dataset.
    train_collate_fn = None
    test_collate_fn = None
    if isinstance(train_data, datasets.HandcraftedForensicsSignalsDataset):
        train_collate_fn = train_data.build_collate_fn()
    if isinstance(test_data, datasets.HandcraftedForensicsSignalsDataset):
        test_collate_fn = test_data.build_collate_fn()

    train_loader = DataLoader(dataset=train_data,
                              shuffle=train_data_sampler is None,
                              batch_size=batch_size,
                              num_workers=workers,
                              pin_memory=False,
                              sampler=train_data_sampler,
                              collate_fn=train_collate_fn)
    test_loader = DataLoader(dataset=test_data,
                             shuffle=False,
                             batch_size=batch_size,
                             num_workers=workers,
                             pin_memory=False,
                             collate_fn=test_collate_fn)

    if isinstance(val_data, dict):
        val_loader: dict[str, DataLoader] = {}
        for d_name, d in val_data.items():
            val_collate_fn = None
            if isinstance(d, datasets.HandcraftedForensicsSignalsDataset):
                val_collate_fn = d.build_collate_fn()
            val_loader[d_name] = DataLoader(dataset=d,
                                            shuffle=False,
                                            batch_size=val_batch_sizes.get(d_name, batch_size),
                                            num_workers=workers,
                                            pin_memory=False,
                                            collate_fn=val_collate_fn)
    else:
        val_collate_fn = None
        if isinstance(val_data, datasets.HandcraftedForensicsSignalsDataset):
            val_collate_fn = val_data.build_collate_fn()
        val_loader: DataLoader = DataLoader(dataset=val_data,
                                            shuffle=False,
                                            batch_size=batch_size,
                                            num_workers=workers,
                                            pin_memory=False,
                                            collate_fn=val_collate_fn)

    return train_loader, val_loader, test_loader


def _create_data_processors(
    train_data: Union[datasets.ForensicsDataset,
                      datasets.HandcraftedForensicsSignalsDataset],
    val_data: Union[datasets.ForensicsDataset,
                    datasets.HandcraftedForensicsSignalsDataset],
    test_data: Union[datasets.ForensicsDataset,
                     datasets.HandcraftedForensicsSignalsDataset]
) -> tuple[Optional[processors.OnlinePreprocessor],
           Optional[processors.OnlinePreprocessor],
           Optional[processors.OnlinePreprocessor]]:

    train_processor: Optional[processors.OnlinePreprocessor] = None
    val_processor: Optional[processors.OnlinePreprocessor] = None
    test_processor: Optional[processors.OnlinePreprocessor] = None

    if isinstance(train_data, datasets.HandcraftedForensicsSignalsDataset):
        train_processor = train_data.get_data_processor()
    if isinstance(val_data, datasets.HandcraftedForensicsSignalsDataset):
        val_processor = val_data.get_data_processor()
    if isinstance(test_data, datasets.HandcraftedForensicsSignalsDataset):
        test_processor = test_data.get_data_processor()

    return train_processor, val_processor, test_processor


def _create_loss(loss_function: str,
                 steps_per_epoch: int) -> nn.Module:
    logging.info(f"LOSS FUNCTION: {loss_function}")
    if loss_function == "localization_detection_bce":
        return losses.LocalizationDetectionBCELoss()
    elif loss_function == "localization_bce":
        return losses.LocalizationDetectionBCELoss(
            detection_loss_weight=.0,
            localization_loss_weight=1.0
        )
    elif loss_function == "bootstrapped_localization_bce":
        return losses.LocalizationDetectionBootstrappedBCE(
            start_warm=15*steps_per_epoch,
            end_warm=25*steps_per_epoch,
            top_p=0.15,
            detection_loss_weight=.0,
            localization_loss_weight=1.0
        )
    elif loss_function == "bootstrapped_localization_bce_5_40_0.25":
        return losses.LocalizationDetectionBootstrappedBCE(
            start_warm=5 * steps_per_epoch,
            end_warm=40 * steps_per_epoch,
            top_p=0.25,
            detection_loss_weight=.0,
            localization_loss_weight=1.0
        )
    elif loss_function == "localization_detection_bce_dice":
        return losses.LocalizationDetectionBCEDiceLoss(
            detection_loss_weight=.25,
            localization_bce_loss_weight=.3,
            dice_loss_weight=.45
        )
    elif loss_function == "class_aware_localization_detection_bce":
        return losses.ClassAwareLocalizationDetectionBCELoss()
    elif loss_function == "class_aware_localization_only_bce":
        return losses.ClassAwareLocalizationDetectionBCELoss(manipulated_loss_weight=0.5,
                                                             authentic_loss_weight=0.5,
                                                             disable_detection_loss=True)
    elif loss_function == "class_aware_localization_only_bce_with_logits":
        return losses.ClassAwareLocalizationDetectionBCELoss(
            manipulated_loss_weight=0.5,
            authentic_loss_weight=0.5,
            disable_detection_loss=True,
            loss_type=losses.LossType.BCE_WITH_LOGITS_LOSS
        )
    elif loss_function == "localization_bce_dice":
        return losses.LocalizationDetectionBCEDiceLoss(
            detection_loss_weight=.0,
            localization_bce_loss_weight=.3,
            dice_loss_weight=.7
        )
    elif loss_function == "class_aware_localization_only_bce_dice":
        return losses.ClassAwareLocalizationDetectionBCEDiceLoss(
            detection_loss_weight=.0,
            manipulated_bce_loss_weight=0.2,
            authentic_bce_loss_weight=0.2,
            dice_loss_weight=0.6
        )
    elif loss_function == "class_aware_localization_detection_bce_dice":
        return losses.ClassAwareLocalizationDetectionBCEDiceLoss(
            detection_loss_weight=0.25,
            manipulated_bce_loss_weight=0.15,
            authentic_bce_loss_weight=0.15,
            dice_loss_weight=0.45
        )
    elif loss_function == "class_aware_localization_detection_bce_dice_more_manipulated":
        return losses.ClassAwareLocalizationDetectionBCEDiceLoss(
            detection_loss_weight=0.1,
            manipulated_bce_loss_weight=0.1,
            authentic_bce_loss_weight=0.2,
            dice_loss_weight=0.6
        )
    else:
        raise RuntimeError(f"{loss_function} is not a valid loss function.")


def _create_scheduler(optimizer: torch.optim.Optimizer,
                      scheduler_name: str,
                      epochs: int,
                      steps_per_epoch: int,
                      warmup_epochs: int = 4,
                      min_lr: float = 1e-6):
    logging.info(f"LR SCHEDULER: {scheduler_name}")

    scheduler = None
    step_per_batch = False

    if scheduler_name == "cosine_annealing_warm_restarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                             T_0=1,
                                                             T_mult=2,
                                                             verbose=False)
        step_per_batch = True
    if scheduler_name == "cosine_annealing_with_warmup":
        logging.info(f"WARMUP EPOCHS: {warmup_epochs}")
        scheduler = WarmUpScheduler(
            optimizer,
            lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-warmup_epochs, eta_min=min_lr),
            len_loader=steps_per_epoch,
            warmup_steps=warmup_epochs*steps_per_epoch,
            warmup_start_lr=1e-6,
        )
        step_per_batch = True

    return scheduler, step_per_batch


def fix_ddp_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    from collections import OrderedDict
    fixed_dict = OrderedDict()
    if next(iter(state_dict.keys())).startswith("module."):
        logging.info("Model was trained using DDP. Fixing state dict!")
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            fixed_dict[name] = v
    else:
        fixed_dict = state_dict

    fixed_dict2 = fixed_dict.copy()
    for k, v in fixed_dict.items():
        if k == "image_feature_extractor.conv.weight":
            fixed_dict2["image_feature_extractor.interpolators.0.weight"] = fixed_dict[k]
            del fixed_dict2[k]
        elif k == "image_feature_extractor.conv.bias":
            fixed_dict2["image_feature_extractor.interpolators.0.bias"] = fixed_dict[k]
            del fixed_dict2[k]
        elif k.startswith("image_feature_extractor.convs"):
            fixed_dict2[k.replace("image_feature_extractor.convs",
                                  "image_feature_extractor.interpolators")] = fixed_dict[k]
            del fixed_dict2[k]

    return fixed_dict2


if __name__ == "__main__":
    cli()
