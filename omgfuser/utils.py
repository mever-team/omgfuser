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

import csv
import math
import pathlib
import re
from enum import Enum
from typing import Optional, Any, Union
from functools import partial

import einops
import numpy as np
import torch
import seaborn
from matplotlib import colormaps, pyplot as plt
from torch import nn
from PIL import Image

from omgfuser import datasets
from omgfuser import training
from omgfuser.data import processors
from omgfuser.models import fusion


class SimilaritiesMapType(Enum):
    COSINE = "cos"
    ATTENTION_LAST_LAYER_SINGLE_HEAD = "att_single_layer_last_head"
    ATTENTION_ALL_LAYERS_ALL_HEADS = "att_all_layers_all_heads"
    GRADCAM = "gradcam"


@torch.no_grad()
def export_best_samples(
    model: nn.Module,
    device: torch.device,
    data: Union[datasets.ForensicsDataset, datasets.HandcraftedForensicsSignalsDataset],
    inputs_names: list[str],
    channels: list[int],
    results: dict[int, dict[str, np.ndarray]],
    output_dir: pathlib.Path,
    requires_sigmoid: bool,
    forward_pass_type: training.ForwardPassType,
    metric: str = "iou_loc",
    max_results: int = 300,
    processor: Optional[processors.OnlinePreprocessor] = None,
    export_similarities: bool = True,
    similarities_type: SimilaritiesMapType = SimilaritiesMapType.GRADCAM
) -> None:
    """Exports the top forgery localization results of a fusion model.

    :param model: The model for computing the outputs.
    :param device: Device where inference will be performed.
    :param data: An instance of the dataset.
    :param inputs_names: The names of the inputs to the model.
    :param channels: The number of channels of each input.
    :param results: A dict that maps the ids of the samples in the dataset, to the
        metrics computed for that sample from an evaluation session.
    :param output_dir: The directory where the top `max_results` samples will be exported.
    :param requires_sigmoid: A flag that when set to True, passes the forgery localization
        output of the model through a sigmoid layer.
    :param forward_pass_type: The type of inference for supporting models with different
        inputs and fusion approaches.
    :param metric: The metric to be used for obtaining the top `max_results` samples.
    :param processor: An optional processor for the input of the model. Meaningful only
        when `forward_pass_type == ForwardPassType.PREPROCESSOR_ATTENTION_MASK_PASS`.
    :param max_results: The number of top samples to export. If a number greater than the
        length of the dataset is provided, the whole dataset is exported.
    :param export_similarities: A flag that when set to True, exports the similarity maps between
        the input signals and the fused signals.
    """
    # Sort results according to the given metric.
    metric_results: dict[int, float] = {
        i: metrics[metric][1] for i, metrics in results.items()
        if not np.isnan(metrics[metric][1])
    }
    sorted_metric_results: list[tuple[int, float]] = sorted(
        metric_results.items(), key=lambda x: x[1], reverse=True
    )

    # Retain the requested number of top samples.
    max_results = min(max_results, len(data))
    sorted_metric_results = sorted_metric_results[:max_results]

    model.eval()
    model.to(device)
    if processor is not None:
        processor.to_device(device)

    similarities: Optional[list[torch.Tensor]] = None

    for i, (sample_id, metric_val) in enumerate(sorted_metric_results):
        if forward_pass_type == training.ForwardPassType.SIMPLE_FORWARD_PASS:
            inputs, mask, _, _ = data[sample_id]
            t_inputs: torch.Tensor = torch.from_numpy(inputs).float().to(device)
            t_inputs = torch.unsqueeze(t_inputs, 0)
            output: torch.Tensor = model(t_inputs)["localization"]
        elif forward_pass_type == training.ForwardPassType.ATTENTION_MASK_PASS:
            assert isinstance(data, datasets.ForensicsDatasetWithAttentionMask)
            inputs, attention_mask, mask, _, _ = data[sample_id]
            t_inputs: torch.Tensor = torch.from_numpy(inputs).float().to(device)
            t_inputs = torch.unsqueeze(t_inputs, 0)
            t_attention_mask: torch.Tensor = attention_mask.bool().to(device)
            t_attention_mask = t_attention_mask.unsqueeze(dim=0)
            sim_label: str = "cos"
            if (similarities_type == SimilaritiesMapType.ATTENTION_ALL_LAYERS_ALL_HEADS
                    or similarities_type == SimilaritiesMapType.ATTENTION_LAST_LAYER_SINGLE_HEAD):
                sim_label = "attn"
            outputs = model(t_inputs, t_attention_mask,
                            return_similarities=export_similarities,
                            similarities_type=sim_label)
            output: torch.Tensor = outputs["localization"]
            if export_similarities and (
                    similarities_type == SimilaritiesMapType.COSINE
                    or similarities_type == SimilaritiesMapType.ATTENTION_LAST_LAYER_SINGLE_HEAD
                    or similarities_type == SimilaritiesMapType.ATTENTION_ALL_LAYERS_ALL_HEADS
            ):
                similarities = outputs["similarities"]
            elif export_similarities and similarities_type == SimilaritiesMapType.GRADCAM:
                similarities = compute_gradcam_map(model, t_inputs, t_attention_mask)
        elif forward_pass_type == training.ForwardPassType.PREPROCESSOR_ATTENTION_MASK_PASS:
            collate_fn = data.build_collate_fn()
            batch: dict[str, Any] = processor.preprocess(collate_fn([data[sample_id],]))
            inputs_batch: torch.Tensor = batch["input"]
            attention_mask_batch: torch.Tensor = batch["attention_mask"]
            inputs_batch = inputs_batch.to(device)
            attention_mask_batch = attention_mask_batch.bool().to(device)
            outputs: dict[str, torch.Tensor] = model(inputs_batch, attention_mask_batch,
                                                     return_similarities=export_similarities)
            output: torch.Tensor = outputs["localization"]
            mask: np.ndarray = batch["mask"].squeeze(dim=0).detach().numpy()
            inputs: np.ndarray = inputs_batch.squeeze(dim=0).cpu().detach().numpy()
            if export_similarities:
                similarities = outputs["similarities"]
        else:
            raise RuntimeError(f"Non-supported forward pass type: {forward_pass_type.name}")

        if requires_sigmoid:
            output = torch.nn.functional.sigmoid(output)
        output = output.detach().cpu()

        if similarities is not None:
            if similarities_type == SimilaritiesMapType.ATTENTION_LAST_LAYER_SINGLE_HEAD:
                # Extract the last transformer layer and the last attention head.
                similarities = [s[:, -1, 0, :, :] for s in similarities]
            elif similarities_type == SimilaritiesMapType.ATTENTION_ALL_LAYERS_ALL_HEADS:
                similarities = [torch.amax(s, dim=(1, 2)) for s in similarities]

            similarities: list[np.ndarray] = [
                np.clip(torch.squeeze(s).detach().cpu().numpy(), .0, 1.0) for s in similarities
            ]

            if similarities == SimilaritiesMapType.COSINE:
                similarities = [s / s.max() if s.max() > 0 else s for s in similarities]

        sample_type: str = "manipulated" if mask.max() > 0.5 else "authentic"
        output_name: str = f"{i}_{sample_id}_{sample_type}_{metric}_{metric_val:.3f}"
        sample_output_dir: pathlib.Path = output_dir / output_name
        sample_output_dir.mkdir(exist_ok=True, parents=True)
        save_sample(
            inputs=inputs,
            predicted=torch.squeeze(output, dim=0).numpy(),
            mask=mask,
            inputs_channels=channels,
            inputs_names=inputs_names,
            output_dir=sample_output_dir,
            imagenet_normalized=data.imagenet_image_normalization,
            similarities=similarities,
        )


def save_sample(
    inputs: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
    inputs_channels: list[int],
    inputs_names: list[str],
    output_dir: pathlib.Path,
    imagenet_normalized: bool = False,
    similarities: Optional[list[np.ndarray]] = None
) -> None:
    # Separate inputs.
    separated_inputs: list[np.ndarray] = []
    channels_start: int = 0
    for c in inputs_channels:
        if c == 0:
            continue
        if (channels_start + c) == inputs.shape[0]:
            separated_inputs.append(inputs[channels_start:, :, :])
        else:
            separated_inputs.append(inputs[channels_start:channels_start+c, :, :])
        channels_start += c
    inputs_channels = inputs_channels.copy()
    inputs_channels.remove(0)
    assert len(separated_inputs) == len(inputs_channels)

    target_shape: tuple[int, int] = (mask.shape[2], mask.shape[1])  # (W, H)
    # Save inputs.

    signals_paths: list[pathlib.Path] = []
    overlayed_signals_path: list[pathlib.Path] = []

    for i, (inp, name) in enumerate(zip(separated_inputs, inputs_names)):
        if name == "image" and imagenet_normalized:
            # Invert the normalization with ImageNet mean and std.
            mean: np.ndarray = np.array((0.485, 0.456, 0.406)).reshape((3, 1, 1))
            std: np.ndarray = np.array((0.229, 0.224, 0.225)).reshape((3, 1, 1))
            inp = inp * std + mean
            image_path = output_dir / f"{name}.png"
            signals_paths.append(image_path)
            save_image(inp, image_path, target_shape)
            if similarities is not None:
                overlayed_image_path = output_dir / f"similarity_{name}.png"
                overlayed_signals_path.append(overlayed_image_path)
                save_image_with_similarity_overlay(
                    inp, similarities[i],
                    overlayed_image_path,
                    target_shape, similarity_color_palette="jet"
                )
        elif name == "dct":
            for j in range(inp.shape[0]):
                inp_ch = inp[j, :, :]
                inp_min = np.min(inp_ch)
                inp_max = np.max(inp_ch)
                dist = inp_max - inp_min
                inp_ch = inp_ch - inp_min
                if dist > 0:
                    inp_ch = inp_ch / dist
                sig_path = output_dir / f"{name}_{j}.png"
                signals_paths.append(sig_path)
                save_image(np.expand_dims(inp_ch, axis=0),
                           sig_path, target_shape)
                if similarities is not None:
                    overlayed_sig_path = output_dir / f"similarity_{name}_{j}.png"
                    overlayed_signals_path.append(overlayed_sig_path)
                    save_image_with_similarity_overlay(
                        np.expand_dims(inp_ch, axis=0), similarities[i],
                        overlayed_sig_path,
                        target_shape, similarity_color_palette="jet"
                    )
        else:
            inp = (inp + 1.0) / 2.0  # Restore the input to [0, 1]
            sig_path = output_dir / f"{name}.png"
            signals_paths.append(sig_path)
            save_image(inp, sig_path, target_shape, color_palette="coolwarm")
            if similarities is not None:
                overlayed_sig_path = output_dir / f"similarity_{name}.png"
                overlayed_signals_path.append(overlayed_sig_path)
                save_image_with_similarity_overlay(
                    inp, similarities[i],
                    overlayed_sig_path,
                    target_shape, image_color_palette="gray", similarity_color_palette="jet"
                )
    # Save predicted output.
    predicted_path = output_dir/f"predicted.png"
    signals_paths.append(predicted_path)
    save_image(predicted, predicted_path, target_shape, color_palette="coolwarm")
    # Save ground-truth mask.
    gt_path = output_dir/f"ground_truth.png"
    overlayed_signals_path.append(gt_path)
    save_image(mask, gt_path)

    labels = inputs_names.copy()
    del labels[-1]
    labels.append("predicted")
    if similarities is not None:
        save_similarity_figure(signals_paths, overlayed_signals_path, labels,
                               output_dir / "similarity_fig.png")


def save_image(
    image: np.ndarray,
    path: pathlib.Path,
    target_size: Optional[tuple[int, int]] = None,
    color_palette: Optional[str] = None
) -> None:
    """Saves a (C, H, W) ndarray as an image."""
    image = image.transpose((1, 2, 0))  # (C, H, W) -> (H, W, C)

    if color_palette is not None:
        if color_palette != "jet":
            cmap = seaborn.color_palette(color_palette, as_cmap=True)
        else:
            cmap = colormaps["jet"]
        image = cmap(image)

    pil_image = Image.fromarray(np.squeeze((image*255).astype(np.uint8)))
    if target_size:
        pil_image = pil_image.resize(target_size)
    pil_image.save(path)


def save_image_with_similarity_overlay(
    image: np.ndarray,
    similarity_map: np.ndarray,
    path: pathlib.Path,
    target_size: Optional[tuple[int, int]] = None,
    image_color_palette: Optional[str] = None,
    similarity_color_palette: Optional[str] = None,
) -> None:
    """Saves a (C, H, W) ndarray as an image, with a similarity map overlayed on top of it."""
    image = image.transpose((1, 2, 0))  # (C, H, W) -> (H, W, C)
    similarity_map = np.expand_dims(similarity_map, axis=2)  # (H, W) -> (H, W, 1)

    # Colormap the image and the similarity map.
    if image_color_palette is not None:
        if image_color_palette != "jet":
            cmap = seaborn.color_palette(image_color_palette, as_cmap=True)
        else:
            cmap = colormaps["jet"]
        image = cmap(image)
    if similarity_color_palette is not None:
        if similarity_color_palette != "jet":
            cmap = seaborn.color_palette(similarity_color_palette, as_cmap=True)
        else:
            cmap = colormaps["jet"]
        similarity_map = cmap(similarity_map)

    pil_image = Image.fromarray(np.squeeze((image * 255).astype(np.uint8))).convert("RGBA")
    pil_similarity = Image.fromarray(
        np.squeeze((similarity_map * 255).astype(np.uint8))).convert("RGBA")

    if target_size:
        pil_image = pil_image.resize(target_size)
        pil_similarity = pil_similarity.resize(target_size)

    overlayed = Image.blend(pil_image, pil_similarity, 0.5)
    overlayed.save(path)


def save_similarity_figure(
    signals: list[pathlib.Path],
    signals_overlayed: list[pathlib.Path],
    labels: list[str],
    path: pathlib.Path
) -> None:
    rows: int = 2
    columns: int = len(signals)
    fig = plt.figure(figsize=(18, 6))

    # Display signals.
    for i in range(len(signals)):
        ax = fig.add_subplot(rows, columns, i+1)
        ax.set_title(labels[i])
        ax.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        with Image.open(signals[i]) as sig:
            plt.imshow(sig, aspect="auto")

    # Display overlayed signals
    for i in range(len(signals)):
        ax = fig.add_subplot(rows, columns, rows*columns//2+i+1)
        ax.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        with Image.open(signals_overlayed[i]) as sig:
            plt.imshow(sig, aspect="auto")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(path, bbox_inches='tight')


@torch.enable_grad()
def compute_gradcam_map(
    model: fusion.MaskedAttentionFusionModelWithImage,
    inputs: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: str = "det_head"
) -> list[torch.Tensor]:
    activations: dict[int, torch.Tensor] = {}
    gradients: dict[int, torch.Tensor] = {}
    output_tensors: list[torch.Tensor] = []
    backward_hooks: list = []
    forward_hooks: list = []

    def output_tensor_hook(module, args, output) -> None:
        output_tensors.append(output)

    def forward_hook(index, module, args, output) -> None:
        activations[index] = output

    def backward_hook(index, module, grad_input, grad_output) -> None:
        gradients[index] = grad_output[0].clone()

    forward_hooks.append(
        model.image_feature_extractor.register_forward_hook(partial(forward_hook, 0))
    )
    backward_hooks.append(
        model.image_feature_extractor.register_full_backward_hook(partial(backward_hook, 0))
    )
    for i in range(5):
        forward_hooks.append(
            model.mask_feature_extractors[i].register_forward_hook(
                partial(forward_hook, i+1)
            )
        )
        backward_hooks.append(
            model.mask_feature_extractors[i].register_full_backward_hook(
                partial(backward_hook, i+1)
            )
        )
    forward_hooks.append(
        model.overall_fusion.positional_fusion_transformers[-1].register_forward_hook(
            output_tensor_hook
        )
    )

    outputs: dict[str, torch.Tensor] = model(inputs, attention_mask)

    if mode == "tft" or mode == "det_head":
        if mode == "tft":  # After TFT.
            out_t: torch.Tensor = output_tensors[0]
            out_t = out_t[:, 0, :]
            out_t = einops.rearrange(out_t, "(b l) c -> b l c", b=inputs.size(dim=0))
            out_t = torch.flatten(out_t)
            out_t = torch.dot(out_t, out_t)
        elif mode == "det_head":  # Detection head.
            out_t = outputs["detection"]

        out_t.backward()

        # Convert image tensors from (B, L, C) to (B, C, H, W).
        activations[0] = einops.rearrange(
            activations[0], "b (h w) c -> b c h w", h=int(math.sqrt(activations[0].size(dim=1)))
        )
        gradients[0] = einops.rearrange(
            gradients[0], "b (h w) c -> b c h w", h=int(math.sqrt(gradients[0].size(dim=1)))
        )

        gradcam_maps: list[torch.Tensor] = [
            compute_gradcam(activations[i], gradients[i]) for i in range(len(activations))
        ]
    elif mode == "loc_head":
        # Localization head
        outp: torch.Tensor = outputs["localization"]
        outp = torch.nn.functional.avg_pool2d(outp, 8, 8)

        gradcam_maps_list: list[list[torch.Tensor]] = []

        # Convert image tensors from (B, L, C) to (B, C, H, W).
        activations[0] = einops.rearrange(
            activations[0], "b (h w) c -> b c h w",
            h=int(math.sqrt(activations[0].size(dim=1)))
        )

        for i in range(outp.size(dim=2)):
            for j in range(outp.size(dim=3)):
                model.zero_grad()
                out_t = outp[0][0][i][j].clone()
                out_t.backward(retain_graph=True)

                gradients[0] = einops.rearrange(
                    gradients[0], "b (h w) c -> b c h w", h=int(math.sqrt(gradients[0].size(dim=1)))
                )

                gradcam_maps: list[torch.Tensor] = [
                    compute_gradcam(activations[i], gradients[i]) for i in range(len(activations))
                ]
                gradcam_maps_list.append(gradcam_maps)

        out_t.backward()

        gradcam_maps = [torch.mean(torch.stack(m, dim=0), dim=0) for m in zip(*gradcam_maps_list)]

    for h in forward_hooks:
        h.remove()
    for h in backward_hooks:
        h.remove()

    maps_max = torch.max(torch.stack(gradcam_maps))
    gradcam_maps = [gm / maps_max for gm in gradcam_maps]

    return gradcam_maps


def compute_gradcam(activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    # weight the channels by corresponding gradients
    for i in range(activations.size(dim=1)):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = torch.nn.functional.relu(heatmap)
    # heatmap /= torch.max(heatmap)
    return heatmap


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    text = str(text)
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def write_csv_file(data: list[dict[str, Any]], output_file: pathlib.Path) -> None:
    with output_file.open("w") as f:
        writer: csv.DictWriter = csv.DictWriter(f, fieldnames=data[0].keys(), delimiter=",")
        writer.writeheader()
        for r in data:
            writer.writerow(r)
