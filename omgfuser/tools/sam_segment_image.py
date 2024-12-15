"""Script that performs instance segmentation using the SAM model.

Version: 1.0

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
import pathlib
from typing import Any

import numpy as np
import torch
import click
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


@click.command()
@click.option("-i", "--input_image", required=True,
              type=click.Path(dir_okay=False, exists=True, path_type=pathlib.Path))
@click.option("-o", "--output_dir", required=True,
              type=click.Path(file_okay=False, path_type=pathlib.Path))
@click.option("--device", type=str, default="cuda", show_default=True)
@click.option("--model_type", type=str, default="vit_h", show_default=True)
@click.option("--sam_checkpoint",
              type=click.Path(dir_okay=False, exists=True,path_type=pathlib.Path),
              default=pathlib.Path("./checkpoints/sam/sam_vit_h_4b8939.pth"), show_default=True)
def cli(
    input_image: pathlib.Path,
    output_dir: pathlib.Path,
    device: str,
    model_type: str,
    sam_checkpoint: pathlib.Path
) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    segment_image(input_image, output_dir, device, model_type, sam_checkpoint)


def segment_image(
    image_path: pathlib.Path,
    output_dir: pathlib.Path,
    device: str,
    model_type: str,
    sam_checkpoint: pathlib.Path
) -> None:
    print(f"Loading image: {str(image_path)}")
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((2048, 2048))
        image = np.array(image)

    print("Predicting instances")
    instances: list[dict[str, Any]] = segment_image_with_sam(
        image, device, model_type, sam_checkpoint
    )

    print(f"Exporting instances under: {str(output_dir)}")
    visualize_instances(instances, image, f"{image_path.stem}_sam", output_dir)
    batch_export_instances([instances], ["sam"], image_path, output_dir)


@torch.no_grad()
def segment_image_with_sam(
    image: np.ndarray,
    device: str,
    model_type: str,
    sam_checkpoint: pathlib.Path
) -> list[dict[str, Any]]:
    # Load model.
    sam = sam_model_registry[model_type](checkpoint=str(sam_checkpoint))
    sam.to(device=device)
    mask_generator: SamAutomaticMaskGenerator = SamAutomaticMaskGenerator(sam)

    # Perform inference.
    instances: list[dict[str, Any]] = mask_generator.generate(image)

    # Cleanup model.
    del sam
    del mask_generator

    return instances


def visualize_instances(
    instances: list[dict[str, Any]],
    image: np.ndarray,
    name: str,
    output_dir: pathlib.Path
) -> None:
    # Visualize the instances as overlays on the original image.
    fig = plt.figure(frameon=False, dpi=2)
    fig.set_size_inches(image.shape[1] // 2, image.shape[0] // 2)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, aspect='auto')
    show_overlayed_mask(instances)
    plt.savefig(output_dir / f"{name}_overlay.png")
    plt.close()

    # Generate instance segmentation map.
    fig = plt.figure(frameon=False, dpi=2)
    fig.set_size_inches(image.shape[1] // 2, image.shape[0] // 2)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(np.zeros(image.shape), aspect='auto')
    show_overlayed_mask(instances, opacity=1.0)
    plt.savefig(output_dir / f"{name}_map.png")
    plt.close()


def batch_export_instances(
    instances_batch: list[list[dict[str, Any]]],
    model_names: list[str],
    image_path: pathlib.Path,
    output_dir: pathlib.Path
) -> None:
    """Exports the instances predicted by multiple instance segmentation models."""
    csv_data: list[dict[str, str]] = []

    # Create a unique directory for the instance maps of each image, under the output directory.
    image_out_dir: pathlib.Path = output_dir / image_path.stem
    image_out_dir.mkdir(exist_ok=True, parents=True)

    # Export all the instances predicted by each model.
    for instances, model_name in zip(instances_batch, model_names):
        for i in range(len(instances)):
            instance_map: np.ndarray = instances[i]["segmentation"]
            instance_map = instance_map.astype(np.int32)

            instance_map_path: pathlib.Path = image_out_dir / f"{model_name}_instance_{i}.png"
            Image.fromarray((instance_map*255).astype(np.uint8)).save(instance_map_path)

            csv_dict: dict[str, str] = {
                "seg_model": model_name,
                "seg_map": instance_map_path.relative_to(image_out_dir),
                "seg_score": instances[i]["predicted_iou"],
            }
            csv_data.append(csv_dict)

    csv_file_path: pathlib.Path = image_out_dir / "segmentation_instances.csv"
    with csv_file_path.open("w") as f:
        writer: csv.DictWriter = csv.DictWriter(
            f, ["seg_model", "seg_map", "seg_score"], delimiter=","
        )
        writer.writeheader()
        writer.writerows(csv_data)


def show_overlayed_mask(instances: list[dict[str, Any]], opacity: float = 0.35) -> None:
    if len(instances) == 0:
        return

    # Sort instances according to their size. Smaller instances should be drawn last, in order
    # to be visible on top of the larger instances.
    sorted_instances: list[dict[str, Any]] = sorted(instances,
                                                    key=(lambda x: x['area']),
                                                    reverse=True)

    ax = plt.gca()
    ax.set_autoscale_on(False)
    img: np.ndarray = np.ones((sorted_instances[0]['segmentation'].shape[0],
                               sorted_instances[0]['segmentation'].shape[1],
                               4))
    img[:, :, 3] = 0  # Set initial opacity to 0.
    for inst in sorted_instances:
        # Fill the area of an instance with a random color with given opacity.
        m: np.ndarray = inst['segmentation']
        color_mask: np.ndarray = np.concatenate([np.random.random(3), [opacity]])
        img[m] = color_mask
    ax.imshow(img)


if __name__ == "__main__":
    cli()
