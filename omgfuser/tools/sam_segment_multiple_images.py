"""Script that perform instance segmentation of multiple images using the SAM model.

The files that are generated for each image are the following:
- An overlay map displaying the predicted instances over the original image.
- A segmentation map displaying the predicted instances.
- N maps for each instance detected in the image.
- A csv file containing the detail about the instances detected in each image.

Version: 1.1

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
import pickle
import shutil
import csv
import threading
import timeit
from functools import partial
from typing import Optional, Any, Generator

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import click
import filetype
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


@click.command()
@click.option("-i", "--input_dir", required=True,
              type=click.Path(file_okay=False, exists=True, path_type=pathlib.Path))
@click.option("-c", "--images_csv",
              type=click.Path(dir_okay=False, exists=True, path_type=pathlib.Path))
@click.option("-o", "--output_dir", required=True,
              type=click.Path(file_okay=False, path_type=pathlib.Path))
@click.option("--no-vis", is_flag=True,
              help="Flag that when provided disables the generation of the overlay map "
                   "and the overall segmentation map. Only the separate masks for each instance "
                   "and the corresponding CSV file are generated in that case.")
@click.option("--no-cache", is_flag=True,
              help="Flag that when provided disables the checking of already computed outputs.")
@click.option("--device", type=str, default="cuda:0", show_default=True,
              help="The device that will be used for inference. E.g. `cpu`, `cuda`, `cuda:1`")
@click.option("--model_type", type=str, default="vit_h", show_default=True,
              help="The type of the pretrained Segment Anything Model (SAM).")
@click.option("--sam_checkpoint",
              type=click.Path(dir_okay=False, exists=True, path_type=pathlib.Path),
              default=pathlib.Path("./checkpoints/sam/sam_vit_h_4b8939.pth"), show_default=True,
              help="Path to a pretrained checkpoint of the Segment Anything Model (SAM). The type "
                   "of the pretrained model should match `model_type`. So, if the pretrained "
                   "model is not ViT-H, change the value of `model_type` accordingly.")
@click.option("--chunk_size", type=int,
              help="When chunk size is provided the instance segmentation is performed in "
                   "multiple iterations, each processing at most the given number of images. "
                   "The use of this parameter allows to reduce the amount of disk required for "
                   "storing the intermediate pickle files that are generated during the "
                   "instance segmentation. However, this argument is currently only supported "
                   "for directory inputs. It has not yet been implemented for CSV inputs.")
def cli(
    input_dir: pathlib.Path,
    images_csv: Optional[pathlib.Path],
    output_dir: pathlib.Path,
    no_vis: bool,
    no_cache: bool,
    device: str,
    model_type: str,
    sam_checkpoint: pathlib.Path,
    chunk_size: Optional[int]
) -> None:
    if not input_dir and not images_csv:
        print("Missing Arguments: One of --input_dir or --images_csv is required.")
        exit(-1)
    if images_csv is not None and chunk_size is not None:
        print("The chunk_size argument is not supported when CSV file is used as an input.")
        exit(-1)

    if images_csv:
        # Read images according to the csv 'image' column, using the
        # input dir as the root dir for paths.
        segment_images_in_csv(
            input_dir,
            images_csv,
            output_dir,
            not no_vis,
            check_cache=not no_cache,
            model_type=model_type,
            sam_checkpoint=sam_checkpoint,
            device=device
        )
    else:
        # Segment all images in the input dir.
        segment_images_in_dir(
            input_dir,
            output_dir,
            not no_vis,
            check_cache=not no_cache,
            model_type=model_type,
            sam_checkpoint=sam_checkpoint,
            device=device,
            chunk_size=chunk_size
        )


def segment_images_in_csv(
    input_dir: pathlib.Path,
    csv_path: pathlib.Path,
    output_dir: pathlib.Path,
    visualization: bool,
    check_cache: bool,
    model_type: str,
    sam_checkpoint: pathlib.Path,
    device: Optional[str] = None,
) -> None:
    with csv_path.open("r") as f:
        csvreader = csv.DictReader(f, delimiter=",")
        image_paths: list[pathlib.Path] = [input_dir/r["image"] for r in csvreader]

    print(f"Images in csv: {len(image_paths)}")

    # Split the images into groups according to their directory they are contained.
    # In that way, the outputs can be generated in a directory structure same as the input one.
    parent_dirs: set[pathlib.Path] = {p.parent for p in image_paths}
    image_paths_by_parent_dirs: dict[pathlib.Path, list[pathlib.Path]] = {
        p: [] for p in parent_dirs
    }
    for p in image_paths:
        image_paths_by_parent_dirs[p.parent].append(p)

    pickles: list[tuple[pathlib.Path, list[pathlib.Path], list[pathlib.Path]]] = []

    # Perform all the GPU computations in one pass.
    for group_dir, group_images in image_paths_by_parent_dirs.items():
        group_output_dir: pathlib.Path = output_dir / group_dir.relative_to(input_dir)
        group_output_dir.mkdir(exist_ok=True, parents=True)
        instance_pickles, computed_images = segment_images(
            group_images, group_output_dir, model_type, sam_checkpoint,
            gen_maps=False, check_cache=check_cache, device=device
        )
        if len(computed_images) > 0:
            pickles.append((group_output_dir, computed_images, instance_pickles))

    # Generate the actual segmentation maps in a second pass.
    for group_output_dir, computed_images, instance_pickles in pickles:
        generate_visualization_maps(computed_images, group_output_dir,
                                    instance_pickles, visualization)


def segment_images_in_dir(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    visualization: bool,
    check_cache: bool,
    model_type: str,
    sam_checkpoint: pathlib.Path,
    device: Optional[str] = None,
    chunk_size: Optional[int] = None
) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Input directory: {str(input_dir)}")
    input_images: list[pathlib.Path] = [f for f in input_dir.iterdir() if filetype.is_image(f)]
    segment_images(input_images,
                   output_dir,
                   model_type,
                   sam_checkpoint,
                   visualization=visualization,
                   check_cache=check_cache,
                   device=device,
                   chunk_size=chunk_size)


def segment_images(
    input_images: list[pathlib.Path],
    output_dir: pathlib.Path,
    model_type: str,
    sam_checkpoint: pathlib.Path,
    gen_maps: bool = True,
    check_cache: bool = True,
    visualization: bool = True,
    device: Optional[str] = None,
    chunk_size: Optional[int] = None
) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    if check_cache:
        input_images = find_images_without_cached_output(input_images, output_dir)
    if input_images == 0:  # TODO: Maybe should be len(input_images)?
        return [], []

    print(f"Segmenting {len(input_images)} images...")

    chunks: list[list[pathlib.Path]]
    if chunk_size is None:
        chunks = [input_images]
    else:
        assert gen_maps, \
            "Processing in chunks is not yet supported when generation of visualization " \
            "maps is disabled."
        chunks = list(split_in_chunks(input_images, chunk_size))
        print(f"Segmenting images in {len(chunks)} chunks...")

    pickles: list[pathlib.Path] = []
    for chunk in chunks:
        # Perform instance segmentation using the SAM model.
        chunk_pickles: list[pathlib.Path] = segment_multiple_images_with_sam(
            chunk, output_dir, model_type, sam_checkpoint, device=device
        )

        if gen_maps:
            generate_visualization_maps(
                chunk, output_dir, chunk_pickles, visualization
            )

        pickles.extend(chunk_pickles)

    return pickles, input_images


def find_images_without_cached_output(images: list[pathlib.Path],
                                      cache_dir: pathlib.Path) -> list[pathlib.Path]:
    uncached: list[pathlib.Path] = []

    print(f"Searching cached images in {cache_dir}")

    for img in tqdm(images, desc="Searching cache", unit="image"):
        target_image: pathlib.Path = cache_dir / img.stem / f"segmentation_instances.csv"
        if not target_image.exists():
            uncached.append(img)

    found: int = len(images) - len(uncached)
    if found > 0:
        print(f"Found {found} images from a previous run.")

    return uncached


def generate_visualization_maps(input_images: list[pathlib.Path],
                                output_dir: pathlib.Path,
                                pickles: list[pathlib.Path],
                                visualization: bool) -> None:
    for image_path, pkl in tqdm(list(zip(input_images, pickles)),
                                desc="Generating visualization maps...",
                                unit="image"):
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image.thumbnail((2048, 2048))
                image = np.array(image)
            with pkl.open("rb") as f:
                instances: list[dict[str, Any]] = pickle.load(f)

            if visualization:
                visualize_instances(instances, image, image_path.stem, output_dir)

            batch_export_instances([instances], ["sam"], image_path, output_dir)
        except Exception as e:
            logging.error(f"Failed to generate visualization for {str(image_path)}")
            raise e

    # Clean pickle directories.
    if pickles:
        shutil.rmtree(pickles[0].parent)


@torch.no_grad()
def segment_multiple_images_with_sam(
    image_paths: list[pathlib.Path],
    output_dir: pathlib.Path,
    model_type: str,
    sam_checkpoint: pathlib.Path,
    device: Optional[str] = None,
    verbose: bool = False
) -> list[pathlib.Path]:
    """Performs instance segmentation on a list of images using a SAM model."""
    print(f"Segmenting {len(image_paths)} images using the SAM ({model_type}) model...")

    # Load model.
    sam = sam_model_registry[model_type](checkpoint=str(sam_checkpoint))
    sam.to(device=device)
    mask_generator: SamAutomaticMaskGenerator = SamAutomaticMaskGenerator(sam)

    # Output instance segmentations to a dir.
    pickles_dir: pathlib.Path = output_dir / "sam_pickles"
    pickles_dir.mkdir(exist_ok=True, parents=True)

    # Find images whose pickle file does not exist.
    images_to_compute: list[pathlib.Path] = []
    target_pickle_files: list[pathlib.Path] = []
    all_pickle_files: list[pathlib.Path] = []
    for path in image_paths:
        pickle_file: pathlib.Path = pickles_dir / f"{path.stem}.pkl"
        if not pickle_file.exists():
            images_to_compute.append(path)
            target_pickle_files.append(pickle_file)
        all_pickle_files.append(pickle_file)

    if verbose:
        inference_time: float = .0
        total_samples: int = 0

    # Compute instance segmentation maps.
    dataset: DirectoryDataset = DirectoryDataset(images_to_compute, target_pickle_files)
    loader: DataLoader = DataLoader(dataset, batch_size=None, num_workers=8, pin_memory=True)
    # The saving of instances is performed in a separate thread, to overlap the computation.
    last_save_thread: Optional[threading.Thread] = None
    for image, pickle_file in tqdm(loader, desc="Computing segmentations with SAM", unit="image"):
        if verbose:
            torch.cuda.synchronize()
            start_t: float = timeit.default_timer()

        instances: list[dict[str, Any]] = mask_generator.generate(image.numpy())

        if verbose:
            torch.cuda.synchronize()
            stop_t: float = timeit.default_timer()

        # Make sure that previous instances have been saved before proceeding to current ones.
        if last_save_thread:
            last_save_thread.join()

        # Send saving of instances to a new thread.
        last_save_thread = threading.Thread(
            target=partial(save_instances_to_pickle, instances, pickle_file))
        last_save_thread.start()

        if verbose:
            inference_time += stop_t-start_t
            total_samples += 1
            print(f"Elapsed time: {(stop_t-start_t)*1000} ms")

    if verbose:
        print(f"Inference time: {inference_time} secs")
        print(f"Inference time per sample: {inference_time/total_samples*1000} ms")

    # Wait for the last saving to finish.
    if last_save_thread:
        last_save_thread.join()

    # Cleanup model.
    del sam
    del mask_generator
    torch.cuda.empty_cache()

    return all_pickle_files


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


def save_instances_to_pickle(instances: list[dict[str, Any]], pickle_file: pathlib.Path) -> None:
    with pickle_file.open("wb") as f:
        pickle.dump(instances, f)


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


def split_in_chunks(lst: list[Any], n: int) -> Generator[list, None, None]:
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class DirectoryDataset(Dataset):

    def __init__(self, images: list[pathlib.Path], outputs: list[pathlib.Path]):
        self.images: list[pathlib.Path] = images
        self.outputs: list[pathlib.Path] = outputs

    def __getitem__(self, item: int) -> tuple[np.ndarray, pathlib.Path]:
        with Image.open(self.images[item]) as image:
            image = image.convert("RGB")
            image.thumbnail((2048, 2048))
            image = np.array(image)

        return image, self.outputs[item]

    def __len__(self) -> int:
        return len(self.images)


if __name__ == "__main__":
    cli()
