"""Script for inverting the colors of ground-truth forgery localization masks.

Some datasets, like DSO-1 and NIST ones, designate the forged areas with black color
(low values) and the authentic ones with white color (high values). However, the most
common approach in the field is the opposite. This script enables the inversion of
the masks in such datasets, in order to match the common representation used in the
field.

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

import click
from tqdm import tqdm
from PIL import Image, ImageOps


__version__: str = "1.0.0"
__revision__: int = 1
__author__: str = "Dimitrios Karageorgiou"
__email__: str = "dkarageo@iti.gr"


@click.command()
@click.option("-i", "--input_dir", required=True,
              type=click.Path(file_okay=False, exists=True, path_type=pathlib.Path),
              help="Path to the directory that contains the forgery localization masks.")
@click.option("-o", "--output_dir", required=True,
              type=click.Path(path_type=pathlib.Path),
              help="Path to the directory where inverted ")
def cli(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path
) -> None:
    masks: list[pathlib.Path] = list(input_dir.iterdir())

    output_dir.mkdir(exist_ok=True, parents=True)

    for m in tqdm(masks, "Creating inverted masks", unit="mask"):
        inverted_m: pathlib.Path = output_dir / m.name
        create_inverted_mask(m, inverted_m)


def create_inverted_mask(
    mask_path: pathlib.Path,
    output_path: pathlib.Path
) -> None:
    with Image.open(mask_path) as mask:
        # Mask should be an 8-bit single-channel image.
        if mask.mode != "L":
            mask = mask.convert("L")
        mask = ImageOps.invert(mask)
        mask.save(output_path)


if __name__ == "__main__":
    cli()
