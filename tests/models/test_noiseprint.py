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
import unittest

import torch
import numpy as np
from PIL import Image

from omgfuser.models import noiseprint


parent_dir: pathlib.Path = pathlib.Path(__file__).parent


class TestNoiseprintPlusPlus(unittest.TestCase):

    def test_noiseprint(self) -> None:
        with Image.open(parent_dir / "test_data" / "tampered1.png") as img:
            img = img.convert("RGB")
            img_array = np.array(img)
            img_array = img_array.transpose((2, 0, 1)) / 256.0

        with torch.no_grad():
            model = noiseprint.NoiseprintPlusPlus(
                pretrained_weights=parent_dir / "test_data/noiseprintplusplus.pth"
            )
            model.eval()

            npp = model(torch.tensor(img_array, dtype=torch.float).unsqueeze(dim=0))

        reference_npp: np.ndarray = np.load(parent_dir / "test_data/tampered1.png.npz")["np++"]

        self.assertTrue(np.all(np.abs(reference_npp - npp.squeeze().detach().cpu().numpy()) < 1e-5))
