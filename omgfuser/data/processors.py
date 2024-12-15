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

# import logging
import pathlib
# import timeit
from typing import Any

import torch
import torchvision.transforms.functional

from omgfuser.models import noiseprint, dctstream


class OnlinePreprocessor:

    def to_device(self, device) -> None:
        raise NotImplementedError

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class NoiseprintProcessor(OnlinePreprocessor):

    def __init__(self, pretrained_weights: pathlib.Path = "checkpoints/noiseprintplusplus.pth"):
        super().__init__()
        self.noiseprint_model: noiseprint.NoiseprintPlusPlus = noiseprint.NoiseprintPlusPlus(
            pretrained_weights=pretrained_weights
        )
        self.noiseprint_model.eval()
        self.device = "cpu"

    def to_device(self, device) -> None:
        self.device = device
        self.noiseprint_model = self.noiseprint_model.to(device)

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        images: list[torch.Tensor] = batch["original_image"]
        image_sizes: tuple[torch.Tensor, torch.Tensor] = batch["unpadded_size"]
        target_size: tuple[int, int] = (batch["image"].size(dim=2), batch["image"].size(dim=3))

        noiseprint_outputs: list[torch.Tensor] = []

        for i, img in enumerate(images):
            img_size: tuple[int, int] = (int(image_sizes[0][i]), int(image_sizes[1][i]))

            img = img.to(self.device)

            img = img / 256.0
            img = img.unsqueeze(dim=0)

            npp = self.noiseprint_model(img)

            npp = torchvision.transforms.functional.resize(npp, list(img_size))

            if img_size != target_size:  # Padding of the noiseprint output is required.
                if img_size[0] < img_size[1]:  # Padding required on vertical axis.
                    npp = torchvision.transforms.functional.pad(
                        npp, [0, 0, 0, target_size[0]-int(img_size[0])]
                    )
                else:  # Padding required on the horizontal axis.
                    npp = torchvision.transforms.functional.pad(
                        npp, [0, 0, target_size[1]-int(img_size[1]), 0]
                    )

            noiseprint_outputs.append(npp)

        batched_noiseprint: torch.Tensor = torch.cat(noiseprint_outputs, dim=0)

        del batch["original_image"]

        batch["image"] = batch["image"].to(self.device)
        if "input" in batch:
            batch["input"] = torch.cat([batch["input"], batched_noiseprint], dim=1)
        else:
            batch["input"] = torch.cat([batch["image"], batched_noiseprint], dim=1)
        return batch


class DCTProcessor(OnlinePreprocessor):
    def __init__(self, pretrained_weights: pathlib.Path = "checkpoints/DCT_djpeg.pth.tar"):
        super().__init__()
        self.dct_model: dctstream.DCTStream = dctstream.DCTStream(include_final_layers=True)
        self.dct_model.init_weights(str(pretrained_weights))

        self.dct_model.eval()
        self.device = "cpu"

    def to_device(self, device) -> None:
        self.device = device
        self.dct_model = self.dct_model.to(device)

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        # torch.cuda.synchronize()
        # start_time: float = timeit.default_timer()

        # start_time_tensors: float = timeit.default_timer()
        dct_vols: list[torch.Tensor] = batch["dct_vol"]
        qtable: torch.Tensor = batch["qtable"]
        image_sizes: tuple[torch.Tensor, torch.Tensor] = batch["unpadded_size"]
        target_size: tuple[int, int] = (batch["image"].size(dim=2), batch["image"].size(dim=3))
        # stop_time_tensors: float = timeit.default_timer()
        # logging.info(f"DCT Processor time (Input): {stop_time_tensors - start_time_tensors:.3f}")

        dct_outputs: list[torch.Tensor] = []

        # torch.cuda.synchronize()
        # start_time_compute: float = timeit.default_timer()

        qtable = qtable.to(self.device)

        for i, dct_vol in enumerate(dct_vols):
            img_size: tuple[int, int] = (int(image_sizes[0][i]), int(image_sizes[1][i]))

            dct_vol = dct_vol.to(self.device)
            dct_vol = dct_vol.unsqueeze(dim=0)

            dct_out = self.dct_model(dct_vol, qtable[i].unsqueeze(dim=0))

            dct_out = torchvision.transforms.functional.resize(dct_out, list(img_size))

            if img_size != target_size:  # Padding of the DCT output is required.
                if img_size[0] < img_size[1]:  # Padding required on vertical axis.
                    dct_out = torchvision.transforms.functional.pad(
                        dct_out, [0, 0, 0, target_size[0]-int(img_size[0])]
                    )
                else:  # Padding required on the horizontal axis.
                    dct_out = torchvision.transforms.functional.pad(
                        dct_out, [0, 0, target_size[1]-int(img_size[1]), 0]
                    )

            dct_outputs.append(dct_out)
        # stop_time_compute: float = timeit.default_timer()
        # logging.info(f"DCT Processor time (Compute): {stop_time_compute - start_time_compute:.3f}")

        # torch.cuda.synchronize()
        # start_time_out: float = timeit.default_timer()
        batched_dct: torch.Tensor = torch.cat(dct_outputs, dim=0)

        del batch["dct_vol"]
        del batch["qtable"]

        batch["image"] = batch["image"].to(self.device)
        if "input" in batch:
            batch["input"] = torch.cat([batch["input"], batched_dct], dim=1)
        else:
            batch["input"] = torch.cat([batch["image"], batched_dct], dim=1)
        # stop_time_out: float = timeit.default_timer()
        # logging.info(f"DCT Processor time (Out): {stop_time_out - start_time_out:.3f}")

        # stop_time: float = timeit.default_timer()
        # logging.info(f"DCT Processor time: {stop_time-start_time:.3f}")

        return batch


class CombinedProcessor(OnlinePreprocessor):

    def __init__(self, processors: list[OnlinePreprocessor]):
        super().__init__()

        self.processors: list[OnlinePreprocessor] = processors

    def to_device(self, device) -> None:
        for p in self.processors:
            p.to_device(device)

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        for p in self.processors:
            batch = p.preprocess(batch)
        return batch
