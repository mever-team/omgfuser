"""Script for generating fusion train-val-test data csv for CASIA datasets.

CASIAv1 is utilized as testing data.
CASIAv2 provides training and validation data. (85% training - 15% validation)

Parametrization can be performed by changing the values of the constants. This
script is intended for a single usage to generate a valid CSV file, so, no much
attention has been paid to the code clarity.

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
from pathlib import Path
from typing import Union

import filetype
from sklearn.model_selection import train_test_split

CASIA_V1_AUTHENTIC_DIR: str = "/nas3/dkarageo/CASIAv1/Au"
CASIA_V1_MANIPULATED_DIR: str = "/nas3/dkarageo/CASIAv1/Tp_all_with_masks"
CASIA_V1_MASKS_DIR: str = "/nas3/dkarageo/CASIAv1/Masks_all"
CASIA_V1_EVA_AUTHENTIC_DIR: str = "/nas3/dkarageo/CASIAv1/Predictions/EVA/authentic_masks"
CASIA_V1_EVA_MANIPULATED_DIR: str = "/nas3/dkarageo/CASIAv1/Predictions/EVA/manipulated_masks"
CASIA_V1_ADQ1_AUTHENTIC_DIR: str = "/nas3/dkarageo/CASIAv1/Predictions/ADQ1/authentic_masks"
CASIA_V1_ADQ1_MANIPULATED_DIR: str = "/nas3/dkarageo/CASIAv1/Predictions/ADQ1/manipulated_masks"
CASIA_V1_BLK_AUTHENTIC_DIR: str = "/nas3/dkarageo/CASIAv1/Predictions/BLK/authentic_masks"
CASIA_V1_BLK_MANIPULATED_DIR: str = "/nas3/dkarageo/CASIAv1/Predictions/BLK/manipulated_masks"
CASIA_V1_CAGI_AUTHENTIC_DIR: str = "/nas3/dkarageo/CASIAv1/Predictions/CAGI/authentic_masks"
CASIA_V1_CAGI_MANIPULATED_DIR: str = "/nas3/dkarageo/CASIAv1/Predictions/CAGI/manipulated_masks"
CASIA_V1_DCT_AUTHENTIC_DIR: str = "/nas3/dkarageo/CASIAv1/Predictions/DCT/authentic_masks"
CASIA_V1_DCT_MANIPULATED_DIR: str = "/nas3/dkarageo/CASIAv1/Predictions/DCT/manipulated_masks"
CASIA_V1_SPLICEBUSTER_AUTHENTIC_DIR: str = \
    "/nas3/dkarageo/CASIAv1/Predictions/Splicebuster_gmm_num=10/authentic_masks"
CASIA_V1_SPLICEBUSTER_MANIPULATED_DIR: str = \
    "/nas3/dkarageo/CASIAv1/Predictions/Splicebuster_gmm_num=10/manipulated_masks"
CASIA_V1_NOISEPRINT_AUTHENTIC_DIR: str = \
    "/nas3/dkarageo/CASIAv1/Predictions/Noiseprint/authentic_masks"
CASIA_V1_NOISEPRINT_MANIPULATED_DIR: str = \
    "/nas3/dkarageo/CASIAv1/Predictions/Noiseprint/manipulated_masks"
CASIA_V1_MANTRANET_AUTHENTIC_DIR: str = \
    "/nas3/dkarageo/CASIAv1/Predictions/Mantranet/authentic_masks"
CASIA_V1_MANTRANET_MANIPULATED_DIR: str = \
    "/nas3/dkarageo/CASIAv1/Predictions/Mantranet/manipulated_masks"
CASIA_V1_SPAN_AUTHENTIC_DIR: str = \
    "/nas3/dkarageo/CASIAv1/Predictions/SPAN/authentic_masks"
CASIA_V1_SPAN_MANIPULATED_DIR: str = \
    "/nas3/dkarageo/CASIAv1/Predictions/SPAN/manipulated_masks"
CASIA_V1_FUSION_AUTHENTIC_DIR: str = \
    "/nas3/dkarageo/CASIAv1/Predictions/Fusion/authentic_masks"
CASIA_V1_FUSION_MANIPULATED_DIR: str = \
    "/nas3/dkarageo/CASIAv1/Predictions/Fusion/manipulated_masks"
CASIA_V1_ADQ2_AUTHENTIC_DIR: str = \
    "/nas3/dkarageo/CASIAv1/Predictions/ADQ2/authentic_masks"
CASIA_V1_ADQ2_MANIPULATED_DIR: str = \
    "/nas3/dkarageo/CASIAv1/Predictions/ADQ2/manipulated_masks"
CASIA_V1_CFA_AUTHENTIC_DIR: str = \
    "/nas3/dkarageo/CASIAv1/Predictions/CFA/authentic_masks"
CASIA_V1_CFA_MANIPULATED_DIR: str = \
    "/nas3/dkarageo/CASIAv1/Predictions/CFA/manipulated_masks"
CASIA_V1_CMFD_AUTHENTIC_DIR: str = \
    "/nas3/dkarageo/CASIAv1/Predictions/CMFD/authentic_masks"
CASIA_V1_CMFD_MANIPULATED_DIR: str = \
    "/nas3/dkarageo/CASIAv1/Predictions/CMFD/manipulated_masks"
CASIA_V1_WAVELET_AUTHENTIC_DIR: str = \
    "/nas3/dkarageo/CASIAv1/Predictions/Wavelet/authentic_masks"
CASIA_V1_WAVELET_MANIPULATED_DIR: str = \
    "/nas3/dkarageo/CASIAv1/Predictions/Wavelet/manipulated_masks"
CASIA_V1_ZERO_AUTHENTIC_DIR: str = \
    "/nas3/dkarageo/CASIAv1/Predictions/Zero/authentic_masks"
CASIA_V1_ZERO_MANIPULATED_DIR: str = \
    "/nas3/dkarageo/CASIAv1/Predictions/Zero/manipulated_masks"

CASIA_V2_AUTHENTIC_DIR: str = "/nas3/dkarageo/CASIAv2/Au_single"
CASIA_V2_MANIPULATED_DIR: str = "/nas3/dkarageo/CASIAv2/Tp"
CASIA_V2_MASKS_DIR: str = "/nas3/dkarageo/CASIAv2/Masks"
CASIA_V2_EVA_AUTHENTIC_DIR: str = "/nas3/dkarageo/CASIAv2/Predictions/EVA/authentic_masks"
CASIA_V2_EVA_MANIPULATED_DIR: str = "/nas3/dkarageo/CASIAv2/Predictions/EVA/manipulated_masks"
CASIA_V2_ADQ1_AUTHENTIC_DIR: str = "/nas3/dkarageo/CASIAv2/Predictions/ADQ1/authentic_masks"
CASIA_V2_ADQ1_MANIPULATED_DIR: str = "/nas3/dkarageo/CASIAv2/Predictions/ADQ1/manipulated_masks"
CASIA_V2_BLK_AUTHENTIC_DIR: str = "/nas3/dkarageo/CASIAv2/Predictions/BLK/authentic_masks"
CASIA_V2_BLK_MANIPULATED_DIR: str = "/nas3/dkarageo/CASIAv2/Predictions/BLK/manipulated_masks"
CASIA_V2_CAGI_AUTHENTIC_DIR: str = "/nas3/dkarageo/CASIAv2/Predictions/CAGI/authentic_masks"
CASIA_V2_CAGI_MANIPULATED_DIR: str = "/nas3/dkarageo/CASIAv2/Predictions/CAGI/manipulated_masks"
CASIA_V2_DCT_AUTHENTIC_DIR: str = "/nas3/dkarageo/CASIAv2/Predictions/DCT/authentic_masks"
CASIA_V2_DCT_MANIPULATED_DIR: str = "/nas3/dkarageo/CASIAv2/Predictions/DCT/manipulated_masks"
CASIA_V2_SPLICEBUSTER_AUTHENTIC_DIR: str = \
    "/nas3/dkarageo/CASIAv2/Predictions/Splicebuster_gmm_num=10/authentic_masks"
CASIA_V2_SPLICEBUSTER_MANIPULATED_DIR: str = \
    "/nas3/dkarageo/CASIAv2/Predictions/Splicebuster_gmm_num=10/manipulated_masks"
CASIA_V2_NOISEPRINT_AUTHENTIC_DIR: str = \
    "/nas3/dkarageo/CASIAv2/Predictions/Noiseprint/authentic_masks"
CASIA_V2_NOISEPRINT_MANIPULATED_DIR: str = \
    "/nas3/dkarageo/CASIAv2/Predictions/Noiseprint/manipulated_masks"
CASIA_V2_MANTRANET_AUTHENTIC_DIR: str = \
    "/nas3/dkarageo/CASIAv2/Predictions/Mantranet/authentic_masks"
CASIA_V2_MANTRANET_MANIPULATED_DIR: str = \
    "/nas3/dkarageo/CASIAv2/Predictions/Mantranet/manipulated_masks"
CASIA_V2_SPAN_AUTHENTIC_DIR: str = \
    "/nas3/dkarageo/CASIAv2/Predictions/SPAN/authentic_masks"
CASIA_V2_SPAN_MANIPULATED_DIR: str = \
    "/nas3/dkarageo/CASIAv2/Predictions/SPAN/manipulated_masks"
CASIA_V2_FUSION_AUTHENTIC_DIR: str = \
    "/nas3/dkarageo/CASIAv2/Predictions/Fusion/authentic_masks"
CASIA_V2_FUSION_MANIPULATED_DIR: str = \
    "/nas3/dkarageo/CASIAv2/Predictions/Fusion/manipulated_masks"
CASIA_V2_ADQ2_AUTHENTIC_DIR: str = \
    "/nas3/dkarageo/CASIAv2/Predictions/ADQ2/authentic_masks"
CASIA_V2_ADQ2_MANIPULATED_DIR: str = \
    "/nas3/dkarageo/CASIAv2/Predictions/ADQ2/manipulated_masks"
CASIA_V2_CFA_AUTHENTIC_DIR: str = \
    "/nas3/dkarageo/CASIAv2/Predictions/CFA/authentic_masks"
CASIA_V2_CFA_MANIPULATED_DIR: str = \
    "/nas3/dkarageo/CASIAv2/Predictions/CFA/manipulated_masks"
CASIA_V2_CMFD_AUTHENTIC_DIR: str = \
    "/nas3/dkarageo/CASIAv2/Predictions/CMFD/authentic_masks"
CASIA_V2_CMFD_MANIPULATED_DIR: str = \
    "/nas3/dkarageo/CASIAv2/Predictions/CMFD/manipulated_masks"
CASIA_V2_WAVELET_AUTHENTIC_DIR: str = \
    "/nas3/dkarageo/CASIAv2/Predictions/Wavelet/authentic_masks"
CASIA_V2_WAVELET_MANIPULATED_DIR: str = \
    "/nas3/dkarageo/CASIAv2/Predictions/Wavelet/manipulated_masks"
CASIA_V2_ZERO_AUTHENTIC_DIR: str = \
    "/nas3/dkarageo/CASIAv2/Predictions/Zero/authentic_masks"
CASIA_V2_ZERO_MANIPULATED_DIR: str = \
    "/nas3/dkarageo/CASIAv2/Predictions/Zero/manipulated_masks"

ALGORITHMS: list[str] = ["eva", "eva_coco", "eva_elvis", "eva_raw",
                         "adq1", "blk", "cagi", "dct", "splicebuster",
                         "noiseprint", "mantranet", "span", "fusion",
                         "adq2", "cfa", "cmfd", "wavelet", "zero"]


ROOT_DIR: str = "/nas3/dkarageo/"
OUT_CSV: str = "/nas3/dkarageo/casia_fusion.csv"


def load_test_authentic_samples() -> list[dict[str, Union[Path, bool]]]:
    # Load CASIAv1 authentic samples.
    samples: list[dict[str, Union[Path, bool]]] = []
    eva_dir: Path = Path(CASIA_V1_EVA_AUTHENTIC_DIR)
    adq1_dir: Path = Path(CASIA_V1_ADQ1_AUTHENTIC_DIR)
    blk_dir: Path = Path(CASIA_V1_BLK_AUTHENTIC_DIR)
    cagi_dir: Path = Path(CASIA_V1_CAGI_AUTHENTIC_DIR)
    dct_dir: Path = Path(CASIA_V1_DCT_AUTHENTIC_DIR)
    splicebuster_dir: Path = Path(CASIA_V1_SPLICEBUSTER_AUTHENTIC_DIR)
    noiseprint_dir: Path = Path(CASIA_V1_NOISEPRINT_AUTHENTIC_DIR)
    mantranet_dir: Path = Path(CASIA_V1_MANTRANET_AUTHENTIC_DIR)
    span_dir: Path = Path(CASIA_V1_SPAN_AUTHENTIC_DIR)
    fusion_dir: Path = Path(CASIA_V1_FUSION_AUTHENTIC_DIR)
    adq2_dir: Path = Path(CASIA_V1_ADQ2_AUTHENTIC_DIR)
    cfa_dir: Path = Path(CASIA_V1_CFA_AUTHENTIC_DIR)
    cmfd_dir: Path = Path(CASIA_V1_CMFD_AUTHENTIC_DIR)
    wavelet_dir: Path = Path(CASIA_V1_WAVELET_AUTHENTIC_DIR)
    zero_dir: Path = Path(CASIA_V1_ZERO_AUTHENTIC_DIR)
    for p in Path(CASIA_V1_AUTHENTIC_DIR).iterdir():
        eva_mask: Path = eva_dir / f"{p.stem}_map.png"
        eva_coco_mask: Path = eva_dir / f"{p.stem}_coco_map.png"
        eva_elvis_mask: Path = eva_dir / f"{p.stem}_elvis_map.png"
        eva_raw_masks: Path = eva_dir / p.stem / "segmentation_instances.csv"
        adq1_mask: Path = adq1_dir / f"{p.stem}.png"
        blk_mask: Path = blk_dir / f"{p.stem}.png"
        cagi_mask: Path = cagi_dir / f"{p.stem}.png"
        dct_mask: Path = dct_dir / f"{p.stem}.png"
        sb_mask: Path = splicebuster_dir / f"{p.stem}.png"
        noiseprint_mask: Path = noiseprint_dir / f"{p.stem}.png"
        mantranet_mask: Path = mantranet_dir / f"{p.stem}.png"
        span_mask: Path = span_dir / f"{p.stem}.png"
        fusion_mask: Path = fusion_dir / f"{p.stem}.png"
        adq2_mask: Path = adq2_dir / f"{p.stem}.png"
        cfa_mask: Path = cfa_dir / f"{p.stem}.png"
        cmfd_mask: Path = cmfd_dir / f"{p.stem}.png"
        wavelet_mask: Path = wavelet_dir / f"{p.stem}.png"
        zero_mask: Path = zero_dir / f"{p.stem}.png"

        sample = {
            "image": str(p.absolute().relative_to(Path(ROOT_DIR))),
            "eva":
                str(eva_mask.absolute().relative_to(Path(ROOT_DIR))) if eva_mask.exists() else "",
            "eva_coco":
                str(eva_coco_mask.absolute().relative_to(Path(ROOT_DIR)))
                if eva_coco_mask.exists() else "",
            "eva_elvis":
                str(eva_elvis_mask.absolute().relative_to(Path(ROOT_DIR)))
                if eva_elvis_mask.exists() else "",
            "eva_raw":
                str(eva_raw_masks.absolute().relative_to(Path(ROOT_DIR)))
                if eva_raw_masks.exists() else "",
            "adq1": str(
                adq1_mask.absolute().relative_to(Path(ROOT_DIR))) if adq1_mask.exists() else "",
            "blk": str(
                blk_mask.absolute().relative_to(Path(ROOT_DIR))) if blk_mask.exists() else "",
            "cagi": str(
                cagi_mask.absolute().relative_to(Path(ROOT_DIR))) if cagi_mask.exists() else "",
            "dct": str(
                dct_mask.absolute().relative_to(Path(ROOT_DIR))) if dct_mask.exists() else "",
            "splicebuster": str(
                sb_mask.absolute().relative_to(Path(ROOT_DIR))) if sb_mask.exists() else "",
            "noiseprint": str(
                noiseprint_mask.absolute().relative_to(Path(ROOT_DIR)))
                if noiseprint_mask.exists() else "",
            "mantranet": str(
                mantranet_mask.absolute().relative_to(Path(ROOT_DIR)))
                if mantranet_mask.exists() else "",
            "span": str(
                span_mask.absolute().relative_to(Path(ROOT_DIR))) if span_mask.exists() else "",
            "fusion": str(
                fusion_mask.absolute().relative_to(Path(ROOT_DIR))) if fusion_mask.exists() else "",
            "adq2": str(
                adq2_mask.absolute().relative_to(Path(ROOT_DIR))) if adq2_mask.exists() else "",
            "cfa": str(
                cfa_mask.absolute().relative_to(Path(ROOT_DIR))) if cfa_mask.exists() else "",
            "cmfd": str(
                cmfd_mask.absolute().relative_to(Path(ROOT_DIR))) if cmfd_mask.exists() else "",
            "wavelet": str(
                wavelet_mask.absolute().relative_to(Path(ROOT_DIR)))
                if wavelet_mask.exists() else "",
            "zero": str(
                zero_mask.absolute().relative_to(Path(ROOT_DIR))) if zero_mask.exists() else "",
            "mask": "",
            "detection": False,  # Sample is authentic.
            "split": "test"
        }
        samples.append(sample)
    print(f"TEST SAMPLES AUTHENTIC: {len(samples)}")
    return samples


def load_test_manipulated_samples() -> list[dict[str, Union[Path, bool]]]:
    # Load CASIAv1 manipulated samples.
    samples: list[dict[str, Union[Path, bool]]] = []
    test_masks_dir: Path = Path(CASIA_V1_MASKS_DIR)
    eva_dir: Path = Path(CASIA_V1_EVA_MANIPULATED_DIR)
    adq1_dir: Path = Path(CASIA_V1_ADQ1_MANIPULATED_DIR)
    blk_dir: Path = Path(CASIA_V1_BLK_MANIPULATED_DIR)
    cagi_dir: Path = Path(CASIA_V1_CAGI_MANIPULATED_DIR)
    dct_dir: Path = Path(CASIA_V1_DCT_MANIPULATED_DIR)
    splicebuster_dir: Path = Path(CASIA_V1_SPLICEBUSTER_MANIPULATED_DIR)
    noiseprint_dir: Path = Path(CASIA_V1_NOISEPRINT_MANIPULATED_DIR)
    mantranet_dir: Path = Path(CASIA_V1_MANTRANET_MANIPULATED_DIR)
    span_dir: Path = Path(CASIA_V1_SPAN_MANIPULATED_DIR)
    fusion_dir: Path = Path(CASIA_V1_FUSION_MANIPULATED_DIR)
    adq2_dir: Path = Path(CASIA_V1_ADQ2_MANIPULATED_DIR)
    cfa_dir: Path = Path(CASIA_V1_CFA_MANIPULATED_DIR)
    cmfd_dir: Path = Path(CASIA_V1_CMFD_MANIPULATED_DIR)
    wavelet_dir: Path = Path(CASIA_V1_WAVELET_MANIPULATED_DIR)
    zero_dir: Path = Path(CASIA_V1_ZERO_MANIPULATED_DIR)
    for p in Path(CASIA_V1_MANIPULATED_DIR).iterdir():
        eva_mask: Path = eva_dir / f"{p.stem}_map.png"
        eva_coco_mask: Path = eva_dir / f"{p.stem}_coco_map.png"
        eva_elvis_mask: Path = eva_dir / f"{p.stem}_elvis_map.png"
        eva_raw_masks: Path = eva_dir / p.stem / "segmentation_instances.csv"
        adq1_mask: Path = adq1_dir / f"{p.stem}.png"
        blk_mask: Path = blk_dir / f"{p.stem}.png"
        cagi_mask: Path = cagi_dir / f"{p.stem}.png"
        dct_mask: Path = dct_dir / f"{p.stem}.png"
        sb_mask: Path = splicebuster_dir / f"{p.stem}.png"
        noiseprint_mask: Path = noiseprint_dir / f"{p.stem}.png"
        mantranet_mask: Path = mantranet_dir / f"{p.stem}.png"
        span_mask: Path = span_dir / f"{p.stem}.png"
        fusion_mask: Path = fusion_dir / f"{p.stem}.png"
        adq2_mask: Path = adq2_dir / f"{p.stem}.png"
        cfa_mask: Path = cfa_dir / f"{p.stem}.png"
        cmfd_mask: Path = cmfd_dir / f"{p.stem}.png"
        wavelet_mask: Path = wavelet_dir / f"{p.stem}.png"
        zero_mask: Path = zero_dir / f"{p.stem}.png"

        sample = {
            "image": str(p.absolute().relative_to(Path(ROOT_DIR))),
            "eva":
                str(eva_mask.absolute().relative_to(Path(ROOT_DIR))) if eva_mask.exists() else "",
            "eva_coco":
                str(eva_coco_mask.absolute().relative_to(Path(ROOT_DIR)))
                if eva_coco_mask.exists() else "",
            "eva_elvis":
                str(eva_elvis_mask.absolute().relative_to(Path(ROOT_DIR)))
                if eva_elvis_mask.exists() else "",
            "eva_raw":
                str(eva_raw_masks.absolute().relative_to(Path(ROOT_DIR)))
                if eva_raw_masks.exists() else "",
            "adq1": str(
                adq1_mask.absolute().relative_to(Path(ROOT_DIR))) if adq1_mask.exists() else "",
            "blk": str(
                blk_mask.absolute().relative_to(Path(ROOT_DIR))) if blk_mask.exists() else "",
            "cagi": str(
                cagi_mask.absolute().relative_to(Path(ROOT_DIR))) if cagi_mask.exists() else "",
            "dct": str(
                dct_mask.absolute().relative_to(Path(ROOT_DIR))) if dct_mask.exists() else "",
            "splicebuster": str(
                sb_mask.absolute().relative_to(Path(ROOT_DIR))) if sb_mask.exists() else "",
            "noiseprint": str(
                noiseprint_mask.absolute().relative_to(Path(ROOT_DIR)))
            if noiseprint_mask.exists() else "",
            "mantranet": str(
                mantranet_mask.absolute().relative_to(Path(ROOT_DIR)))
            if mantranet_mask.exists() else "",
            "span": str(
                span_mask.absolute().relative_to(Path(ROOT_DIR))) if span_mask.exists() else "",
            "fusion": str(
                fusion_mask.absolute().relative_to(Path(ROOT_DIR))) if fusion_mask.exists() else "",
            "adq2": str(
                adq2_mask.absolute().relative_to(Path(ROOT_DIR))) if adq2_mask.exists() else "",
            "cfa": str(
                cfa_mask.absolute().relative_to(Path(ROOT_DIR))) if cfa_mask.exists() else "",
            "cmfd": str(
                cmfd_mask.absolute().relative_to(Path(ROOT_DIR))) if cmfd_mask.exists() else "",
            "wavelet": str(
                wavelet_mask.absolute().relative_to(Path(ROOT_DIR)))
            if wavelet_mask.exists() else "",
            "zero": str(
                zero_mask.absolute().relative_to(Path(ROOT_DIR))) if zero_mask.exists() else "",
            "mask": str((test_masks_dir / f"{p.stem}.png").absolute().relative_to(Path(ROOT_DIR))),
            "detection": True,  # Sample is manipulated.
            "split": "test"
        }
        assert (Path(ROOT_DIR) / sample["mask"]).exists()
        samples.append(sample)
    print(f"TEST SAMPLES MANIPULATED: {len(samples)}")
    return samples


def load_train_val_authentic_samples() -> list[dict[str, Union[Path, bool]]]:
    # Load CASIAv2 authentic samples.
    authentic_paths: list[Path] = [p for p in Path(CASIA_V2_AUTHENTIC_DIR).iterdir()
                                   if filetype.is_image(p)]
    train_paths, val_paths = train_test_split(authentic_paths, train_size=0.85, random_state=123)

    samples: list[dict[str, Union[Path, bool]]] = []
    eva_dir: Path = Path(CASIA_V2_EVA_AUTHENTIC_DIR)
    adq1_dir: Path = Path(CASIA_V2_ADQ1_AUTHENTIC_DIR)
    blk_dir: Path = Path(CASIA_V2_BLK_AUTHENTIC_DIR)
    cagi_dir: Path = Path(CASIA_V2_CAGI_AUTHENTIC_DIR)
    dct_dir: Path = Path(CASIA_V2_DCT_AUTHENTIC_DIR)
    splicebuster_dir: Path = Path(CASIA_V2_SPLICEBUSTER_AUTHENTIC_DIR)
    noiseprint_dir: Path = Path(CASIA_V2_NOISEPRINT_AUTHENTIC_DIR)
    mantranet_dir: Path = Path(CASIA_V2_MANTRANET_AUTHENTIC_DIR)
    span_dir: Path = Path(CASIA_V2_SPAN_AUTHENTIC_DIR)
    fusion_dir: Path = Path(CASIA_V2_FUSION_AUTHENTIC_DIR)
    adq2_dir: Path = Path(CASIA_V2_ADQ2_AUTHENTIC_DIR)
    cfa_dir: Path = Path(CASIA_V2_CFA_AUTHENTIC_DIR)
    cmfd_dir: Path = Path(CASIA_V2_CMFD_AUTHENTIC_DIR)
    wavelet_dir: Path = Path(CASIA_V2_WAVELET_AUTHENTIC_DIR)
    zero_dir: Path = Path(CASIA_V2_ZERO_AUTHENTIC_DIR)
    for p in train_paths:
        eva_mask: Path = eva_dir / f"{p.stem}_map.png"
        eva_coco_mask: Path = eva_dir / f"{p.stem}_coco_map.png"
        eva_elvis_mask: Path = eva_dir / f"{p.stem}_elvis_map.png"
        eva_raw_masks: Path = eva_dir / p.stem / "segmentation_instances.csv"
        adq1_mask: Path = adq1_dir / f"{p.stem}.png"
        blk_mask: Path = blk_dir / f"{p.stem}.png"
        cagi_mask: Path = cagi_dir / f"{p.stem}.png"
        dct_mask: Path = dct_dir / f"{p.stem}.png"
        sb_mask: Path = splicebuster_dir / f"{p.stem}.png"
        noiseprint_mask: Path = noiseprint_dir / f"{p.stem}.png"
        mantranet_mask: Path = mantranet_dir / f"{p.stem}.png"
        span_mask: Path = span_dir / f"{p.stem}.png"
        fusion_mask: Path = fusion_dir / f"{p.stem}.png"
        adq2_mask: Path = adq2_dir / f"{p.stem}.png"
        cfa_mask: Path = cfa_dir / f"{p.stem}.png"
        cmfd_mask: Path = cmfd_dir / f"{p.stem}.png"
        wavelet_mask: Path = wavelet_dir / f"{p.stem}.png"
        zero_mask: Path = zero_dir / f"{p.stem}.png"

        sample = {
            "image": str(p.absolute().relative_to(Path(ROOT_DIR))),
            "eva":
                str(eva_mask.absolute().relative_to(Path(ROOT_DIR))) if eva_mask.exists() else "",
            "eva_coco":
                str(eva_coco_mask.absolute().relative_to(Path(ROOT_DIR)))
                if eva_coco_mask.exists() else "",
            "eva_elvis":
                str(eva_elvis_mask.absolute().relative_to(Path(ROOT_DIR)))
                if eva_elvis_mask.exists() else "",
            "eva_raw":
                str(eva_raw_masks.absolute().relative_to(Path(ROOT_DIR)))
                if eva_raw_masks.exists() else "",
            "adq1": str(
                adq1_mask.absolute().relative_to(Path(ROOT_DIR))) if adq1_mask.exists() else "",
            "blk": str(
                blk_mask.absolute().relative_to(Path(ROOT_DIR))) if blk_mask.exists() else "",
            "cagi": str(
                cagi_mask.absolute().relative_to(Path(ROOT_DIR))) if cagi_mask.exists() else "",
            "dct": str(
                dct_mask.absolute().relative_to(Path(ROOT_DIR))) if dct_mask.exists() else "",
            "splicebuster": str(
                sb_mask.absolute().relative_to(Path(ROOT_DIR))) if sb_mask.exists() else "",
            "noiseprint": str(
                noiseprint_mask.absolute().relative_to(Path(ROOT_DIR)))
            if noiseprint_mask.exists() else "",
            "mantranet": str(
                mantranet_mask.absolute().relative_to(Path(ROOT_DIR)))
            if mantranet_mask.exists() else "",
            "span": str(
                span_mask.absolute().relative_to(Path(ROOT_DIR))) if span_mask.exists() else "",
            "fusion": str(
                fusion_mask.absolute().relative_to(Path(ROOT_DIR))) if fusion_mask.exists() else "",
            "adq2": str(
                adq2_mask.absolute().relative_to(Path(ROOT_DIR))) if adq2_mask.exists() else "",
            "cfa": str(
                cfa_mask.absolute().relative_to(Path(ROOT_DIR))) if cfa_mask.exists() else "",
            "cmfd": str(
                cmfd_mask.absolute().relative_to(Path(ROOT_DIR))) if cmfd_mask.exists() else "",
            "wavelet": str(
                wavelet_mask.absolute().relative_to(Path(ROOT_DIR)))
            if wavelet_mask.exists() else "",
            "zero": str(
                zero_mask.absolute().relative_to(Path(ROOT_DIR))) if zero_mask.exists() else "",
            "mask": "",
            "detection": False,  # Sample is authentic.
            "split": "train"
        }
        samples.append(sample)
    for p in val_paths:
        eva_mask: Path = eva_dir / f"{p.stem}_map.png"
        eva_coco_mask: Path = eva_dir / f"{p.stem}_coco_map.png"
        eva_elvis_mask: Path = eva_dir / f"{p.stem}_elvis_map.png"
        eva_raw_masks: Path = eva_dir / p.stem / "segmentation_instances.csv"
        adq1_mask: Path = adq1_dir / f"{p.stem}.png"
        blk_mask: Path = blk_dir / f"{p.stem}.png"
        cagi_mask: Path = cagi_dir / f"{p.stem}.png"
        dct_mask: Path = dct_dir / f"{p.stem}.png"
        sb_mask: Path = splicebuster_dir / f"{p.stem}.png"
        noiseprint_mask: Path = noiseprint_dir / f"{p.stem}.png"
        mantranet_mask: Path = mantranet_dir / f"{p.stem}.png"
        span_mask: Path = span_dir / f"{p.stem}.png"
        fusion_mask: Path = fusion_dir / f"{p.stem}.png"
        adq2_mask: Path = adq2_dir / f"{p.stem}.png"
        cfa_mask: Path = cfa_dir / f"{p.stem}.png"
        cmfd_mask: Path = cmfd_dir / f"{p.stem}.png"
        wavelet_mask: Path = wavelet_dir / f"{p.stem}.png"
        zero_mask: Path = zero_dir / f"{p.stem}.png"

        sample = {
            "image": str(p.absolute().relative_to(Path(ROOT_DIR))),
            "eva":
                str(eva_mask.absolute().relative_to(Path(ROOT_DIR))) if eva_mask.exists() else "",
            "eva_coco":
                str(eva_coco_mask.absolute().relative_to(Path(ROOT_DIR)))
                if eva_coco_mask.exists() else "",
            "eva_elvis":
                str(eva_elvis_mask.absolute().relative_to(Path(ROOT_DIR)))
                if eva_elvis_mask.exists() else "",
            "eva_raw":
                str(eva_raw_masks.absolute().relative_to(Path(ROOT_DIR)))
                if eva_raw_masks.exists() else "",
            "adq1": str(
                adq1_mask.absolute().relative_to(Path(ROOT_DIR))) if adq1_mask.exists() else "",
            "blk": str(
                blk_mask.absolute().relative_to(Path(ROOT_DIR))) if blk_mask.exists() else "",
            "cagi": str(
                cagi_mask.absolute().relative_to(Path(ROOT_DIR))) if cagi_mask.exists() else "",
            "dct": str(
                dct_mask.absolute().relative_to(Path(ROOT_DIR))) if dct_mask.exists() else "",
            "splicebuster": str(
                sb_mask.absolute().relative_to(Path(ROOT_DIR))) if sb_mask.exists() else "",
            "noiseprint": str(
                noiseprint_mask.absolute().relative_to(Path(ROOT_DIR)))
            if noiseprint_mask.exists() else "",
            "mantranet": str(
                mantranet_mask.absolute().relative_to(Path(ROOT_DIR)))
            if mantranet_mask.exists() else "",
            "span": str(
                span_mask.absolute().relative_to(Path(ROOT_DIR))) if span_mask.exists() else "",
            "fusion": str(
                fusion_mask.absolute().relative_to(Path(ROOT_DIR))) if fusion_mask.exists() else "",
            "adq2": str(
                adq2_mask.absolute().relative_to(Path(ROOT_DIR))) if adq2_mask.exists() else "",
            "cfa": str(
                cfa_mask.absolute().relative_to(Path(ROOT_DIR))) if cfa_mask.exists() else "",
            "cmfd": str(
                cmfd_mask.absolute().relative_to(Path(ROOT_DIR))) if cmfd_mask.exists() else "",
            "wavelet": str(
                wavelet_mask.absolute().relative_to(Path(ROOT_DIR)))
            if wavelet_mask.exists() else "",
            "zero": str(
                zero_mask.absolute().relative_to(Path(ROOT_DIR))) if zero_mask.exists() else "",
            "mask": "",
            "detection": False,  # Sample is authentic.
            "split": "eval"
        }
        samples.append(sample)
    print(f"TRAIN-VAL SAMPLES AUTHENTIC: {len(samples)}")
    return samples


def load_train_val_manipulated_samples() -> list[dict[str, Union[Path, bool]]]:
    # Load CASIAv2 manipulated samples.
    paths: list[Path] = [p for p in Path(CASIA_V2_MANIPULATED_DIR).iterdir()
                         if filetype.is_image(p)]
    train_paths, val_paths = train_test_split(paths, train_size=0.85, random_state=123)

    samples: list[dict[str, Union[Path, bool]]] = []
    masks_dir: Path = Path(CASIA_V2_MASKS_DIR)
    eva_dir: Path = Path(CASIA_V2_EVA_MANIPULATED_DIR)
    adq1_dir: Path = Path(CASIA_V2_ADQ1_MANIPULATED_DIR)
    blk_dir: Path = Path(CASIA_V2_BLK_MANIPULATED_DIR)
    cagi_dir: Path = Path(CASIA_V2_CAGI_MANIPULATED_DIR)
    dct_dir: Path = Path(CASIA_V2_DCT_MANIPULATED_DIR)
    splicebuster_dir: Path = Path(CASIA_V2_SPLICEBUSTER_MANIPULATED_DIR)
    noiseprint_dir: Path = Path(CASIA_V2_NOISEPRINT_MANIPULATED_DIR)
    mantranet_dir: Path = Path(CASIA_V2_MANTRANET_MANIPULATED_DIR)
    span_dir: Path = Path(CASIA_V2_SPAN_MANIPULATED_DIR)
    fusion_dir: Path = Path(CASIA_V2_FUSION_MANIPULATED_DIR)
    adq2_dir: Path = Path(CASIA_V2_ADQ2_MANIPULATED_DIR)
    cfa_dir: Path = Path(CASIA_V2_CFA_MANIPULATED_DIR)
    cmfd_dir: Path = Path(CASIA_V2_CMFD_MANIPULATED_DIR)
    wavelet_dir: Path = Path(CASIA_V2_WAVELET_MANIPULATED_DIR)
    zero_dir: Path = Path(CASIA_V2_ZERO_MANIPULATED_DIR)
    for p in train_paths:
        eva_mask: Path = eva_dir / f"{p.stem}_map.png"
        eva_coco_mask: Path = eva_dir / f"{p.stem}_coco_map.png"
        eva_elvis_mask: Path = eva_dir / f"{p.stem}_elvis_map.png"
        eva_raw_masks: Path = eva_dir / p.stem / "segmentation_instances.csv"
        adq1_mask: Path = adq1_dir / f"{p.stem}.png"
        blk_mask: Path = blk_dir / f"{p.stem}.png"
        cagi_mask: Path = cagi_dir / f"{p.stem}.png"
        dct_mask: Path = dct_dir / f"{p.stem}.png"
        sb_mask: Path = splicebuster_dir / f"{p.stem}.png"
        noiseprint_mask: Path = noiseprint_dir / f"{p.stem}.png"
        mantranet_mask: Path = mantranet_dir / f"{p.stem}.png"
        span_mask: Path = span_dir / f"{p.stem}.png"
        fusion_mask: Path = fusion_dir / f"{p.stem}.png"
        adq2_mask: Path = adq2_dir / f"{p.stem}.png"
        cfa_mask: Path = cfa_dir / f"{p.stem}.png"
        cmfd_mask: Path = cmfd_dir / f"{p.stem}.png"
        wavelet_mask: Path = wavelet_dir / f"{p.stem}.png"
        zero_mask: Path = zero_dir / f"{p.stem}.png"

        sample = {
            "image": str(p.absolute().relative_to(Path(ROOT_DIR))),
            "eva":
                str(eva_mask.absolute().relative_to(Path(ROOT_DIR))) if eva_mask.exists() else "",
            "eva_coco":
                str(eva_coco_mask.absolute().relative_to(Path(ROOT_DIR)))
                if eva_coco_mask.exists() else "",
            "eva_elvis":
                str(eva_elvis_mask.absolute().relative_to(Path(ROOT_DIR)))
                if eva_elvis_mask.exists() else "",
            "eva_raw":
                str(eva_raw_masks.absolute().relative_to(Path(ROOT_DIR)))
                if eva_raw_masks.exists() else "",
            "adq1": str(
                adq1_mask.absolute().relative_to(Path(ROOT_DIR))) if adq1_mask.exists() else "",
            "blk": str(
                blk_mask.absolute().relative_to(Path(ROOT_DIR))) if blk_mask.exists() else "",
            "cagi": str(
                cagi_mask.absolute().relative_to(Path(ROOT_DIR))) if cagi_mask.exists() else "",
            "dct": str(
                dct_mask.absolute().relative_to(Path(ROOT_DIR))) if dct_mask.exists() else "",
            "splicebuster": str(
                sb_mask.absolute().relative_to(Path(ROOT_DIR))) if sb_mask.exists() else "",
            "noiseprint": str(
                noiseprint_mask.absolute().relative_to(Path(ROOT_DIR)))
            if noiseprint_mask.exists() else "",
            "mantranet": str(
                mantranet_mask.absolute().relative_to(Path(ROOT_DIR)))
            if mantranet_mask.exists() else "",
            "span": str(
                span_mask.absolute().relative_to(Path(ROOT_DIR))) if span_mask.exists() else "",
            "fusion": str(
                fusion_mask.absolute().relative_to(Path(ROOT_DIR))) if fusion_mask.exists() else "",
            "adq2": str(
                adq2_mask.absolute().relative_to(Path(ROOT_DIR))) if adq2_mask.exists() else "",
            "cfa": str(
                cfa_mask.absolute().relative_to(Path(ROOT_DIR))) if cfa_mask.exists() else "",
            "cmfd": str(
                cmfd_mask.absolute().relative_to(Path(ROOT_DIR))) if cmfd_mask.exists() else "",
            "wavelet": str(
                wavelet_mask.absolute().relative_to(Path(ROOT_DIR)))
            if wavelet_mask.exists() else "",
            "zero": str(
                zero_mask.absolute().relative_to(Path(ROOT_DIR))) if zero_mask.exists() else "",
            "mask": str((masks_dir / f"{p.stem}.png").absolute().relative_to(Path(ROOT_DIR))),
            "detection": True,  # Sample is authentic.
            "split": "train"
        }
        samples.append(sample)
    for p in val_paths:
        eva_mask: Path = eva_dir / f"{p.stem}_map.png"
        eva_coco_mask: Path = eva_dir / f"{p.stem}_coco_map.png"
        eva_elvis_mask: Path = eva_dir / f"{p.stem}_elvis_map.png"
        eva_raw_masks: Path = eva_dir / p.stem / "segmentation_instances.csv"
        adq1_mask: Path = adq1_dir / f"{p.stem}.png"
        blk_mask: Path = blk_dir / f"{p.stem}.png"
        cagi_mask: Path = cagi_dir / f"{p.stem}.png"
        dct_mask: Path = dct_dir / f"{p.stem}.png"
        sb_mask: Path = splicebuster_dir / f"{p.stem}.png"
        noiseprint_mask: Path = noiseprint_dir / f"{p.stem}.png"
        mantranet_mask: Path = mantranet_dir / f"{p.stem}.png"
        span_mask: Path = span_dir / f"{p.stem}.png"
        fusion_mask: Path = fusion_dir / f"{p.stem}.png"
        adq2_mask: Path = adq2_dir / f"{p.stem}.png"
        cfa_mask: Path = cfa_dir / f"{p.stem}.png"
        cmfd_mask: Path = cmfd_dir / f"{p.stem}.png"
        wavelet_mask: Path = wavelet_dir / f"{p.stem}.png"
        zero_mask: Path = zero_dir / f"{p.stem}.png"

        sample = {
            "image": str(p.absolute().relative_to(Path(ROOT_DIR))),
            "eva":
                str(eva_mask.absolute().relative_to(Path(ROOT_DIR))) if eva_mask.exists() else "",
            "eva_coco":
                str(eva_coco_mask.absolute().relative_to(Path(ROOT_DIR)))
                if eva_coco_mask.exists() else "",
            "eva_elvis":
                str(eva_elvis_mask.absolute().relative_to(Path(ROOT_DIR)))
                if eva_elvis_mask.exists() else "",
            "eva_raw":
                str(eva_raw_masks.absolute().relative_to(Path(ROOT_DIR)))
                if eva_raw_masks.exists() else "",
            "adq1": str(
                adq1_mask.absolute().relative_to(Path(ROOT_DIR))) if adq1_mask.exists() else "",
            "blk": str(
                blk_mask.absolute().relative_to(Path(ROOT_DIR))) if blk_mask.exists() else "",
            "cagi": str(
                cagi_mask.absolute().relative_to(Path(ROOT_DIR))) if cagi_mask.exists() else "",
            "dct": str(
                dct_mask.absolute().relative_to(Path(ROOT_DIR))) if dct_mask.exists() else "",
            "splicebuster": str(
                sb_mask.absolute().relative_to(Path(ROOT_DIR))) if sb_mask.exists() else "",
            "noiseprint": str(
                noiseprint_mask.absolute().relative_to(Path(ROOT_DIR)))
            if noiseprint_mask.exists() else "",
            "mantranet": str(
                mantranet_mask.absolute().relative_to(Path(ROOT_DIR)))
            if mantranet_mask.exists() else "",
            "span": str(
                span_mask.absolute().relative_to(Path(ROOT_DIR))) if span_mask.exists() else "",
            "fusion": str(
                fusion_mask.absolute().relative_to(Path(ROOT_DIR))) if fusion_mask.exists() else "",
            "adq2": str(
                adq2_mask.absolute().relative_to(Path(ROOT_DIR))) if adq2_mask.exists() else "",
            "cfa": str(
                cfa_mask.absolute().relative_to(Path(ROOT_DIR))) if cfa_mask.exists() else "",
            "cmfd": str(
                cmfd_mask.absolute().relative_to(Path(ROOT_DIR))) if cmfd_mask.exists() else "",
            "wavelet": str(
                wavelet_mask.absolute().relative_to(Path(ROOT_DIR)))
            if wavelet_mask.exists() else "",
            "zero": str(
                zero_mask.absolute().relative_to(Path(ROOT_DIR))) if zero_mask.exists() else "",
            "mask": str((masks_dir / f"{p.stem}.png").absolute().relative_to(Path(ROOT_DIR))),
            "detection": True,  # Sample is authentic.
            "split": "eval"
        }
        assert (Path(ROOT_DIR) / sample["mask"]).exists()
        samples.append(sample)
    print(f"TRAIN-VAL SAMPLES MANIPULATED: {len(samples)}")
    return samples


def count_samples_with_all_algorithms(samples: list[dict[str, Union[Path, bool]]]) -> int:
    count: int = 0
    for s in samples:
        checks: list[bool] = [s[a] != "" for a in ALGORITHMS]
        if all(checks):
            count += 1
    return count


def main() -> None:
    print("ALGORITHMS:")
    for i, a in enumerate(ALGORITHMS):
        print(f"\t{i+1}.  {a}")

    data: list[dict[str, Union[Path, bool]]] = []
    data.extend(load_test_authentic_samples())
    data.extend(load_test_manipulated_samples())
    data.extend(load_train_val_authentic_samples())
    data.extend(load_train_val_manipulated_samples())

    print(f"SAMPLES WITH ALL ALGORITHMS: {count_samples_with_all_algorithms(data)}")

    # Export data to csv.
    fieldnames = ["image", "mask", "detection", "split"]
    fieldnames.extend(ALGORITHMS)
    csv_path: Path = Path(OUT_CSV)
    with csv_path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=",")
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"Exported csv to {str(csv_path.absolute())}")


if __name__ == "__main__":
    main()
