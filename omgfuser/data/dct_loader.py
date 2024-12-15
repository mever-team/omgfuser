"""
Created by Dimitrios Karageorgiou, email: dkarageo@iti.gr

Originally distributed under: https://github.com/mever-team/omgfuser

Copyright 2024 Media Analysis, Verification and Retrieval Group -
Information Technologies Institute - Centre for Research and Technology Hellas, Greece

Some pieces of code have been based on code from https://github.com/mjkwon2021/CAT-Net

This piece of code is licensed under the Apache License, Version 2.0.
A copy of the license can be found in the LICENSE file distributed together
with this file, as well as under https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under this repository is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the license for the specific language governing permissions and
limitations under the License.
"""

import os
import pathlib
import tempfile
import io
import shutil
from typing import Union, Optional, BinaryIO

from PIL import Image
import numpy as np
import jpegio  # See https://github.com/dwgoon/jpegio/blob/master/examples/jpegio_tutorial.ipynb
import torch
import random


def create_dct_volume(
    img_file: Union[pathlib.Path, io.BytesIO]
) -> tuple[torch.Tensor, torch.Tensor]:
    tmp_file: Optional[tempfile.TemporaryFile] = None

    try:
        if isinstance(img_file, pathlib.Path):
            with Image.open(img_file) as img:
                if img.format != "JPEG":
                    tmp_file = tempfile.NamedTemporaryFile(mode="w+b", delete=False, suffix=".jpg")
                    img.convert("RGB").save(tmp_file, quality=100, subsampling=0)
                    img_path: pathlib.Path = pathlib.Path(tmp_file.name)
                else:
                    img_path: pathlib.Path = img_file
        else:
            tmp_file = tempfile.NamedTemporaryFile(mode="w+b", delete=False, suffix=".jpg")
            with Image.open(img_file) as img:
                if img.format != "JPEG":
                    img.convert("RGB").save(tmp_file, quality=100, subsampling=0)
                else:
                    img_file.seek(0)
                    shutil.copyfileobj(img_file, tmp_file)
            img_path: pathlib.Path = pathlib.Path(tmp_file.name)

        dctvol = create_jpeg_dct_volume(img_path)
        qtable = torch.tensor(get_jpeg_qtable(img_path)[:1], dtype=torch.float)
    finally:
        if tmp_file is not None:
            tmp_file.close()
            os.unlink(tmp_file.name)

    return dctvol, qtable


def create_dct_volume_from_augmented(
    img_file: Union[pathlib.Path, io.BytesIO, BinaryIO],
    augmented: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a DCTVolume from images that have been augmented in some way.
    :param img_file: Original image file or stream.
    :param augmented: Augmented image array in range [0, 255].
    :return:
    """
    qtable_tmp_file: Optional[tempfile.TemporaryFile] = None
    dct_vol_tmp_file: Optional[tempfile.TemporaryFile] = None

    try:
        dct_vol_tmp_file = tempfile.NamedTemporaryFile(
            mode="w+b", delete=False, suffix=".jpg"
        )
        Image.fromarray(augmented.astype(np.uint8)).save(
            dct_vol_tmp_file, quality=100, subsampling=0
        )
        dctvol_img_path: pathlib.Path = pathlib.Path(dct_vol_tmp_file.name)
        dct_vol_tmp_file.flush()

        with Image.open(img_file) as img:
            if img.format != "JPEG":
                qtable_img_path: pathlib.Path = pathlib.Path(dct_vol_tmp_file.name)
            else:
                if isinstance(img_file, pathlib.Path):
                    qtable_img_path: pathlib.Path = img_file
                else:
                    qtable_tmp_file = tempfile.NamedTemporaryFile(mode="w+b", delete=False,
                                                                  suffix=".jpg")
                    img_file.seek(0)
                    shutil.copyfileobj(img_file, qtable_tmp_file)
                    qtable_tmp_file.flush()
                    qtable_img_path: pathlib.Path = pathlib.Path(qtable_tmp_file.name)
        dctvol = create_jpeg_dct_volume(dctvol_img_path)
        qtable = torch.tensor(
            get_jpeg_qtable(qtable_img_path)[0], dtype=torch.float
        ).unsqueeze(dim=0)
    finally:
        for tmp_file in [qtable_tmp_file, dct_vol_tmp_file]:
            if tmp_file is not None:
                tmp_file.close()
                os.unlink(tmp_file.name)

    return dctvol, qtable


def create_jpeg_dct_volume(
    img_path: pathlib.Path,
) -> torch.Tensor:

    with Image.open(img_path) as img:
        h, w = img.height, img.width

    dct_coef = get_dct_coef(img_path)

    # Smallest 8x8 grid crop that contains image.
    crop_size: tuple[int, int] = (-(-h//8) * 8, -(-w//8) * 8)

    # Pad if crop_size is larger than image size.
    if h < crop_size[0] or w < crop_size[1]:
        # pad dct_coef
        max_h = max(crop_size[0], max([dct_coef[c].shape[0] for c in range(1)]))
        max_w = max(crop_size[1], max([dct_coef[c].shape[1] for c in range(1)]))
        for i in range(1):
            temp = np.full((max_h, max_w), 0.0)  # pad with 0
            temp[:dct_coef[i].shape[0], :dct_coef[i].shape[1]] = dct_coef[i][:, :]
            dct_coef[i] = temp

    s_r = (random.randint(0, max(h - crop_size[0], 0)) // 8) * 8
    s_c = (random.randint(0, max(w - crop_size[1], 0)) // 8) * 8

    # crop dct_coef
    for i in range(1):
        dct_coef[i] = dct_coef[i][s_r:s_r+crop_size[0], s_c:s_c+crop_size[1]]
    t_dct_coef = torch.tensor(dct_coef[0], dtype=torch.float).unsqueeze(dim=0)  # final (but used below)

    # handle 'DCTvol'
    T = 20
    t_dct_vol = torch.zeros(size=(T+1, t_dct_coef.shape[1], t_dct_coef.shape[2]))
    t_dct_vol[0] += (t_dct_coef == 0).float().squeeze()
    for i in range(1, T):
        t_dct_vol[i] += (t_dct_coef == i).float().squeeze()
        t_dct_vol[i] += (t_dct_coef == -i).float().squeeze()
    t_dct_vol[T] += (t_dct_coef >= T).float().squeeze()
    t_dct_vol[T] += (t_dct_coef <= -T).float().squeeze()

    return t_dct_vol


def get_dct_coef(im_path):
    """
    :param im_path: JPEG image path
    :return: DCT_coef (Y)
    """
    num_channels = 1
    jpeg = jpegio.read(str(im_path))

    # determine which axes to up-sample
    ci = jpeg.comp_info
    need_scale = [[ci[i].v_samp_factor, ci[i].h_samp_factor] for i in range(num_channels)]
    if num_channels == 3:
        if ci[0].v_samp_factor == ci[1].v_samp_factor == ci[2].v_samp_factor:
            need_scale[0][0] = need_scale[1][0] = need_scale[2][0] = 2
        if ci[0].h_samp_factor == ci[1].h_samp_factor == ci[2].h_samp_factor:
            need_scale[0][1] = need_scale[1][1] = need_scale[2][1] = 2
    else:
        need_scale[0][0] = 2
        need_scale[0][1] = 2

    # up-sample DCT coefficients to match image size
    dct_coef = []
    for i in range(num_channels):
        r, c = jpeg.coef_arrays[i].shape
        coef_view = jpeg.coef_arrays[i].reshape(r//8, 8, c//8, 8).transpose(0, 2, 1, 3)
        # case 1: row scale (O) and col scale (O)
        if need_scale[i][0] == 1 and need_scale[i][1] == 1:
            out_arr = np.zeros((r * 2, c * 2))
            out_view = out_arr.reshape(r * 2 // 8, 8, c * 2 // 8, 8).transpose(0, 2, 1, 3)
            out_view[::2, ::2, :, :] = coef_view[:, :, :, :]
            out_view[1::2, ::2, :, :] = coef_view[:, :, :, :]
            out_view[::2, 1::2, :, :] = coef_view[:, :, :, :]
            out_view[1::2, 1::2, :, :] = coef_view[:, :, :, :]

        # case 2: row scale (O) and col scale (X)
        elif need_scale[i][0]==1 and need_scale[i][1]==2:
            out_arr = np.zeros((r * 2, c))
            dct_coef.append(out_arr)
            out_view = out_arr.reshape(r*2//8, 8, c // 8, 8).transpose(0, 2, 1, 3)
            out_view[::2, :, :, :] = coef_view[:, :, :, :]
            out_view[1::2, :, :, :] = coef_view[:, :, :, :]

        # case 3: row scale (X) and col scale (O)
        elif need_scale[i][0]==2 and need_scale[i][1]==1:
            out_arr = np.zeros((r, c * 2))
            out_view = out_arr.reshape(r // 8, 8, c * 2 // 8, 8).transpose(0, 2, 1, 3)
            out_view[:, ::2, :, :] = coef_view[:, :, :, :]
            out_view[:, 1::2, :, :] = coef_view[:, :, :, :]

        # case 4: row scale (X) and col scale (X)
        elif need_scale[i][0]==2 and need_scale[i][1]==2:
            out_arr = np.zeros((r, c))
            out_view = out_arr.reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3)
            out_view[:, :, :, :] = coef_view[:, :, :, :]

        else:
            raise KeyError("Something wrong here.")

        dct_coef.append(out_arr)

    return dct_coef


def get_jpeg_qtable(im_path):
    """
    :param im_path: JPEG image path
    :return: qtables (Y)
    """
    num_channels = 1
    jpeg = jpegio.read(str(im_path))

    ci = jpeg.comp_info

    # quantization tables
    qtables = [jpeg.quant_tables[ci[i].quant_tbl_no].astype(float) for i in range(num_channels)]

    return qtables
