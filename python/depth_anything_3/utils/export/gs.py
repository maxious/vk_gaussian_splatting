# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional
import torch

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.gsply_helpers import save_gaussian_ply


def export_to_gs_ply(
    prediction: Prediction,
    export_dir: str,
    gs_views_interval: Optional[
        int
    ] = 1,  # export GS every N views, useful for extremely dense inputs
):
    gs_world = prediction.gaussians
    pred_depth = torch.from_numpy(prediction.depth).unsqueeze(-1).to(gs_world.means)  # v h w 1
    idx = 0
    os.makedirs(os.path.join(export_dir, "gs_ply"), exist_ok=True)
    save_path = os.path.join(export_dir, f"gs_ply/{idx:04d}.ply")
    if gs_views_interval is None:  # select around 12 views in total
        gs_views_interval = max(pred_depth.shape[0] // 12, 1)
    save_gaussian_ply(
        gaussians=gs_world,
        save_path=save_path,
        ctx_depth=pred_depth,
        shift_and_scale=False,
        save_sh_dc_only=True,
        gs_views_interval=gs_views_interval,
        inv_opacity=True,
        prune_by_depth_percent=0.9,
        prune_border_gs=True,
        match_3dgs_mcmc_dev=False,
    )


def export_to_gs_video(*args, **kwargs):
    raise NotImplementedError(
        "Video rendering removed from vendored depth_anything_3. "
        "Use export_to_gs_ply() to export PLY files only."
    )
