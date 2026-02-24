# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Post-training configs for view-transfer with inpaint control input.

These experiments are intended for custom inpaint datasets where the control signal is
novel-view rendering (for example, depth-based reprojection) and control branch parameters
should be trained from scratch.

Weight initialization details:
- model.config.base_load_from points to a public base checkpoint so base weights are loaded.
- Control branch is initialized from base weights by copy_weights_to_control_branch().

Example usage:
    torchrun --nproc_per_node=8 -m scripts.train \
      --config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py \
      -- experiment=transfer2_singleview_posttrain_inpaint_viewtransfer_predictbase \
      dataloader_train.dataset.dataset_dir=/path/to/your/dataset
"""

import os

from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import CheckpointConfig
from cosmos_transfer2.config import DEFAULT_BASE_EXPERIMENT

# Public checkpoint choices for base initialization.
# 1) Predict2.5 base pre-trained checkpoint (no control specialization).
PREDICT2P5_BASE_PRETRAIN_CHECKPOINT = CheckpointConfig.from_uri("d20b7120-df3e-4911-919d-db6e08bad31c")


def _to_base_load_from(checkpoint_s3_uri: str) -> dict:
    """Convert a registered checkpoint S3 URI into base_load_from format."""
    if not checkpoint_s3_uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {checkpoint_s3_uri}")
    return dict(
        # load_base_model() appends "/model" internally; keep this as .../checkpoints/iter_xxx
        load_path=checkpoint_s3_uri.removeprefix("s3://").removesuffix("/model"),
        credentials="credentials/s3_checkpoint.secret",
    )


def _build_inpaint_viewtransfer_experiment(name: str, base_checkpoint_s3_uri: str) -> dict:
    return dict(
        defaults=[
            DEFAULT_BASE_EXPERIMENT,
            {"override /data_train": "example_singleview_train_data_agibot_viewtransfer"},
        ],
        job=dict(
            project="cosmos_transfer2_posttrain",
            group="view_transfer",
            name=name,
        ),
        checkpoint=dict(
            save_iter=1000,
            load_path="",  # Keep empty: do not load edge/depth/seg/vis control checkpoint weights.
            load_training_state=False,
            strict_resume=False,
            load_from_object_store=dict(enabled=False),
            save_to_object_store=dict(enabled=False),
        ),
        model=dict(
            config=dict(
                hint_keys="inpaint",
                base_load_from=_to_base_load_from(base_checkpoint_s3_uri),
            ),
        ),
        dataloader_train=dict(
            dataset=dict(
                # Used by custom datasets that support control_input_type; harmless for placeholder config.
                control_input_type="inpaint",
            ),
        ),
        trainer=dict(
            max_iter=5000,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                heart_beat=dict(save_s3=False),
                iter_speed=dict(save_s3=False),
                device_monitor=dict(save_s3=False),
                every_n_sample_reg=dict(save_s3=False, every_n=200),
                every_n_sample_ema=dict(save_s3=False, every_n=200),
                wandb=dict(save_s3=False),
                wandb_10x=dict(save_s3=False),
                dataloader_speed=dict(save_s3=False),
                frame_loss_log=dict(save_s3=False),
            ),
        ),
        scheduler=dict(
            cycle_lengths=[5000],
        ),
        model_parallel=dict(
            context_parallel_size=int(os.environ.get("WORLD_SIZE", "1")),
        ),
    )

transfer2_singleview_posttrain_inpaint_viewtransfer_predictbase = _build_inpaint_viewtransfer_experiment(
    name="transfer2_singleview_posttrain_inpaint_viewtransfer_predictbase",
    base_checkpoint_s3_uri=PREDICT2P5_BASE_PRETRAIN_CHECKPOINT.s3.uri,
)


cs = ConfigStore.instance()
for _item in [
    transfer2_singleview_posttrain_inpaint_viewtransfer_predictbase,
]:
    _name: str = _item["job"]["name"]  # pyrefly: ignore
    cs.store(
        group="experiment",
        package="_global_",
        name=_name,
        node=_item,
    )
