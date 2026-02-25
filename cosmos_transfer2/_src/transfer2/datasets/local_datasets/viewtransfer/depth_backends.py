# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Depth cache generation backends for Agibot view-transfer."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel
from tqdm import tqdm

from cosmos_transfer2._src.transfer2.datasets.local_datasets.viewtransfer.cache_io import (
    file_lock,
    get_video_fps,
    write_video_atomic,
)
from cosmos_transfer2._src.transfer2.datasets.local_datasets.viewtransfer.path_templates import (
    depth_cache_path,
    raw_depth_pattern,
    raw_video_path,
)

_SUPPORTED_DEPTH_ESTIMATORS = {"agibot", "moge"}


def _generate_agibot_depth_cache(
    *,
    dataset_dir: str,
    out_path: Path,
    task: str,
    episode: str,
    source_clip: str,
) -> None:
    pattern = raw_depth_pattern(dataset_dir, task, episode, source_clip)
    if "%06d" not in pattern.name:
        raise ValueError(f"Expected %06d pattern in raw depth filename: {pattern}")

    depth_frames: list[np.ndarray] = []
    idx = 0
    while True:
        cur = pattern.parent / pattern.name.replace("%06d", f"{idx:06d}")
        if not cur.exists():
            break
        depth = cv2.imread(str(cur), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Failed to read depth frame: {cur}")
        depth_frames.append(depth)
        idx += 1

    if not depth_frames:
        raise FileNotFoundError(f"No raw depth frames found for pattern: {pattern}")

    source_video = raw_video_path(dataset_dir, task, episode, source_clip)
    fps = get_video_fps(source_video)
    write_video_atomic(path=out_path, frames=depth_frames, fps=fps)


def _generate_moge_depth_cache(
    *,
    dataset_dir: str,
    out_path: Path,
    task: str,
    episode: str,
    source_clip: str,
    moge_device: str,
    moge_batch_size: int,
) -> None:
    source_video = raw_video_path(dataset_dir, task, episode, source_clip)
    if not source_video.exists():
        raise FileNotFoundError(f"Source video not found for MoGe depth: {source_video}")

    frames_rgb, fps = _read_all_rgb_frames(source_video)
    if len(frames_rgb) == 0:
        raise RuntimeError(f"No frames read from source video: {source_video}")

    output = _run_batch_moge_estimation(
        np.asarray(frames_rgb, dtype=np.uint8),
        batch_size=moge_batch_size,
        device=moge_device,
    )
    depth = output["depth"]
    depth_np = depth.detach().cpu().numpy() if hasattr(depth, "detach") else np.asarray(depth)
    depth_frames = [d for d in depth_np]
    write_video_atomic(path=out_path, frames=depth_frames, fps=fps)


def ensure_depth_cache(
    *,
    dataset_dir: str,
    cache_root: str,
    task: str,
    episode: str,
    source_clip: str,
    depth_estimator: str,
    lock_timeout_sec: float,
    lock_poll_sec: float,
    moge_device: str = "cuda",
    moge_batch_size: int = 8,
) -> Path:
    """Ensure depth cache exists for a source clip and return cache path."""
    if depth_estimator not in _SUPPORTED_DEPTH_ESTIMATORS:
        raise ValueError(
            f"Unsupported depth_estimator={depth_estimator!r}. Expected one of {_SUPPORTED_DEPTH_ESTIMATORS}."
        )

    out_path = depth_cache_path(
        cache_root=cache_root,
        task=task,
        episode=episode,
        source_clip=source_clip,
        depth_estimator=depth_estimator,
    )
    if out_path.exists():
        return out_path

    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    with file_lock(lock_path, timeout_sec=lock_timeout_sec, poll_sec=lock_poll_sec):
        if out_path.exists():
            return out_path
        if depth_estimator == "agibot":
            _generate_agibot_depth_cache(
                dataset_dir=dataset_dir,
                out_path=out_path,
                task=task,
                episode=episode,
                source_clip=source_clip,
            )
        else:
            _generate_moge_depth_cache(
                dataset_dir=dataset_dir,
                out_path=out_path,
                task=task,
                episode=episode,
                source_clip=source_clip,
                moge_device=moge_device,
                moge_batch_size=moge_batch_size,
            )
    return out_path


def _read_all_rgb_frames(video_path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = get_video_fps(video_path)
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def _run_batch_moge_estimation(rgb_np_uint8, batch_size=8, model=None, device="cuda"):
    """
    Runs MoGe estimation on a batch of RGB images.

    Args:
        rgb_np_uint8 (np.ndarray): A numpy array of shape (N, H, W, 3) containing RGB images in uint8 format.
        batch_size (int): The batch size to use for processing the images.
        model (MoGeModel, optional): An instance of the MoGeModel. If None, a pre-trained model will be loaded.
        device (str or torch.device): The device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        A dictionary containing the following keys:
        - `points`: output tensor of shape (B, H, W, 3) or (H, W, 3).
        - `depth`: tensor of shape (B, H, W) or (H, W) containing the depth map.
        - `intrinsics`: tensor of shape (B, 3, 3) or (3, 3) containing the camera intrinsics.
    """
    assert rgb_np_uint8.shape[-1] == 3, "Input images must have 3 channels (RGB)"
    if rgb_np_uint8.ndim == 3:
        rgb_np_uint8 = rgb_np_uint8[None]  # Add batch dimension if missing
    assert rgb_np_uint8.ndim == 4, "Input must be a 4D array of shape (N, H, W, 3)"

    device = torch.device(device)
    if model is None:
        model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl").to(device)
    else:
        model = model.to(device)

    input = torch.tensor(rgb_np_uint8 / 255, dtype=torch.float32, device=device).permute(0, 3, 1, 2)

    outputs = []
    for batch_start in tqdm(range(0, len(input), batch_size), desc="Running MoGe estimation"):
        batch = input[batch_start : batch_start + batch_size]
        outputs.append(model.infer(batch))

    return {k: torch.cat([o[k] for o in outputs], dim=0) for k in outputs[0].keys()}
