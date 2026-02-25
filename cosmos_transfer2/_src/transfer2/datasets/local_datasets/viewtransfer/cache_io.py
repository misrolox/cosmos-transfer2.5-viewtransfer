# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache IO helpers (locks + atomic writes) for view-transfer datasets."""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from torchcodec.decoders import VideoDecoder


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def file_lock(lock_path: str | Path, *, timeout_sec: float, poll_sec: float) -> Iterator[None]:
    """Simple lockfile using atomic creation."""
    lock_path = Path(lock_path)
    ensure_parent_dir(lock_path)
    deadline = time.time() + timeout_sec
    acquired = False

    while time.time() <= deadline:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(f"{os.getpid()} {time.time()}\n")
            acquired = True
            break
        except FileExistsError:
            time.sleep(poll_sec)

    if not acquired:
        raise TimeoutError(f"Timeout acquiring lock: {lock_path}")

    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def atomic_save_npz(path: str | Path, **arrays) -> None:
    path = Path(path)
    ensure_parent_dir(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def write_video_atomic(*, path: str | Path, frames: Iterable[np.ndarray] | list[np.ndarray], fps: float) -> None:
    """Write frames to H264 mp4 via temp file + atomic rename."""
    path = Path(path)
    ensure_parent_dir(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.mp4")

    if fps <= 0:
        raise ValueError(f"FPS must be > 0. Got {fps} for path={path}")
    fps = float(fps)

    write_video_h264_ffmpeg(frames=frames, fps=fps, out_path=tmp)
    os.replace(tmp, path)


def write_video_h264_ffmpeg(
    *,
    frames: Iterable[np.ndarray] | list[np.ndarray],
    fps: float,
    out_path: str | Path,
) -> None:
    """Write RGB or grayscale uint8 frames to mp4 (libx264) by piping rawvideo into ffmpeg."""
    out_path = Path(out_path)
    ensure_parent_dir(out_path)
    frame_iter = iter(frames)
    try:
        first = next(frame_iter)
    except StopIteration as e:
        raise ValueError(f"No frames provided for video write: {out_path}") from e

    first, color_mode = _normalize_first_raw_frame_uint8(first)
    height, width = first.shape[:2]
    pix_fmt = "rgb24" if color_mode == "rgb" else "gray"
    if fps <= 0:
        raise ValueError(f"FPS must be > 0. Got {fps} for out_path={out_path}")

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(float(fps)),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None

    try:
        process.stdin.write(np.ascontiguousarray(first).tobytes())
        for frame in frame_iter:
            frame = _normalize_raw_frame_uint8(frame, color_mode=color_mode)
            if frame.shape[0] != height or frame.shape[1] != width:
                raise RuntimeError(
                    f"Frame size mismatch: got {frame.shape[1]}x{frame.shape[0]}, expected {width}x{height}."
                )
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close()
        stderr_bytes = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
    except Exception as e:
        try:
            process.stdin.close()
        except Exception:
            pass
        process.kill()
        _, stderr_bytes = process.communicate()
        stderr_msg = stderr_bytes.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg failed while writing {out_path}: {stderr_msg}") from e

    if return_code != 0:
        stderr_msg = stderr_bytes.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg failed with return code {return_code} for {out_path}: {stderr_msg}")


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    if np.issubdtype(frame.dtype, np.floating):
        if not np.isfinite(frame).all():
            raise ValueError("Float frame contains NaN/Inf values.")
        min_val = float(frame.min())
        max_val = float(frame.max())
        if min_val < 0:
            raise ValueError(f"Float frame has negative values: min={min_val}, max={max_val}")
        if max_val <= 1.0:
            return np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
        if max_val <= 255.0:
            return np.clip(frame, 0.0, 255.0).astype(np.uint8)
        raise ValueError(
            f"Float frame values out of expected range [0,1] or [0,255]. Observed min={min_val}, max={max_val}."
        )
    if np.issubdtype(frame.dtype, np.integer):
        min_val = int(frame.min())
        max_val = int(frame.max())
        if min_val < 0 or max_val > 255:
            raise ValueError(f"Integer frame values out of uint8 range [0,255]. Observed min={min_val}, max={max_val}.")
        return frame.astype(np.uint8)
    raise ValueError(f"Unsupported frame dtype={frame.dtype}. Expected uint8/float/int.")


def _normalize_first_raw_frame_uint8(frame: np.ndarray) -> tuple[np.ndarray, str]:
    """Normalize first frame and infer stream mode: 'rgb' or 'gray'."""
    frame_uint8 = _to_uint8(frame)
    if frame_uint8.ndim == 2:
        return frame_uint8, "gray"
    if frame_uint8.ndim == 3 and frame_uint8.shape[2] == 1:
        return frame_uint8[..., 0], "gray"
    if frame_uint8.ndim == 3 and frame_uint8.shape[2] == 3:
        return frame_uint8, "rgb"
    raise ValueError(f"Expected grayscale (H,W)/(H,W,1) or RGB (H,W,3), got shape={frame_uint8.shape}")


def _normalize_raw_frame_uint8(frame: np.ndarray, *, color_mode: str) -> np.ndarray:
    frame_uint8 = _to_uint8(frame)
    if color_mode == "gray":
        if frame_uint8.ndim == 2:
            return frame_uint8
        if frame_uint8.ndim == 3 and frame_uint8.shape[2] == 1:
            return frame_uint8[..., 0]
        raise ValueError(f"Expected grayscale frame for stream mode='gray', got shape={frame_uint8.shape}")
    if color_mode == "rgb":
        if frame_uint8.ndim == 3 and frame_uint8.shape[2] == 3:
            return frame_uint8
        raise ValueError(f"Expected RGB frame for stream mode='rgb', got shape={frame_uint8.shape}")
    raise ValueError(f"Unsupported color_mode={color_mode!r}")


def get_video_fps(path: str | Path) -> float:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    try:
        decoder = VideoDecoder(str(path), device="cpu")
        fps = decoder.metadata.average_fps
        assert fps is not None, "Decord returned None for average_fps"
        fps = float(fps)
        del decoder
    except Exception as e:
        raise RuntimeError(f"Failed to obtain a valid FPS via decord for video: {path}") from e

    if fps <= 0:
        raise RuntimeError(f"Failed to obtain a valid FPS via decord for video: {path}")
    return fps
