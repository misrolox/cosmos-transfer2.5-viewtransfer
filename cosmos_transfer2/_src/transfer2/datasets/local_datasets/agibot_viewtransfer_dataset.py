# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agibot single-view dataset for inpaint-based view-transfer post-training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torchcodec import FrameBatch
from torchcodec.decoders import VideoDecoder

from cosmos_transfer2._src.imaginaire.lazy_config import instantiate
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.transfer2.datasets.augmentor_provider import get_video_augmentor_v2_with_control
from cosmos_transfer2._src.transfer2.datasets.local_datasets.viewtransfer.cache_io import (
    get_video_fps,
    load_depth_mkv_ffv1_mm_u16,
    load_full_video_frames,
    load_mask_mkv_ffv1,
    save_mask_mkv_ffv1,
    save_render_mp4_h264,
)
from cosmos_transfer2._src.transfer2.datasets.local_datasets.viewtransfer.camera_parameters import (
    ensure_camera_parameters_cache,
)
from cosmos_transfer2._src.transfer2.datasets.local_datasets.viewtransfer.depth_backends import ensure_depth_cache
from cosmos_transfer2._src.transfer2.datasets.local_datasets.viewtransfer.gen3c.cache_3d import Cache4D
from cosmos_transfer2._src.transfer2.datasets.local_datasets.viewtransfer.path_templates import (
    point_cloud_cache_path,
    raw_video_path,
    render_cache_paths,
)
from cosmos_transfer2._src.transfer2.datasets.local_datasets.viewtransfer.sample_index import (
    AnchorDirection,
    PairMode,
    ViewTransferPairSample,
    build_samples,
)
from cosmos_transfer2._src.transfer2.utils.input_handling import detect_aspect_ratio


# Mock URL object for augmentor compatibility
class MockUrlMeta:
    """Mock metadata object for WebDataset compatibility."""

    def __init__(self):
        self.opts = {}


class MockUrl:
    """Mock URL object that augmentors expect from WebDataset."""

    def __init__(self, url: str):
        self._url = url
        self.meta = MockUrlMeta()

    def __str__(self) -> str:
        return self._url

    def __repr__(self) -> str:
        return f"MockUrl({self._url})"


class AgibotViewTransferDataset(Dataset):
    """Local dataset with cache-first view-transfer conditioning for inpaint control."""

    def __init__(
        self,
        dataset_dir: str,
        num_frames: int,
        video_size: tuple[int, int],
        resolution: str = "720",
        hint_key: str = "control_input_inpaint",
        is_train: bool = True,
        caption_type: str = "t2w_qwen2p5_7b",
        cache_root: str = ".agibot_cache",
        clip_names: tuple[str, ...] = ("head", "hand_left", "hand_right"),
        depth_estimator: str = "agibot",
        pair_mode: str | PairMode = "all_ordered_nonself",
        anchor_clip: str | None = None,
        anchor_direction: str | AnchorDirection | None = "source_to_others",
        explicit_pairs: tuple[tuple[str, str], ...] | None = None,
        urdf_path: str = "",
        usd_path: str | None = None,
        camera_prims: dict[str, str] | None = None,
        base_frame: str = "base_link",
        decoder_device: str = "cpu",
        moge_device: str = "cuda",
        moge_batch_size: int = 8,
        lock_timeout_sec: float = 600.0,
        lock_poll_sec: float = 0.25,
        **kwargs: object,
    ) -> None:
        super().__init__()
        del kwargs
        if hint_key != "control_input_inpaint":
            raise ValueError(
                f"AgibotViewTransferDataset currently supports only hint_key='control_input_inpaint', got {hint_key!r}"
            )

        self.dataset_dir = dataset_dir
        self.sequence_length = int(num_frames)
        self.video_size = tuple(video_size)
        self.resolution = resolution
        self.hint_key = hint_key
        self.ctrl_type = hint_key.replace("control_input_", "")
        self.is_train = is_train
        self.caption_type = caption_type

        self.cache_root = cache_root
        self.clip_names = tuple(clip_names)
        self.depth_estimator = depth_estimator
        self.pair_mode = pair_mode
        self.anchor_clip = anchor_clip
        self.anchor_direction = anchor_direction
        self.explicit_pairs = explicit_pairs
        self._pair_mode = self._coerce_pair_mode(self.pair_mode)
        self._anchor_direction = self._coerce_anchor_direction(self.anchor_direction)

        self.urdf_path = urdf_path
        self.usd_path = usd_path
        self.camera_prims = camera_prims
        self.base_frame = base_frame

        self.decoder_device = decoder_device
        self.moge_device = moge_device
        self.moge_batch_size = int(moge_batch_size)
        self.lock_timeout_sec = float(lock_timeout_sec)
        self.lock_poll_sec = float(lock_poll_sec)

        self.samples = build_samples(
            dataset_dir=dataset_dir,
            clip_names=self.clip_names,
            pair_mode=self._pair_mode,
            anchor_clip=self.anchor_clip,
            anchor_direction=self._anchor_direction,
            explicit_pairs=self.explicit_pairs,
        )
        if not self.samples:
            raise RuntimeError(
                "AgibotViewTransferDataset found no samples after scanning dataset root. "
                f"dataset_dir={dataset_dir}, clip_names={self.clip_names}, pair_mode={self.pair_mode}"
            )

        self.num_failed = 0
        self.bad_sample_indices = set()  # Track samples that fail

        # Use proper augmentor pipeline for training quality
        # This includes randomized edge detection, reflection padding, and text transforms
        # Pass embedding_type=None since we're handling T5 embeddings ourselves
        # (if embedding_type is set, the function returns early with only video_parsing)
        augmentor_config = get_video_augmentor_v2_with_control(
            resolution=resolution,
            caption_type=caption_type,
            embedding_type=None,  # We handle embeddings ourselves, get full augmentor pipeline
            control_input_type=self.ctrl_type,
            use_random=is_train,  # Enable random augmentations for training
        )

        # Filter out augmentors that don't apply to local datasets
        # The augmentor pipeline includes augmentors designed for S3/WebDataset that need to be skipped:
        # - video_parsing: Decodes video bytes from S3 → we already load tensors from local MP4 files
        # - depth_parsing: Decodes depth bytes from S3 key "depth_pervideo_video_depth_anything" → we load from local depth/ folder
        # - seg_parsing: Decodes seg bytes from S3 key "segmentation_sam2_color_video_v2" → we load from local seg/ folder
        # - merge_datadict: Merges multiple WebDataset shards → not needed for single local dataset
        # - text_transform: Loads pre-computed T5 embeddings → we pass raw captions for on-the-fly encoding
        skip_augmentors = ["video_parsing", "merge_datadict", "text_transform", "depth_parsing", "seg_parsing"]
        augmentor_config = {k: v for k, v in augmentor_config.items() if k not in skip_augmentors}

        log.info(f"Filtered augmentors: {list(augmentor_config.keys())}")

        # Instantiate augmentors
        self.augmentor = {k: instantiate(v) for k, v in augmentor_config.items()}

        # Double-check text_transform is not present
        if "text_transform" in self.augmentor:
            raise RuntimeError("text_transform should have been filtered out but is still present!")

        log.info(f"Initialized AgibotViewTransferDataset with {len(self.samples)} videos")
        log.info(f"  Dataset dir: {self.dataset_dir}")
        log.info(f"  Control type: {self.ctrl_type}")
        log.info(f"  Resolution: {resolution}, Video size: {video_size}")
        log.info(f"  Required frames: {self.sequence_length}")

    @staticmethod
    def _coerce_pair_mode(pair_mode: str | PairMode) -> PairMode:
        if isinstance(pair_mode, PairMode):
            return pair_mode
        try:
            return PairMode(pair_mode)
        except ValueError as exc:
            expected = {mode.value for mode in PairMode}
            raise ValueError(f"Unsupported pair_mode={pair_mode!r}. Expected one of {expected}.") from exc

    @staticmethod
    def _coerce_anchor_direction(anchor_direction: str | AnchorDirection | None) -> AnchorDirection | None:
        if anchor_direction is None:
            return None
        if isinstance(anchor_direction, AnchorDirection):
            return anchor_direction
        try:
            return AnchorDirection(anchor_direction)
        except ValueError as exc:
            expected = {direction.value for direction in AnchorDirection}
            raise ValueError(f"Unsupported anchor_direction={anchor_direction!r}. Expected one of {expected}.") from exc

    def __len__(self) -> int:
        return len(self.samples)

    def __str__(self) -> str:
        return f"AgibotViewTransferDataset: {len(self.samples)} samples from {self.dataset_dir}"

    def __getitem__(self, index: int) -> dict[str, Any]:
        max_retries = 10  # Try up to 10 different samples
        original_index = index

        for retry in range(max_retries):
            # Skip known bad videos
            if index in self.bad_sample_indices:
                index = (index + 1) % len(self.samples)
                continue

            sample = self.samples[index]
            try:
                source_video_path = raw_video_path(self.dataset_dir, sample.task, sample.episode, sample.source_clip)
                target_video_path = raw_video_path(self.dataset_dir, sample.task, sample.episode, sample.target_clip)
                render_path, mask_path = render_cache_paths(
                    self.cache_root,
                    sample.task,
                    sample.episode,
                    sample.source_clip,
                    sample.target_clip,
                    self.depth_estimator,
                )

                if not (render_path.exists() and mask_path.exists()):
                    self._prepare_render_cache(sample)

                (
                    source_frames_nchw_uint8,
                    target_frames_nchw_uint8,
                    render_frames_nchw_uint8,
                    render_mask_frames_n1hw_uint8,
                    fps,
                    frame_ids,
                ) = self._load_video_frames(
                    source_video_path=source_video_path,
                    target_video_path=target_video_path,
                    render_video_path=render_path,
                    render_mask_path=mask_path,
                )  # N C H W uint8, N C H W uint8, N C H W uint8, N 1 H W uint8, float, list[int]

                # Permute to (C, T, H, W) format expected by augmentors
                source_video = source_frames_nchw_uint8.permute(1, 0, 2, 3)  # C T H W
                target_video = target_frames_nchw_uint8.permute(1, 0, 2, 3)  # C T H W
                control = render_frames_nchw_uint8.permute(1, 0, 2, 3)  # C T H W
                mask = render_mask_frames_n1hw_uint8.permute(1, 0, 2, 3)  # 1 T H W

                target_h, target_w = target_video.shape[-2:]
                aspect_ratio = detect_aspect_ratio((target_w, target_h))

                # Build data dictionary
                data = {
                    "video": target_video,
                    "source_video": source_video,
                    "inpaint": control,
                    "inpaint_mask": mask,
                    "aspect_ratio": aspect_ratio,
                    "fps": fps,
                    "frame_start": frame_ids[0],
                    "frame_end": frame_ids[-1] + 1,
                    "num_frames": self.sequence_length,
                    "chunk_index": 0,
                    "frame_indices": frame_ids,
                    "n_orig_video_frames": len(frame_ids),
                }

                # Load caption placeholder
                caption = "a robot video"
                data[self.caption_type] = caption

                # Create metadata structure for augmentor compatibility
                # The augmentor expects "metas" with window information
                # Map caption_type to the expected caption key in window_data
                if self.caption_type == "t2w_qwen2p5_7b":
                    caption_key_in_window = "qwen2p5_7b_caption"
                else:
                    caption_key_in_window = self.caption_type

                window_data = {
                    "start_frame": frame_ids[0],
                    "end_frame": frame_ids[-1] + 1,
                    caption_key_in_window: caption,
                }
                data["metas"] = {
                    "framerate": fps,
                    "nb_frames": len(frame_ids),
                    # Create a single window spanning the entire video segment
                    # Include both windows and t2w_windows for different caption types
                    "windows": [window_data],
                    "t2w_windows": [window_data],
                    "i2w_windows_later_frames": [window_data],
                }

                # Pass raw caption for on-the-fly encoding by model's text encoder
                # (Like multiview dataset - model will encode with Qwen/reason1 encoder)
                data["ai_caption"] = caption

                # Add URL and key for logging (used by augmentors and training)
                # Use MockUrl object for augmentor compatibility (augmentors expect __url__.meta.opts)
                data["__url__"] = MockUrl(str(self.dataset_dir))
                data["__key__"] = f"Agibot_{sample.task}_{sample.episode}_{sample.source_clip}_to_{sample.target_clip}"

                # Apply augmentation pipeline
                # This includes: resizing, padding, and control input generation
                for aug_name, aug_fn in self.augmentor.items():
                    result = aug_fn(data)  # pyright: ignore
                    # Check if augmentor returned None (e.g., filtering)
                    if result is None:
                        raise ValueError(f"Augmentor {aug_name} filtered out the sample")
                    data = result

                # Convert MockUrl back to string for DataLoader collate compatibility
                # (PyTorch's collate function can't handle custom objects)
                if isinstance(data.get("__url__"), MockUrl):
                    data["__url__"] = str(data["__url__"])

                # Add final metadata (after augmentation)
                c, t, h, w = data["video"].shape
                if "image_size" not in data:
                    data["image_size"] = torch.tensor([h, w, h, w])
                if "padding_mask" not in data:
                    data["padding_mask"] = torch.ones(1, h, w)  # All valid (no padding)

                # Validate output format after augmentation
                assert data["video"].dtype == torch.uint8, f"Video dtype is {data['video'].dtype}, expected uint8"
                assert data["video"].shape[0] == 3, f"Video should have 3 channels, got {data['video'].shape[0]}"
                assert data["video"].shape[1] == self.sequence_length, (
                    f"Video should have {self.sequence_length} frames, got {data['video'].shape[1]}"
                )

                # Check control input exists and has correct format
                ctrl_key = f"control_input_{self.ctrl_type}"
                assert ctrl_key in data, f"Control input key '{ctrl_key}' not found in data"
                assert data[ctrl_key].dtype == torch.uint8, (
                    f"Control input dtype is {data[ctrl_key].dtype}, expected uint8"
                )
                assert data[ctrl_key].shape == data["video"].shape, (
                    f"Control input shape {data[ctrl_key].shape} doesn't match video shape {data['video'].shape}"
                )

                log.debug(
                    f"Dataset sample ready: video={data['video'].shape} {data['video'].dtype}, "
                    f"{ctrl_key}={data[ctrl_key].shape} {data[ctrl_key].dtype}, "
                )

                return data

            except Exception as e:
                self.num_failed_loads += 1
                # Mark this sample as bad so we skip it in the future
                self.bad_sample_indices.add(index)

                log.warning(
                    f"Failed to load sample {self.samples[index]} (index {index}): {e}. "
                    f"Marking as bad and trying next sample. "
                    f"(attempt {retry + 1}/{max_retries}, total bad samples: {len(self.bad_sample_indices)})",
                    rank0_only=False,
                )

                if retry == max_retries - 1:
                    log.error(
                        f"Failed to load data after {max_retries} attempts starting from index {original_index}. "
                        f"Total bad samples: {len(self.bad_sample_indices)}/{len(self.samples)}"
                    )
                    raise RuntimeError(
                        f"Failed to load data after {max_retries} attempts. "
                        f"Original index: {original_index}, last tried: {self.samples[index]}"
                    )

                # Try the next sample in sequence (wraps around at end)
                index = (index + 1) % len(self.samples)

        raise RuntimeError("Should not reach here")

    def _prepare_render_cache(self, sample: ViewTransferPairSample) -> None:
        """
        Prepares the render cache and mask for a given sample.
        """
        render_path, mask_path = render_cache_paths(
            self.cache_root,
            sample.task,
            sample.episode,
            sample.source_clip,
            sample.target_clip,
            self.depth_estimator,
        )
        if render_path.exists() and mask_path.exists():
            return

        pcd_path = point_cloud_cache_path(
            self.cache_root,
            sample.task,
            sample.episode,
            sample.source_clip,
            sample.target_clip,
            self.depth_estimator,
        )
        cache_4d = self._reconstruct_point_cloud_cache(sample=sample, pcd_path=pcd_path)

        self._render_from_point_cloud_cache(
            sample=sample,
            cache_4d=cache_4d,
            render_path=render_path,
            mask_path=mask_path,
        )

    def _ensure_depth_cache(self, sample: ViewTransferPairSample) -> Path:
        return ensure_depth_cache(
            dataset_dir=self.dataset_dir,
            cache_root=self.cache_root,
            task=sample.task,
            episode=sample.episode,
            source_clip=sample.source_clip,
            depth_estimator=self.depth_estimator,
            moge_device=self.moge_device,
            moge_batch_size=self.moge_batch_size,
            lock_timeout_sec=self.lock_timeout_sec,
            lock_poll_sec=self.lock_poll_sec,
        )

    def _ensure_camera_parameters_cache(self, sample: ViewTransferPairSample, clip_name: str) -> Path:
        return ensure_camera_parameters_cache(
            dataset_dir=self.dataset_dir,
            cache_root=self.cache_root,
            task=sample.task,
            episode=sample.episode,
            clip_name=clip_name,
            clip_names=self.clip_names,
            urdf_path=self.urdf_path,
            usd_path=self.usd_path,
            camera_prims=self.camera_prims,
            base_frame=self.base_frame,
            lock_timeout_sec=self.lock_timeout_sec,
            lock_poll_sec=self.lock_poll_sec,
        )

    def _reconstruct_point_cloud_cache(self, *, sample: ViewTransferPairSample, pcd_path: Path) -> Cache4D:
        """
        Reconstructs the point cloud cache for a given sample using depth estimation and source extrinsics.
        """
        source_video_path = raw_video_path(self.dataset_dir, sample.task, sample.episode, sample.source_clip)
        input_frames_nchw_uint8 = load_full_video_frames(source_video_path, decoder_device=self.decoder_device)
        image_tensor = (input_frames_nchw_uint8.float() / 127.5) - 1.0  # [0,255] -> [-1,1]

        if pcd_path.exists():
            points = torch.from_numpy(np.load(pcd_path)["points"])  # F, H, W, 3
            return Cache4D(
                input_image=image_tensor,  # [F, C, H, W]
                input_depth=None,
                input_w2c=None,
                input_intrinsics=None,
                input_format=["F", "C", "H", "W"],
                input_points=points,  # [F, H, W, 3]
            )

        depth_path = self._ensure_depth_cache(sample)
        _, _, height, width = image_tensor.shape
        depth_nhw_m_float64_np = load_depth_mkv_ffv1_mm_u16(
            path=depth_path, width=width, height=height, return_float64=True
        )
        depth_n1hw_m_float64 = torch.from_numpy(depth_nhw_m_float64_np).unsqueeze(1)  # [F, 1, H, W]
        source_cam_pam_path = self._ensure_camera_parameters_cache(sample, sample.source_clip)
        cam_pams = np.load(source_cam_pam_path)
        intrinsics_n33 = torch.from_numpy(cam_pams["intrinsics"]).float()  # [F, 3, 3]
        initial_w2c_n44 = torch.from_numpy(cam_pams["extrinsics"]).float()  # [F, 4, 4]

        cache_4d = Cache4D(
            input_image=image_tensor,  # [F, C, H, W]
            input_depth=depth_n1hw_m_float64,  # [F, 1, H, W]
            input_w2c=initial_w2c_n44,  # [F, 4, 4]
            input_intrinsics=intrinsics_n33,  # [F, 3, 3]
            input_format=["F", "C", "H", "W"],
        )

        # Save point cloud cache
        cache_4d.export_point_cloud(pcd_path)

        return cache_4d

    def _render_from_point_cloud_cache(
        self,
        *,
        sample: ViewTransferPairSample,
        cache_4d: Cache4D,
        render_path: Path,
        mask_path: Path,
    ) -> None:
        """
        Renders the target view and disocclusion mask from the 4D cache for a given sample.
        """
        target_cam_pam_path = self._ensure_camera_parameters_cache(sample, sample.target_clip)
        cam_pams = np.load(target_cam_pam_path)
        target_w2c_n44 = torch.from_numpy(cam_pams["extrinsics"]).float().unsqueeze(0)  # [1, F, 4, 4]
        target_intrinsics_n33 = torch.from_numpy(cam_pams["intrinsics"]).float().unsqueeze(0)  # [1, F, 3, 3]

        rendered_warp_images, rendered_warp_masks = cache_4d.render_cache(
            target_w2cs=target_w2c_n44,
            target_intrinsics=target_intrinsics_n33,
        )  # [B, F, N, C, H, W]

        rendered_images = rendered_warp_images.squeeze(0).squeeze(1).cpu()  # [C, H, W]
        rendered_masks = rendered_warp_masks.squeeze(0).squeeze(1).cpu()  # [1, H, W]
        rendered_images = (rendered_images + 1.0) * 127.5  # [-1,1] -> [0,255]
        rendered_images = rendered_images.clamp(0, 255).byte().numpy()  # uint8
        rendered_masks = (rendered_masks > 0).byte().numpy()  # Binary mask, 1 for known, 0 for unknown

        source_video_path = raw_video_path(self.dataset_dir, sample.task, sample.episode, sample.source_clip)
        fps = get_video_fps(source_video_path)
        save_render_mp4_h264(frames=rendered_images, path=render_path, fps=fps)
        save_mask_mkv_ffv1(masks=rendered_masks, path=mask_path, fps=fps)

    def _load_video_frames(
        self,
        *,
        source_video_path: Path,
        target_video_path: Path,
        render_video_path: Path,
        render_mask_path: Path,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, list[int]]:
        source_decoder = VideoDecoder(str(source_video_path), device=self.decoder_device)
        target_decoder = VideoDecoder(str(target_video_path), device=self.decoder_device)
        render_decoder = VideoDecoder(str(render_video_path), device=self.decoder_device)
        width, height = self._get_shape_from_decoder(render_decoder)
        mask_frames = load_mask_mkv_ffv1(path=render_mask_path, width=width, height=height)  # T H W, uint8

        min_frames = min(len(source_decoder), len(target_decoder), len(render_decoder), len(mask_frames))
        if min_frames < self.sequence_length:
            raise ValueError(
                f"Not enough frames for sample. source={len(source_decoder)}, target={len(target_decoder)}, "
                f"render={len(render_decoder)}, mask={len(mask_frames)}, required={self.sequence_length}"
            )

        if self.is_train:
            start = np.random.randint(0, min_frames - self.sequence_length + 1)
        else:
            start = 0
        frame_ids = list(range(start, start + self.sequence_length))

        source_frames: FrameBatch = source_decoder.get_frames_at(frame_ids)  # N C H W, uint8
        target_frames: FrameBatch = target_decoder.get_frames_at(frame_ids)  # N C H W, uint8
        render_frames: FrameBatch = render_decoder.get_frames_at(frame_ids)  # N C H W, uint8
        mask_frames = torch.from_numpy(mask_frames[frame_ids]).unsqueeze(1)  # N 1 H W, uint8

        fps = self._get_fps_from_decoder(target_decoder)
        source_fps = self._get_fps_from_decoder(source_decoder)
        if abs(source_fps - fps) > 1e-3:
            log.warning(
                f"Source/target FPS mismatch for sample: source_fps={source_fps}, target_fps={fps}. Using target FPS."
            )

        return (
            source_frames.data,
            target_frames.data,
            render_frames.data,
            mask_frames,
            fps,
            frame_ids,
        )

    def _get_fps_from_decoder(self, decoder: VideoDecoder) -> float:
        fps = decoder.metadata.average_fps
        assert fps is not None, "Decord returned None for average_fps"
        fps = float(fps)
        if fps <= 0:
            raise RuntimeError("Failed to obtain a valid FPS")
        return fps

    def _get_shape_from_decoder(self, decoder: VideoDecoder) -> tuple[int, int]:
        width = decoder.metadata.width
        height = decoder.metadata.height
        if width is None or height is None:
            raise RuntimeError("Failed to obtain valid video dimensions from decoder metadata")
        return width, height
