# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FK-based camera extrinsics cache generation for Agibot view-transfer."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pinocchio as pin
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from cosmos_transfer2._src.transfer2.datasets.local_datasets.viewtransfer.cache_io import atomic_save_npz, file_lock
from cosmos_transfer2._src.transfer2.datasets.local_datasets.viewtransfer.path_templates import (
    CacheCategory,
    cache_episode_dir,
    extrinsics_cache_path,
    raw_proprio_h5_path,
)

DEFAULT_CAMERA_PRIMS: dict[str, str] = {
    "head": "/G1/head_link2/Head_Camera",
    "hand_right": "/G1/gripper_r_base_link/Right_Camera",
    "hand_left": "/G1/gripper_l_base_link/Left_Camera",
}


def ensure_extrinsics_cache(
    *,
    dataset_dir: str,
    cache_root: str,
    task: str,
    episode: str,
    clip_name: str,
    clip_names: tuple[str, ...],
    urdf_path: str,
    usd_path: str | None,
    camera_prims: dict[str, str] | None,
    base_frame: str,
    lock_timeout_sec: float,
    lock_poll_sec: float,
) -> Path:
    """Ensure per-clip FK extrinsics cache exists and return clip cache path."""
    out_path = extrinsics_cache_path(cache_root, task, episode, clip_name)
    if out_path.exists():
        return out_path

    episode_extrinsics_dir = cache_episode_dir(cache_root, CacheCategory.EXTRINSICS, task, episode)
    lock_path = episode_extrinsics_dir / "fk_generation.lock"
    with file_lock(lock_path, timeout_sec=lock_timeout_sec, poll_sec=lock_poll_sec):
        if out_path.exists():
            return out_path
        _generate_fk_extrinsics_for_episode(
            dataset_dir=dataset_dir,
            cache_root=cache_root,
            task=task,
            episode=episode,
            clip_names=clip_names,
            urdf_path=urdf_path,
            usd_path=usd_path,
            camera_prims=camera_prims,
            base_frame=base_frame,
        )
    if not out_path.exists():
        raise RuntimeError(f"FK extrinsics generation did not create expected cache: {out_path}")
    return out_path


class FKFromH5:
    """
    Loads URDF + H5 once, then provides random-access FK queries by timestep.
    """

    def __init__(
        self,
        urdf_path: str,
        h5_path: str,
        *,
        base_frame: str = "base_link",
        h5_joint_key: str = "/state/joint/position",
        h5_waist_key: str = "/state/waist/position",
        h5_head_key: str = "/state/head/position",
        h5_timestamp_key: str = "/timestamp",
        # H5->URDF joint mapping
        left_arm_joints: list[str] | None = None,
        right_arm_joints: list[str] | None = None,
        waist_joints: list[str] | None = None,
        head_joints: list[str] | None = None,
    ):
        self.urdf_path = urdf_path
        self.h5_path = h5_path
        self.base_frame = base_frame

        # --- load model/data ---
        self.model = pin.buildModelFromUrdf(urdf_path)  # pyright: ignore[reportAttributeAccessIssue]
        self.data = self.model.createData()

        # --- load signals ---
        self.timestamp_ns = self._load_h5(h5_path, h5_timestamp_key)
        self.arm_q = self._load_h5(h5_path, h5_joint_key)
        self.waist_q = self._load_h5(h5_path, h5_waist_key)
        self.head_q = self._load_h5(h5_path, h5_head_key)

        self.N = int(self.arm_q.shape[0])
        if not (
            self.timestamp_ns.shape[0] == self.N and self.waist_q.shape[0] == self.N and self.head_q.shape[0] == self.N
        ):
            raise ValueError(
                f"Length mismatch: ts={self.timestamp_ns.shape}, arm={self.arm_q.shape}, waist={self.waist_q.shape}, head={self.head_q.shape}"
            )

        # --- mapping ---
        self.name_to_q = self._joint_name_to_q_index()
        self.q_to_name = {v: k for k, v in self.name_to_q.items()}

        self.left_arm_joints = left_arm_joints or [
            "idx21_arm_l_joint1",
            "idx22_arm_l_joint2",
            "idx23_arm_l_joint3",
            "idx24_arm_l_joint4",
            "idx25_arm_l_joint5",
            "idx26_arm_l_joint6",
            "idx27_arm_l_joint7",
        ]
        self.right_arm_joints = right_arm_joints or [
            "idx61_arm_r_joint1",
            "idx62_arm_r_joint2",
            "idx63_arm_r_joint3",
            "idx64_arm_r_joint4",
            "idx65_arm_r_joint5",
            "idx66_arm_r_joint6",
            "idx67_arm_r_joint7",
        ]
        self.waist_joints = waist_joints or ["idx01_body_joint1", "idx02_body_joint2"]
        self.head_joints = head_joints or ["idx11_head_joint1", "idx12_head_joint2"]

        self._joint_sources: list[tuple[str, callable]] = []  # pyright: ignore[reportGeneralTypeIssues]
        for i, jn in enumerate(self.left_arm_joints):
            self._joint_sources.append((jn, lambda k, i=i: self.arm_q[k, i]))
        for i, jn in enumerate(self.right_arm_joints):
            self._joint_sources.append((jn, lambda k, i=i: self.arm_q[k, 7 + i]))
        for i, jn in enumerate(self.waist_joints):
            self._joint_sources.append((jn, lambda k, i=i: self.waist_q[k, i]))
        for i, jn in enumerate(self.head_joints):
            self._joint_sources.append((jn, lambda k, i=i: self.head_q[k, i]))

        missing = [jn for jn, _ in self._joint_sources if jn not in self.name_to_q]
        if missing:
            available = sorted(self.name_to_q.keys())
            raise ValueError(
                "Some mapped joint names were not found in the URDF:\n"
                f"  Missing: {missing}\n"
                f"  Example available joints (first 80): {available[:80]}"
            )

        # base frame id
        self._base_fid = self.model.getFrameId(base_frame)
        if self._base_fid >= len(self.model.frames):
            raise ValueError(f"Base frame '{base_frame}' not found in model.frames")

        # reusable q buffer + a tiny cache for repeated queries
        self._q = np.zeros(self.model.nq, dtype=float)
        self._last_idx: int | None = None

    @staticmethod
    def _load_h5(h5_path: str, key: str) -> np.ndarray:
        with h5py.File(h5_path, "r") as f:
            return np.asarray(f[key])

    @staticmethod
    def _normalize_link_name(link_name: str) -> str:
        """
        Accepts either 'head_link2' or '/G1/head_link2' (USD-style path).
        Returns the likely URDF frame/link name portion.
        """
        # If it's a USD path, take the last token
        if "/" in link_name:
            # handle trailing slash too
            tokens = [t for t in link_name.split("/") if t]
            if tokens:
                return tokens[-1]
        return link_name

    def _joint_name_to_q_index(self) -> dict[str, int]:
        """
        Map joint name -> starting index in q (Pinocchio configuration vector).
        """
        mapping: dict[str, int] = {}
        for j in range(1, self.model.njoints):  # skip "universe"
            jname = self.model.names[j]
            idx_q = int(self.model.joints[j].idx_q)
            mapping[jname] = idx_q
        return mapping

    def get_joint_configuration_dict(self, idx: int) -> dict[str, float]:
        """
        Returns a dict of joint name -> value at the given timestep idx.
        """
        return {jn: float(getval(idx)) for jn, getval in self._joint_sources}

    def _compute_fk_at(self, idx: int):
        if idx < 0 or idx >= self.N:
            raise IndexError(f"idx out of range: {idx} (valid: 0..{self.N - 1})")

        # simple cache: if same idx repeated, skip recompute
        if self._last_idx == idx:
            return

        self._q[:] = 0.0
        for jname, getval in self._joint_sources:
            self._q[self.name_to_q[jname]] = float(getval(idx))

        pin.forwardKinematics(self.model, self.data, self._q)  # pyright: ignore[reportAttributeAccessIssue]
        pin.updateFramePlacements(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        self._last_idx = idx

    def available_frames(self) -> list[str]:
        """
        Names you can query with link_pose_in_base (frame names in the Pinocchio model).
        """
        return [fr.name for fr in self.model.frames]

    def link_pose_in_base(self, link_name: str, idx: int) -> pin.SE3:  # pyright: ignore[reportAttributeAccessIssue]
        """
        Returns pose of a *link BODY frame* relative to the base frame at timestep idx.

        link_name can be:
        - "head_link2"
        - "/G1/head_link2"  (USD-style path; we take the last token)

        Output pose is base->link.
        """
        self._compute_fk_at(idx)

        fr_name = self._normalize_link_name(link_name)

        fid = self.model.getFrameId(fr_name)
        if fid >= len(self.model.frames) or self.model.frames[fid].name != fr_name:
            raise ValueError(f"Frame '{fr_name}' not found. Tip: call available_frames() to see valid names.")

        # Enforce that the requested name corresponds to a BODY frame (i.e., a link pose)
        fr = self.model.frames[fid]
        if hasattr(pin, "FrameType") and hasattr(pin.FrameType, "BODY"):  # pyright: ignore[reportAttributeAccessIssue]
            if fr.type != pin.FrameType.BODY:  # pyright: ignore[reportAttributeAccessIssue]
                raise ValueError(
                    f"'{fr_name}' exists but is not a BODY frame (type={fr.type}). "
                    f"This function expects link names like base_link, head_link2, gripper_l_base_link, etc."
                )

        # data.oMf[fid] is world(o)->frame(fid). We want base->link:
        H_world_base = self.data.oMf[self._base_fid]
        H_world_link = self.data.oMf[fid]
        H_base_link = H_world_base.inverse() * H_world_link

        return H_base_link


class CameraExtrinsics:
    REALSENSE_LINK_OPTICAL = np.eye(4)
    REALSENSE_LINK_OPTICAL[:3, :3] = R.from_euler("x", [180], degrees=True).as_matrix()

    def __init__(
        self,
        *,
        h5_path: str | None = None,
        urdf_path: str | None = None,
        fk: FKFromH5 | None = None,
        usd_path: str | None = None,
        base_frame: str = "base_link",
        robot_root: str = "/G1",
        camera_prims: dict[str, str] | None = None,
        optical_transform: np.ndarray | None = None,
    ):
        if fk is None and (h5_path is None or urdf_path is None):
            raise ValueError("Provide either fk, or both h5_path and urdf_path.")

        self.h5_path = h5_path if h5_path is not None else fk.h5_path  # pyright: ignore[reportOptionalMemberAccess]
        self.urdf_path = urdf_path if urdf_path is not None else fk.urdf_path  # pyright: ignore[reportOptionalMemberAccess]
        self.usd_path = usd_path if usd_path is not None else self.urdf_path.replace(".urdf", ".usda")
        self.robot_root = robot_root
        self.camera_prims = camera_prims or DEFAULT_CAMERA_PRIMS
        self.optical_transform = (
            np.asarray(optical_transform, dtype=float) if optical_transform is not None else self.REALSENSE_LINK_OPTICAL
        )

        self.fk = fk if fk is not None else FKFromH5(self.urdf_path, self.h5_path, base_frame=base_frame)
        self.stage = self._get_stage(self.usd_path)
        self.xcache = UsdGeom.XformCache(Usd.TimeCode.Default())  # pyright: ignore[reportAttributeAccessIssue]

        self._parent_link_prim_cache: dict[str, str] = {}
        self._H_link_camera_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _split_usd_path(path: str) -> tuple[str, ...]:
        return tuple(token for token in path.split("/") if token)

    def _parent_link_prim_path(self, camera_prim_path: str) -> str:
        """
        For this asset, link prims are directly under /G1:
            /G1/<link_name>/...
        So parent link prim is '/G1/<link_name>'.
        """
        if camera_prim_path in self._parent_link_prim_cache:
            return self._parent_link_prim_cache[camera_prim_path]

        tokens = self._split_usd_path(camera_prim_path)
        if not tokens:
            raise ValueError(f"Invalid prim path: {camera_prim_path}")

        root_tokens = self._split_usd_path(self.robot_root)
        if len(tokens) < len(root_tokens) + 1:
            raise ValueError(f"Camera prim path '{camera_prim_path}' doesn't look like '{self.robot_root}/<link>/...'")

        if tokens[: len(root_tokens)] != root_tokens:
            raise ValueError(f"Camera prim path '{camera_prim_path}' is not under robot_root '{self.robot_root}'")

        link_name = tokens[len(root_tokens)]
        parent_link_prim_path = self.robot_root.rstrip("/") + "/" + link_name
        self._parent_link_prim_cache[camera_prim_path] = parent_link_prim_path
        return parent_link_prim_path

    @staticmethod
    def _get_stage(usd_path: str) -> Usd.Stage:  # pyright: ignore[reportAttributeAccessIssue]
        stage = Usd.Stage.Open(usd_path)  # pyright: ignore[reportAttributeAccessIssue]
        if stage is None:
            raise RuntimeError(f"Failed to open USD stage: {usd_path}")
        return stage

    def _get_H_link_cam(self, camera_prim_path: str) -> np.ndarray:
        if camera_prim_path in self._H_link_camera_cache:
            return self._H_link_camera_cache[camera_prim_path]

        cam_prim = self.stage.GetPrimAtPath(camera_prim_path)
        if not cam_prim.IsValid():
            raise ValueError(f"Camera prim not found in USD: {camera_prim_path}")

        link_prim_path = self._parent_link_prim_path(camera_prim_path)
        link_prim = self.stage.GetPrimAtPath(link_prim_path)
        if not link_prim.IsValid():
            raise ValueError(f"Parent link prim not found in USD: {link_prim_path}")

        # Compute static camera mount transform relative to its parent link.
        gf_mat, _ = self.xcache.ComputeRelativeTransform(cam_prim, link_prim)
        H_link_cam = np.array(gf_mat).reshape(4, 4).T
        if not np.allclose(H_link_cam[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError(f"USD relative transform is not homogeneous.\n{H_link_cam}")

        self._H_link_camera_cache[camera_prim_path] = H_link_cam
        return H_link_cam

    def get_H_base_cam(self, camera_prim_path: str, idx: int) -> np.ndarray:
        link_prim_path = self._parent_link_prim_path(camera_prim_path)
        H_base_link = self.fk.link_pose_in_base(link_prim_path, idx=idx).homogeneous
        H_link_cam = self._get_H_link_cam(camera_prim_path)
        return H_base_link @ H_link_cam @ self.optical_transform

    def get_H_base_cams(self, index: int) -> dict[str, np.ndarray]:
        results: dict[str, np.ndarray] = {}
        for camera_name, camera_prim_path in self.camera_prims.items():
            results[camera_name] = self.get_H_base_cam(camera_prim_path, idx=index)
        return results

    def generate_camera_extrinsics(self) -> list[dict[str, np.ndarray]]:
        results: list[dict[str, np.ndarray]] = []
        for idx in tqdm(range(self.fk.N), desc="Generating camera extrinsics"):
            results.append(self.get_H_base_cams(index=idx))
        return results


def _generate_fk_extrinsics_for_episode(
    *,
    dataset_dir: str,
    cache_root: str,
    task: str,
    episode: str,
    clip_names: tuple[str, ...],
    urdf_path: str,
    usd_path: str | None,
    camera_prims: dict[str, str] | None,
    base_frame: str,
) -> None:
    if not urdf_path:
        raise ValueError("urdf_path must be provided to generate FK extrinsics.")

    h5_path = raw_proprio_h5_path(dataset_dir, task, episode)
    if not h5_path.exists():
        raise FileNotFoundError(f"FK input h5 not found: {h5_path}")

    if not camera_prims:
        camera_prims = DEFAULT_CAMERA_PRIMS

    missing = [clip for clip in clip_names if clip not in camera_prims]
    if missing:
        raise ValueError(
            f"Missing camera prim mappings for clips={missing}. Provide camera_prims for all clip_names={clip_names}."
        )

    generator = CameraExtrinsics(
        h5_path=str(h5_path),
        urdf_path=urdf_path,
        usd_path=usd_path,
        camera_prims={clip: camera_prims[clip] for clip in clip_names},
        base_frame=base_frame,
    )
    all_extrinsics = generator.generate_camera_extrinsics()
    if not all_extrinsics:
        raise RuntimeError(f"Generated empty extrinsics list for task={task} episode={episode}")

    for clip in clip_names:
        per_clip = []
        for frame_extrinsics in all_extrinsics:
            if clip not in frame_extrinsics:
                raise KeyError(f"Clip {clip!r} missing from generated extrinsics for task={task} episode={episode}.")
            per_clip.append(np.asarray(frame_extrinsics[clip], dtype=np.float32))

        extrinsics = np.stack(per_clip, axis=0)
        out_path = extrinsics_cache_path(cache_root, task, episode, clip)
        atomic_save_npz(out_path, extrinsics=extrinsics)
