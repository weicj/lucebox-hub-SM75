from __future__ import annotations

"""Shared backend/device placement plumbing for script-side test_dflash launchers."""

import os
from dataclasses import dataclass


def resolve_visible_devices(visible_devices: str | None,
                            fallback_device: int | None = None) -> str | None:
    if visible_devices:
        return visible_devices
    if fallback_device is None:
        return None
    return str(fallback_device)


def apply_backend_visible_devices(backend: str,
                                  *,
                                  visible_devices: str | None = None,
                                  fallback_device: int | None = None,
                                  base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    resolved = resolve_visible_devices(visible_devices, fallback_device)
    if resolved is None:
        return env
    if backend == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = resolved
        return env
    if backend == "hip":
        env["HIP_VISIBLE_DEVICES"] = resolved
        env["ROCR_VISIBLE_DEVICES"] = resolved
        return env
    raise ValueError(f"unknown backend: {backend!r}")


@dataclass(frozen=True)
class TestDflashLaunchArgs:
    draft_feature_mirror: bool = False
    peer_access: bool = False
    target_gpu: int | None = None
    draft_gpu: int | None = None
    target_gpus: str | None = None
    target_layer_split: str | None = None
    target_split_load_draft: bool = False
    target_split_dflash: bool = False
    draft_ipc_bin: str | None = None
    draft_ipc_gpu: int | None = None
    draft_ipc_work_dir: str | None = None
    draft_ipc_ring_cap: int | None = None
    max_ctx: int | None = None

    def to_cli_args(self) -> list[str]:
        out: list[str] = []
        if self.draft_feature_mirror:
            out.append("--draft-feature-mirror")
        if self.peer_access:
            out.append("--peer-access")
        if self.target_gpu is not None:
            out.append(f"--target-gpu={self.target_gpu}")
        if self.draft_gpu is not None:
            out.append(f"--draft-gpu={self.draft_gpu}")
        if self.target_gpus:
            out.append(f"--target-gpus={self.target_gpus}")
        if self.target_layer_split:
            out.append(f"--target-layer-split={self.target_layer_split}")
        if self.target_split_load_draft:
            out.append("--target-split-load-draft")
        if self.target_split_dflash:
            out.append("--target-split-dflash")
        if self.draft_ipc_bin:
            out.append(f"--draft-ipc-bin={self.draft_ipc_bin}")
        if self.draft_ipc_gpu is not None:
            out.append(f"--draft-ipc-gpu={self.draft_ipc_gpu}")
        if self.draft_ipc_work_dir:
            out.append(f"--draft-ipc-work-dir={self.draft_ipc_work_dir}")
        if self.draft_ipc_ring_cap is not None:
            out.append(f"--draft-ipc-ring-cap={self.draft_ipc_ring_cap}")
        if self.max_ctx is not None:
            out.append(f"--max-ctx={self.max_ctx}")
        return out
