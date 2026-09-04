#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Edit LeRobot datasets using various transformation tools.

Requires: pip install 'lerobot[dataset]'

This script allows you to delete episodes, split datasets, merge datasets,
remove features, modify tasks, recompute stats, and convert image datasets to video format.
When new_repo_id is specified, creates a new dataset.

Path semantics (v2): --root and --new_root are exact dataset folders containing
meta/, data/, videos/. When omitted, defaults to $HF_LEROBOT_HOME/{repo_id}.

Usage Examples:

Delete episodes 0, 2, and 5 from a dataset:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type delete_episodes \
        --operation.episode_indices "[0, 2, 5]"

Delete episodes from a local dataset at a specific path:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --root /path/to/pusht \
        --operation.type delete_episodes \
        --operation.episode_indices "[0, 2, 5]"

Delete episodes and save to a new dataset at a specific path and with a new repo_id:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --new_repo_id lerobot/pusht_filtered \
        --new_root /path/to/pusht_filtered \
        --operation.type delete_episodes \
        --operation.episode_indices "[0, 2, 5]"

Split dataset by fractions (pusht_train, pusht_val):
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type split \
        --operation.splits '{"train": 0.8, "val": 0.2}'

Split dataset by fractions and save split datasets to a specific folder (base_folder/train, base_folder/val):
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --new_root /path/to/base_folder \
        --operation.type split \
        --operation.splits '{"train": 0.8, "val": 0.2}'

Split dataset by episode indices:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type split \
        --operation.splits '{"train": [0, 1, 2, 3], "val": [4, 5]}'

Split into more than two splits:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type split \
        --operation.splits '{"train": 0.6, "val": 0.2, "test": 0.2}'

Merge multiple datasets:
    lerobot-edit-dataset \
        --new_repo_id lerobot/pusht_merged \
        --operation.type merge \
        --operation.repo_ids "['lerobot/pusht_train', 'lerobot/pusht_val']"

Merge multiple datasets to a specific output path:
    lerobot-edit-dataset \
        --new_repo_id lerobot/pusht_merged \
        --new_root /path/to/pusht_merged \
        --operation.type merge \
        --operation.repo_ids "['lerobot/pusht_train', 'lerobot/pusht_val']"

Merge multiple datasets from a list of local dataset paths:
    lerobot-edit-dataset \
        --new_repo_id lerobot/pusht_merged \
        --operation.type merge \
        --operation.repo_ids "['pusht_train', 'pusht_val']" \
        --operation.roots "['/path/to/pusht_train', '/path/to/pusht_val']"

Merge multiple datasets while keeping one file per source file (no video/data stitching):
    lerobot-edit-dataset \
        --new_repo_id lerobot/pusht_merged \
        --operation.type merge \
        --operation.repo_ids "['lerobot/pusht_train', 'lerobot/pusht_val']" \
        --operation.concatenate_videos false \
        --operation.concatenate_data false

Remove camera feature:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type remove_feature \
        --operation.feature_names "['observation.image']"

Modify tasks - set a single task for all episodes (WARNING: modifies in-place):
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type modify_tasks \
        --operation.new_task "Pick up the cube and place it"

Modify tasks - set different tasks for specific episodes (WARNING: modifies in-place):
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type modify_tasks \
        --operation.episode_tasks '{"0": "Task A", "1": "Task B", "2": "Task A"}'

Modify tasks - set default task with overrides for specific episodes (WARNING: modifies in-place):
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type modify_tasks \
        --operation.new_task "Default task" \
        --operation.episode_tasks '{"5": "Special task for episode 5"}'

Modify tasks - replace existing task strings in-place (WARNING: modifies in-place):
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type modify_tasks \
        --operation.task_replacements '{"Pick up the red cube": "Lift the red cube"}'

Convert image dataset to video format and save locally:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht_image \
        --new_root /path/to/output/pusht_video \
        --operation.type convert_image_to_video

Convert image dataset (with depth maps) to video format, customizing the depth encoder:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht_image \
        --new_root /path/to/output/pusht_video \
        --operation.type convert_image_to_video \
        --operation.depth_encoder.depth_min 0.01 \
        --operation.depth_encoder.depth_max 10.0 \
        --operation.depth_encoder.use_log true

Convert image dataset to video format and save with new repo_id:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht_image \
        --new_repo_id lerobot/pusht_video \
        --operation.type convert_image_to_video

Convert image dataset to video format and push to hub:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht_image \
        --new_repo_id lerobot/pusht_video \
        --operation.type convert_image_to_video \
        --push_to_hub true

Show dataset information:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht_image \
        --operation.type info \
        --operation.show_features true

Show dataset information without feature details:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht_image \
        --operation.type info \
        --operation.show_features false

Recompute dataset statistics (saves to lerobot/pusht_recomputed_stats by default):
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type recompute_stats

Recompute stats and save to a specific new repo_id:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --new_repo_id lerobot/pusht_new_stats \
        --operation.type recompute_stats

Recompute stats in-place (overwrites original dataset stats):
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --new_repo_id lerobot/pusht \
        --operation.type recompute_stats \
        --operation.overwrite true

Recompute stats for relative actions and push to hub:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type recompute_stats \
        --operation.relative_action true \
        --operation.chunk_size 50 \
        --operation.relative_exclude_joints "['gripper']" \
        --operation.num_workers 4 \
        --push_to_hub true

Re-encode all videos in a dataset (saves to lerobot/pusht_reencoded by default):
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type reencode_videos \
        --operation.rgb_encoder.vcodec h264 \
        --operation.rgb_encoder.pix_fmt yuv420p \
        --operation.rgb_encoder.crf 23

Re-encode videos into a new dataset using 4 parallel processes:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --new_repo_id lerobot/pusht_h264 \
        --operation.type reencode_videos \
        --operation.rgb_encoder.vcodec h264 \
        --operation.rgb_encoder.crf 23 \
        --operation.num_workers 4

Re-encode videos in-place (overwrites original dataset):
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --new_repo_id lerobot/pusht \
        --operation.type reencode_videos \
        --operation.rgb_encoder.vcodec h264 \
        --operation.overwrite true

Re-encode RGB videos using CUDA and NVIDIA NVENC:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type reencode_videos \
        --operation.rgb_encoder.vcodec h264 \
        --operation.use_gpu true

Re-encode both RGB and depth videos in a dataset (depth quantization params are preserved):
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --new_repo_id lerobot/pusht_h264 \
        --operation.type reencode_videos \
        --operation.rgb_encoder.vcodec h264 \
        --operation.depth_encoder.vcodec h264 \
        --operation.rgb_encoder.crf 23 \
        --operation.depth_encoder.crf 23 \
        --operation.num_workers 4

Resize videos in-place to 640x480 and back up the original dataset as a zip archive:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type resize_videos \
        --operation.width 640 \
        --operation.height 480

Resize videos using CUDA decode, scaling, and NVIDIA hardware encoding:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type resize_videos \
        --operation.width 640 \
        --operation.height 480 \
        --operation.use_gpu true

Resize videos in-place without creating a zip backup:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --new_repo_id lerobot/pusht \
        --operation.type resize_videos \
        --operation.width 640 \
        --operation.height 480 \
        --operation.overwrite true

Resize videos to a new dataset:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --new_repo_id lerobot/pusht_resized \
        --operation.type resize_videos \
        --operation.width 640 \
        --operation.height 480

Re-encode both RGB and depth videos in a dataset (depth quantization params are preserved):
    lerobot-edit-dataset \
        --repo_id lerobot/pusht_depth \
        --operation.type reencode_videos \
        --operation.rgb_encoder.vcodec h264 \
        --operation.depth_encoder.extra_options '{"x265-params": "lossless=1"}'

Rename a camera feature from wrist to hand:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type rename_cameras \
        --operation.renames '{"wrist": "hand"}'

Change the dataset robot type:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type change_robot_type \
        --operation.robot_type so_follower

Change follower and leader robot types in a dataset that stores both fields:
    lerobot-edit-dataset \
        --repo_id lerobot/pusht \
        --operation.type change_robot_type \
        --operation.follower_robot_type so_follower \
        --operation.leader_robot_type so_leader

Using JSON config file:
    lerobot-edit-dataset \
        --config_path path/to/edit_config.json
"""

import abc
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import pyarrow.parquet as pq
from tqdm import tqdm

from lerobot.configs import (
    DepthEncoderConfig,
    RGBEncoderConfig,
    depth_encoder_defaults,
    parser,
    rgb_encoder_defaults,
)
from lerobot.datasets import (
    LeRobotDataset,
    convert_image_to_video_dataset,
    delete_episodes,
    merge_datasets,
    modify_tasks,
    recompute_stats,
    reencode_dataset,
    remove_feature,
    split_dataset,
)
from lerobot.datasets.io_utils import write_info
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging


@dataclass
class OperationConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@OperationConfig.register_subclass("delete_episodes")
@dataclass
class DeleteEpisodesConfig(OperationConfig):
    episode_indices: list[int] | None = None


@OperationConfig.register_subclass("rename_cameras")
@dataclass
class RenameCamerasConfig(OperationConfig):
    renames: dict[str, str] | None = None


@OperationConfig.register_subclass("change_robot_type")
@dataclass
class ChangeRobotTypeConfig(OperationConfig):
    robot_type: str | None = None
    follower_robot_type: str | None = None
    leader_robot_type: str | None = None


@OperationConfig.register_subclass("resize_videos")
@dataclass
class ResizeVideosConfig(OperationConfig):
    width: int
    height: int
    overwrite: bool = False
    use_gpu: bool = False


@OperationConfig.register_subclass("split")
@dataclass
class SplitConfig(OperationConfig):
    splits: dict[str, float | list[int]] | None = None


@OperationConfig.register_subclass("merge")
@dataclass
class MergeConfig(OperationConfig):
    repo_ids: list[str] | None = None
    roots: list[str] | None = None
    # When False, keep one file per source file instead of packing into shards.
    concatenate_videos: bool = True
    concatenate_data: bool = True


@OperationConfig.register_subclass("remove_feature")
@dataclass
class RemoveFeatureConfig(OperationConfig):
    feature_names: list[str] | None = None


@OperationConfig.register_subclass("modify_tasks")
@dataclass
class ModifyTasksConfig(OperationConfig):
    new_task: str | None = None
    episode_tasks: dict[str, str] | None = None
    task_replacements: dict[str, str] | None = None


@OperationConfig.register_subclass("convert_image_to_video")
@dataclass
class ConvertImageToVideoConfig(OperationConfig):
    output_dir: str | None = None
    rgb_encoder: RGBEncoderConfig = field(default_factory=rgb_encoder_defaults)
    depth_encoder: DepthEncoderConfig = field(default_factory=depth_encoder_defaults)
    episode_indices: list[int] | None = None
    num_workers: int = 4
    max_episodes_per_batch: int | None = None
    max_frames_per_batch: int | None = None


@OperationConfig.register_subclass("recompute_stats")
@dataclass
class RecomputeStatsConfig(OperationConfig):
    skip_image_video: bool = True
    relative_action: bool = False
    relative_exclude_joints: list[str] | None = None
    chunk_size: int = 50
    num_workers: int = 0
    overwrite: bool = False


@OperationConfig.register_subclass("reencode_videos")
@dataclass
class ReencodeVideosConfig(OperationConfig):
    rgb_encoder: RGBEncoderConfig = field(default_factory=rgb_encoder_defaults)
    depth_encoder: DepthEncoderConfig = field(default_factory=depth_encoder_defaults)
    num_workers: int = 0
    encoder_threads: int | None = None
    overwrite: bool = False
    use_gpu: bool = False


@OperationConfig.register_subclass("info")
@dataclass
class InfoConfig(OperationConfig):
    show_features: bool = False


@dataclass
class EditDatasetConfig:
    # Operation configuration.
    operation: OperationConfig
    # Input dataset identifier. Always required unless for Merge operation.
    repo_id: str | None = None
    # Root directory where the input dataset is stored. If not specified, defaults to $HF_LEROBOT_HOME/repo_id.
    root: str | None = None
    # Edited dataset identifier. When both new_repo_id (resp. new_root) and repo_id (resp. root) are identical, modifications are applied in-place and a backup of the original dataset is created. Required for Merge operation.
    new_repo_id: str | None = None
    # Root directory where the edited dataset will be stored. If not specified, defaults to $HF_LEROBOT_HOME/new_repo_id. For Split operation, this is the base directory for the split datasets.
    new_root: str | None = None
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = False


def _resolve_io_paths(
    repo_id: str,
    new_repo_id: str | None,
    root: Path | str | None,
    new_root: Path | str | None,
    default_new_repo_id: str | None = None,
) -> tuple[str, Path, Path]:
    """Resolve input/output paths and repo_id for dataset operations.

    Returns (output_repo_id, input_path, output_path) with resolved (symlink-safe) paths.
    """
    input_path = (Path(root) if root else HF_LEROBOT_HOME / repo_id).resolve()
    output_repo_id = new_repo_id or default_new_repo_id or repo_id
    output_path = (Path(new_root) if new_root else HF_LEROBOT_HOME / output_repo_id).resolve()
    return output_repo_id, input_path, output_path


def _is_in_place(input_path: Path, output_path: Path) -> bool:
    """Whether both paths point to the same dataset directory.

    Uses os.path.samefile (device+inode) which is robust to case-insensitive filesystems, hardlinks
    and symlinks.
    """
    try:
        return os.path.samefile(input_path, output_path)
    except OSError:
        return False


def _backup_dataset_if_requested(dataset_root: Path) -> None:
    backup_archive = dataset_root.with_name(dataset_root.name + "_old.zip")
    if backup_archive.exists():
        logging.warning(f"Using existing original dataset backup: {backup_archive}")
        return

    try:
        response = input(f"Create backup archive {backup_archive}? [Y/n] ").strip().lower()
    except EOFError:
        response = ""
        logging.warning("No interactive input available; creating the backup by default")

    if response in {"n", "no"}:
        logging.warning("Skipping original dataset backup")
        return

    logging.info(f"Backing up original dataset to {backup_archive}")
    shutil.make_archive(str(backup_archive.with_suffix("")), "zip", root_dir=dataset_root)


def _choose_reencode_output() -> int:
    try:
        response = input(
            "Choose re-encode output: [1] create a new *_h264 dataset, "
            "[2] overwrite the source dataset (default: 2): "
        ).strip()
    except EOFError:
        response = ""
        logging.warning("No interactive input available; choosing in-place re-encode by default")

    if response in {"", "2"}:
        return 2
    if response == "1":
        return 1
    raise ValueError("Invalid choice. Enter 1 for a new dataset or 2 to overwrite the source dataset.")


def _reencode_videos_with_gpu(
    dataset: LeRobotDataset,
    rgb_encoder: RGBEncoderConfig,
    encoder_threads: int | None,
) -> None:
    if dataset.meta.depth_keys:
        raise ValueError(
            "GPU re-encoding does not support depth videos. Re-encode depth videos without "
            "--operation.use_gpu."
        )

    codec_map = {"h264": "h264_nvenc", "hevc": "hevc_nvenc", "av1": "av1_nvenc"}
    encoder = codec_map.get(rgb_encoder.vcodec, rgb_encoder.vcodec)
    if not encoder.endswith("_nvenc"):
        raise ValueError(
            f"GPU re-encoding requires an NVIDIA-compatible codec (h264, hevc, or av1), got {rgb_encoder.vcodec!r}"
        )

    video_files = sorted((dataset.root / "videos").rglob("*.mp4"))
    if not video_files:
        logging.warning("Dataset has no videos to re-encode")
        return

    logging.info("Re-encoding %d video file(s) with CUDA and %s", len(video_files), encoder)
    for video_file in tqdm(video_files, desc="GPU re-encoding videos"):
        temp_output_path = video_file.with_name(f"{video_file.stem}_temp{video_file.suffix}")
        cmd = [
            "ffmpeg",
            "-hwaccel",
            "cuda",
            "-i",
            str(video_file),
            "-c:v",
            encoder,
            "-pix_fmt",
            rgb_encoder.pix_fmt,
            "-c:a",
            "copy",
            "-y",
            str(temp_output_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logging.error("FFmpeg GPU error for %s: %s", video_file, result.stderr)
                raise RuntimeError(f"Failed to GPU re-encode video {video_file}")
            os.replace(temp_output_path, video_file)
        except Exception:
            if temp_output_path.exists():
                temp_output_path.unlink()
            raise

    for video_key in dataset.meta.video_keys:
        dataset.meta.update_video_info(video_key=video_key, video_encoder=rgb_encoder)
    write_info(dataset.meta.info, dataset.meta.root)


def get_output_path(
    repo_id: str,
    new_repo_id: str | None,
    root: Path | str | None,
    new_root: Path | str | None,
) -> tuple[str, Path, Path | None]:
    output_repo_id, input_path, output_path = _resolve_io_paths(repo_id, new_repo_id, root, new_root)

    # In case of in-place modification, create a backup of the original dataset (if it exists).
    backup_path: Path | None = None
    if _is_in_place(input_path, output_path):
        backup_path = input_path.with_name(input_path.name + "_old")
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.move(input_path, backup_path)

    return output_repo_id, output_path, backup_path


def handle_delete_episodes(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, DeleteEpisodesConfig):
        raise ValueError("Operation config must be DeleteEpisodesConfig")

    if not cfg.operation.episode_indices:
        raise ValueError("episode_indices must be specified for delete_episodes operation")

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    output_repo_id, output_dir, backup_path = get_output_path(
        cfg.repo_id,
        new_repo_id=cfg.new_repo_id,
        root=cfg.root,
        new_root=cfg.new_root,
    )

    # In case of in-place modification, make the dataset point to the backup directory
    if backup_path is not None:
        dataset.root = backup_path

    logging.info(f"Deleting episodes {cfg.operation.episode_indices} from {cfg.repo_id}")
    new_dataset = delete_episodes(
        dataset,
        episode_indices=cfg.operation.episode_indices,
        output_dir=output_dir,
        repo_id=output_repo_id,
    )

    logging.info(f"Dataset saved to {output_dir}")
    logging.info(f"Episodes: {new_dataset.meta.total_episodes}, Frames: {new_dataset.meta.total_frames}")

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {output_repo_id}")
        LeRobotDataset(output_repo_id, root=output_dir).push_to_hub()


def handle_split(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, SplitConfig):
        raise ValueError("Operation config must be SplitConfig")

    if not cfg.operation.splits:
        raise ValueError(
            "splits dict must be specified with split names as keys and fractions/episode lists as values"
        )

    if cfg.new_repo_id is not None:
        logging.warning(
            "split uses the original dataset identifier --repo_id to generate split names. The --new_repo_id parameter is ignored."
        )

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)

    logging.info(f"Splitting dataset {cfg.repo_id} with splits: {cfg.operation.splits}")
    split_datasets = split_dataset(
        dataset,
        splits=cfg.operation.splits,
        output_dir=cfg.new_root,
    )

    for split_name, split_ds in split_datasets.items():
        logging.info(
            f"{split_name}: {split_ds.meta.total_episodes} episodes, {split_ds.meta.total_frames} frames"
        )

        if cfg.push_to_hub:
            logging.info(f"Pushing {split_name} split to hub as {split_ds.repo_id}")
            LeRobotDataset(split_ds.repo_id, root=split_ds.root).push_to_hub()


def _merge_compatibility_issues(datasets: list[LeRobotDataset]) -> list[str]:
    reference = datasets[0].meta
    issues: list[str] = []

    for index, dataset in enumerate(datasets[1:], start=1):
        meta = dataset.meta
        label = dataset.repo_id

        if meta.robot_type != reference.robot_type:
            issues.append(
                f"robot_type differs: dataset 0={reference.robot_type!r}, {label}={meta.robot_type!r}"
            )
        if meta.fps != reference.fps:
            issues.append(f"fps differs: dataset 0={reference.fps}, {label}={meta.fps}")

        reference_keys = set(reference.features)
        dataset_keys = set(meta.features)
        if reference_keys != dataset_keys:
            issues.append(
                f"feature keys differ: dataset 0={sorted(reference_keys)}, "
                f"{label}={sorted(dataset_keys)}"
            )
        for feature_key in sorted(reference_keys & dataset_keys):
            reference_feature = reference.features[feature_key]
            dataset_feature = meta.features[feature_key]
            reference_structure = {
                key: reference_feature.get(key)
                for key in ("dtype", "shape", "names")
            }
            dataset_structure = {
                key: dataset_feature.get(key)
                for key in ("dtype", "shape", "names")
            }
            if reference_structure != dataset_structure:
                issues.append(
                    f"feature {feature_key!r} differs: dataset 0={reference_structure}, "
                    f"{label}={dataset_structure}"
                )

            if reference_feature.get("dtype") == "video":
                reference_info = reference_feature.get("info") or {}
                dataset_info = dataset_feature.get("info") or {}
                for info_key in ("video.codec", "video.width", "video.height", "video.fps"):
                    if reference_info.get(info_key) != dataset_info.get(info_key):
                        issues.append(
                            f"feature {feature_key!r} {info_key} differs: "
                            f"dataset 0={reference_info.get(info_key)!r}, "
                            f"{label}={dataset_info.get(info_key)!r}"
                        )

        for metadata_key in ("storage_format", "data_path", "video_path"):
            if getattr(meta, metadata_key) != getattr(reference, metadata_key):
                issues.append(
                    f"{metadata_key} differs: dataset 0={getattr(reference, metadata_key)!r}, "
                    f"{label}={getattr(meta, metadata_key)!r}"
                )

    return issues


def _validate_merge_compatibility(datasets: list[LeRobotDataset]) -> None:
    logging.info("Checking merge compatibility for %d datasets", len(datasets))
    issues = _merge_compatibility_issues(datasets)
    categories = {
        "robot_type": any(issue.startswith("robot_type ") for issue in issues),
        "fps": any(issue.startswith("fps ") for issue in issues),
        "feature structure": any(
            issue.startswith("feature keys ")
            or (issue.startswith("feature ") and not any(
                f" {key} differs:" in issue
                for key in ("video.codec", "video.width", "video.height", "video.fps")
            ))
            for issue in issues
        ),
        "video codec/resolution/fps": any(
            any(
                f" {key} differs:" in issue
                for key in ("video.codec", "video.width", "video.height", "video.fps")
            )
            for issue in issues
        ),
        "storage metadata": any(
            issue.startswith(("storage_format ", "data_path ", "video_path ")) for issue in issues
        ),
    }
    for check, failed in categories.items():
        log = logging.error if failed else logging.info
        log("Merge compatibility: %s: %s", check, "FAILED" if failed else "OK")

    if issues:
        logging.error("Merge compatibility check failed with %d difference(s):", len(issues))
        for issue in issues:
            logging.error("  - %s", issue)
        raise ValueError(
            "Datasets are not compatible for merge. Resolve the differences reported above "
            "before merging."
        )

def handle_merge(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, MergeConfig):
        raise ValueError("Operation config must be MergeConfig")

    if not cfg.operation.repo_ids:
        raise ValueError("repo_ids must be specified for merge operation")

    if cfg.repo_id is not None or cfg.root is not None:
        logging.warning(
            "merge uses --new_repo_id and --new_root for the merged dataset. The --repo_id and --root parameters are ignored."
        )

    if cfg.operation.roots:
        if len(cfg.operation.roots) != len(cfg.operation.repo_ids):
            raise ValueError("repo_ids and roots must have the same length for merge operation")
        logging.info(f"Loading {len(cfg.operation.roots)} datasets to merge")
        datasets = [
            LeRobotDataset(repo_id=repo_id, root=root)
            for repo_id, root in zip(cfg.operation.repo_ids, cfg.operation.roots, strict=True)
        ]
    else:
        logging.info(f"Loading {len(cfg.operation.repo_ids)} datasets to merge")
        datasets = [LeRobotDataset(repo_id) for repo_id in cfg.operation.repo_ids]

    _validate_merge_compatibility(datasets)

    output_dir = Path(cfg.new_root) if cfg.new_root else HF_LEROBOT_HOME / cfg.new_repo_id

    logging.info(f"Merging datasets into {cfg.new_repo_id}")
    merged_dataset = merge_datasets(
        datasets,
        output_repo_id=cfg.new_repo_id,
        output_dir=output_dir,
        concatenate_videos=cfg.operation.concatenate_videos,
        concatenate_data=cfg.operation.concatenate_data,
    )

    logging.info(f"Merged dataset saved to {output_dir}")
    logging.info(
        f"Episodes: {merged_dataset.meta.total_episodes}, Frames: {merged_dataset.meta.total_frames}"
    )

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {cfg.new_repo_id}")
        LeRobotDataset(merged_dataset.repo_id, root=output_dir).push_to_hub()


def handle_remove_feature(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, RemoveFeatureConfig):
        raise ValueError("Operation config must be RemoveFeatureConfig")

    if not cfg.operation.feature_names:
        raise ValueError("feature_names must be specified for remove_feature operation")

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    output_repo_id, output_dir, backup_path = get_output_path(
        cfg.repo_id,
        new_repo_id=cfg.new_repo_id,
        root=cfg.root,
        new_root=cfg.new_root,
    )

    # In case of in-place modification, make the dataset point to the backup directory
    if backup_path is not None:
        dataset.root = backup_path

    logging.info(f"Removing features {cfg.operation.feature_names} from {cfg.repo_id}")
    new_dataset = remove_feature(
        dataset,
        feature_names=cfg.operation.feature_names,
        output_dir=output_dir,
        repo_id=output_repo_id,
    )

    logging.info(f"Dataset saved to {output_dir}")
    logging.info(f"Remaining features: {list(new_dataset.meta.features.keys())}")

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {output_repo_id}")
        LeRobotDataset(output_repo_id, root=output_dir).push_to_hub()


def handle_modify_tasks(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, ModifyTasksConfig):
        raise ValueError("Operation config must be ModifyTasksConfig")

    new_task = cfg.operation.new_task
    episode_tasks_raw = cfg.operation.episode_tasks
    task_replacements = cfg.operation.task_replacements

    if new_task is None and episode_tasks_raw is None and task_replacements is None:
        raise ValueError(
            "Must specify at least one of new_task, episode_tasks, or task_replacements for modify_tasks operation"
        )

    if cfg.new_repo_id is not None or cfg.new_root is not None:
        logging.warning(
            "modify_tasks modifies datasets in-place. The --new_repo_id and --new_root parameters are ignored."
        )

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    logging.warning(f"Modifying dataset in-place at {dataset.root}. Original data will be overwritten.")

    # Convert episode_tasks keys from string to int if needed (CLI passes strings)
    episode_tasks: dict[int, str] | None = None
    if episode_tasks_raw is not None:
        episode_tasks = {int(k): v for k, v in episode_tasks_raw.items()}

    logging.info(f"Modifying tasks in {cfg.repo_id}")
    if new_task:
        logging.info(f"  Default task: '{new_task}'")
    if episode_tasks:
        logging.info(f"  Episode-specific tasks: {episode_tasks}")
    if task_replacements:
        logging.info(f"  Task replacements: {task_replacements}")

    modified_dataset = modify_tasks(
        dataset,
        new_task=new_task,
        episode_tasks=episode_tasks,
        task_replacements=task_replacements,
    )

    logging.info(f"Dataset modified at {dataset.root}")
    logging.info(f"Tasks: {list(modified_dataset.meta.tasks.index)}")

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {cfg.repo_id}")
        modified_dataset.push_to_hub()


def handle_convert_image_to_video(cfg: EditDatasetConfig) -> None:
    # Note: Parser may create any config type with the right fields, so we access fields directly
    # instead of checking isinstance()
    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)

    output_dir_config = getattr(cfg.operation, "output_dir", None)
    if output_dir_config:
        logging.warning(
            "--operation.output_dir is deprecated and will be removed in future versions. "
            "Please use --new_root instead."
        )

    if cfg.new_root:
        output_dir = Path(cfg.new_root)
        output_repo_id = cfg.new_repo_id or f"{cfg.repo_id}_video"
        logging.info(f"Saving to new_root: {output_dir} as {output_repo_id}")
    elif cfg.new_repo_id:
        output_repo_id = cfg.new_repo_id
        output_dir = HF_LEROBOT_HOME / cfg.new_repo_id
        logging.info(f"Saving to new dataset: {cfg.new_repo_id} at {output_dir}")
    elif output_dir_config:
        output_dir = Path(output_dir_config)
        output_repo_id = output_dir.name
        logging.info(f"Saving to local directory: {output_dir} as {output_repo_id}")
    else:
        output_repo_id = f"{cfg.repo_id}_video"
        output_dir = HF_LEROBOT_HOME / output_repo_id
        logging.info(f"Saving to auto-generated location: {output_dir} as {output_repo_id}")

    logging.info(f"Converting dataset {cfg.repo_id} to video format")

    new_dataset = convert_image_to_video_dataset(
        dataset=dataset,
        output_dir=output_dir,
        repo_id=output_repo_id,
        rgb_encoder=getattr(cfg.operation, "rgb_encoder", None) or rgb_encoder_defaults(),
        depth_encoder=getattr(cfg.operation, "depth_encoder", None) or depth_encoder_defaults(),
        episode_indices=getattr(cfg.operation, "episode_indices", None),
        num_workers=getattr(cfg.operation, "num_workers", 4),
        max_episodes_per_batch=getattr(cfg.operation, "max_episodes_per_batch", None),
        max_frames_per_batch=getattr(cfg.operation, "max_frames_per_batch", None),
    )

    logging.info("Video dataset created successfully!")
    logging.info(f"Location: {output_dir}")
    logging.info(f"Episodes: {new_dataset.meta.total_episodes}")
    logging.info(f"Frames: {new_dataset.meta.total_frames}")

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {output_repo_id}...")
        new_dataset.push_to_hub()
        logging.info("✓ Successfully pushed to hub!")
    else:
        logging.info("Dataset saved locally (not pushed to hub)")


def handle_recompute_stats(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, RecomputeStatsConfig):
        raise ValueError("Operation config must be RecomputeStatsConfig")

    output_repo_id, input_root, output_root = _resolve_io_paths(
        cfg.repo_id,
        cfg.new_repo_id,
        cfg.root,
        cfg.new_root,
        default_new_repo_id=f"{cfg.repo_id}_recomputed_stats",
    )
    in_place = _is_in_place(input_root, output_root)

    if in_place and not cfg.operation.overwrite:
        raise ValueError(
            f"recompute_stats would overwrite the dataset in-place at {input_root}. "
            "Pass --operation.overwrite true to allow in-place modification, "
            "or use --new_repo_id / --new_root to write to a different location. "
            f"Default output repo_id when neither is set: '{cfg.repo_id}_recomputed_stats'."
        )

    if in_place:
        logging.warning(
            f"Overwriting dataset stats in-place at {input_root}. The original stats will be lost."
        )
        dataset = LeRobotDataset(cfg.repo_id, root=input_root)
    else:
        logging.info(f"Copying dataset from {input_root} to {output_root}")
        if output_root.exists():
            backup_path = output_root.with_name(output_root.name + "_old")
            logging.warning(f"Output directory {output_root} already exists. Moving to {backup_path}")
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.move(output_root, backup_path)
        shutil.copytree(input_root, output_root)
        dataset = LeRobotDataset(output_repo_id, root=output_root)

    logging.info(f"Recomputing stats for {cfg.repo_id}")
    if cfg.operation.relative_action:
        logging.info(
            f"Relative action stats enabled (chunk_size={cfg.operation.chunk_size}, "
            f"exclude_joints={cfg.operation.relative_exclude_joints})"
        )

    recompute_stats(
        dataset,
        skip_image_video=cfg.operation.skip_image_video,
        relative_action=cfg.operation.relative_action,
        relative_exclude_joints=cfg.operation.relative_exclude_joints,
        chunk_size=cfg.operation.chunk_size,
        num_workers=cfg.operation.num_workers,
    )

    logging.info(f"Stats written to {dataset.root}")

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {dataset.repo_id}...")
        dataset.push_to_hub()


def handle_reencode_videos(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, ReencodeVideosConfig):
        raise ValueError("Operation config must be ReencodeVideosConfig")

    output_choice = None
    if cfg.new_repo_id is None and cfg.new_root is None:
        output_choice = _choose_reencode_output()

    if output_choice == 2:
        input_root = (Path(cfg.root) if cfg.root else HF_LEROBOT_HOME / cfg.repo_id).resolve()
        output_repo_id = cfg.repo_id
        output_root = input_root
    else:
        output_repo_id, input_root, output_root = _resolve_io_paths(
            cfg.repo_id,
            cfg.new_repo_id,
            cfg.root,
            cfg.new_root,
            default_new_repo_id=f"{cfg.repo_id}_h264",
        )
    in_place = _is_in_place(input_root, output_root)

    if in_place and output_choice != 2 and not cfg.operation.overwrite:
        raise ValueError(
            f"reencode_videos would overwrite the dataset in-place at {input_root}. "
            "Pass --operation.overwrite true to allow in-place modification, "
            "or use --new_repo_id / --new_root to write to a different location. "
            f"Default output repo_id when neither is set: '{cfg.repo_id}_reencoded'."
        )

    if in_place:
        if output_choice == 2:
            if not input_root.is_dir():
                raise FileNotFoundError(f"Dataset directory does not exist: {input_root}")
            _backup_dataset_if_requested(input_root)
        logging.warning(f"Overwriting dataset videos in-place at {input_root}")
        dataset = LeRobotDataset(cfg.repo_id, root=input_root)
    else:
        logging.info(f"Copying dataset from {input_root} to {output_root}")
        if output_root.exists():
            backup_path = output_root.with_name(output_root.name + "_old")
            logging.warning(f"Output directory {output_root} already exists. Moving to {backup_path}")
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.move(output_root, backup_path)
        shutil.copytree(input_root, output_root)
        dataset = LeRobotDataset(output_repo_id, root=output_root)

    logging.info(
        f"Re-encoding videos in {output_repo_id} with RGB encoder {cfg.operation.rgb_encoder} "
        f"and depth encoder {cfg.operation.depth_encoder}"
    )
    if cfg.operation.use_gpu:
        _reencode_videos_with_gpu(
            dataset,
            rgb_encoder=cfg.operation.rgb_encoder,
            encoder_threads=cfg.operation.encoder_threads,
        )
    else:
        reencode_dataset(
            dataset,
            rgb_encoder=cfg.operation.rgb_encoder,
            depth_encoder=cfg.operation.depth_encoder,
            encoder_threads=cfg.operation.encoder_threads,
            num_workers=cfg.operation.num_workers,
        )

    logging.info(f"All videos re-encoded at {dataset.root}")

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {output_repo_id}...")
        dataset.push_to_hub()


def _normalize_camera_name(name: str) -> str:
    prefix = "observation.images."
    name = name.removeprefix(prefix)
    if not name or "/" in name:
        raise ValueError(f"Invalid camera name: {name!r}")
    return name


def _rename_camera_metadata(root: Path, renames: dict[str, str]) -> None:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset metadata file does not exist: {info_path}")

    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    episodes_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    video_root = root / "videos"

    for old_name, new_name in renames.items():
        old_name = _normalize_camera_name(old_name)
        new_name = _normalize_camera_name(new_name)
        if old_name == new_name:
            raise ValueError(f"Camera rename has identical source and destination: {old_name!r}")

        old_feature = f"observation.images.{old_name}"
        new_feature = f"observation.images.{new_name}"
        old_prefix = f"videos/{old_feature}/"
        new_prefix = f"videos/{new_feature}/"
        old_dir = video_root / old_feature
        new_dir = video_root / new_feature

        info_has_old = old_feature in features
        info_has_new = new_feature in features
        old_columns = {
            path: [column for column in pq.read_schema(path).names if column.startswith(old_prefix)]
            for path in episodes_files
        }
        new_columns = {
            path: [column for column in pq.read_schema(path).names if column.startswith(new_prefix)]
            for path in episodes_files
        }
        parquet_has_old = any(old_columns.values())
        parquet_has_new = any(new_columns.values())
        sources = sum((info_has_old, old_dir.is_dir(), parquet_has_old))
        if sources != 3:
            logging.warning(
                "Camera %r is only present in %d of 3 dataset locations (info.json, videos/, episodes parquet); "
                "renaming available entries only.",
                old_name,
                sources,
            )

        if info_has_old and info_has_new:
            raise ValueError(f"Cannot rename camera {old_name!r}: destination already exists in info.json")
        if old_dir.is_dir() and new_dir.exists():
            raise ValueError(f"Cannot rename camera {old_name!r}: destination directory already exists: {new_dir}")
        if parquet_has_old and parquet_has_new:
            raise ValueError(f"Cannot rename camera {old_name!r}: destination already exists in episodes parquet")

        if info_has_old:
            features[new_feature] = features.pop(old_feature)

        for path, columns in old_columns.items():
            if not columns:
                continue
            table = pq.read_table(path)
            renamed_columns = [
                column.replace(old_prefix, new_prefix, 1) if column.startswith(old_prefix) else column
                for column in table.column_names
            ]
            temp_path = path.with_suffix(".parquet.tmp")
            pq.write_table(table.rename_columns(renamed_columns), temp_path)
            os.replace(temp_path, path)

        if old_dir.is_dir():
            old_dir.rename(new_dir)

    temp_info_path = info_path.with_suffix(".json.tmp")
    temp_info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp_info_path, info_path)


def handle_rename_cameras(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, RenameCamerasConfig):
        raise ValueError("Operation config must be RenameCamerasConfig")
    if not cfg.operation.renames:
        raise ValueError("renames must be specified for rename_cameras operation")

    input_root = (Path(cfg.root) if cfg.root else HF_LEROBOT_HOME / cfg.repo_id).resolve()
    rename_in_place = cfg.new_repo_id is None and cfg.new_root is None
    if rename_in_place:
        output_repo_id = cfg.repo_id
        output_root = input_root
        if not input_root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {input_root}")
        _backup_dataset_if_requested(input_root)
    else:
        output_repo_id, input_root, output_root = _resolve_io_paths(
            cfg.repo_id,
            cfg.new_repo_id,
            cfg.root,
            cfg.new_root,
            default_new_repo_id=f"{cfg.repo_id}_renamed",
        )
        if output_root.exists():
            backup_path = output_root.with_name(output_root.name + "_old")
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.move(output_root, backup_path)
        shutil.copytree(input_root, output_root)

    _rename_camera_metadata(output_root, cfg.operation.renames)
    logging.info(f"Camera names renamed in {output_root}")

    if cfg.push_to_hub:
        dataset = LeRobotDataset(output_repo_id, root=output_root)
        dataset.push_to_hub()


def handle_change_robot_type(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, ChangeRobotTypeConfig):
        raise ValueError("Operation config must be ChangeRobotTypeConfig")
    requested = {
        "robot_type": cfg.operation.robot_type,
        "follower_robot_type": cfg.operation.follower_robot_type,
        "leader_robot_type": cfg.operation.leader_robot_type,
    }
    requested = {key: value for key, value in requested.items() if value is not None}
    if not requested:
        raise ValueError(
            "Specify at least one of robot_type, follower_robot_type, or leader_robot_type"
        )

    input_root = (Path(cfg.root) if cfg.root else HF_LEROBOT_HOME / cfg.repo_id).resolve()
    change_in_place = cfg.new_repo_id is None and cfg.new_root is None
    if change_in_place:
        output_repo_id = cfg.repo_id
        output_root = input_root
        if not input_root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {input_root}")
        _backup_dataset_if_requested(input_root)
    else:
        output_repo_id, input_root, output_root = _resolve_io_paths(
            cfg.repo_id,
            cfg.new_repo_id,
            cfg.root,
            cfg.new_root,
            default_new_repo_id=f"{cfg.repo_id}_robot_type_changed",
        )
        if output_root.exists():
            backup_path = output_root.with_name(output_root.name + "_old")
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.move(output_root, backup_path)
        shutil.copytree(input_root, output_root)

    info_path = output_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset metadata file does not exist: {info_path}")
    info = json.loads(info_path.read_text())
    for key, value in requested.items():
        if key not in info:
            logging.warning("Dataset metadata has no %r field; no value was changed", key)
            continue
        old_value = info[key]
        info[key] = value
        logging.info("Changed %s from %r to %r", key, old_value, value)

    temp_info_path = info_path.with_suffix(".json.tmp")
    temp_info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp_info_path, info_path)

    if cfg.push_to_hub:
        dataset = LeRobotDataset(output_repo_id, root=output_root)
        dataset.push_to_hub()


def handle_resize_videos(cfg: EditDatasetConfig) -> None:
    if not isinstance(cfg.operation, ResizeVideosConfig):
        raise ValueError("Operation config must be ResizeVideosConfig")

    if cfg.operation.width <= 0 or cfg.operation.height <= 0:
        raise ValueError("Width and height must be positive integers")

    input_root = (Path(cfg.root) if cfg.root else HF_LEROBOT_HOME / cfg.repo_id).resolve()
    resize_in_place = cfg.new_repo_id is None and cfg.new_root is None
    if resize_in_place:
        output_repo_id = cfg.repo_id
        output_root = input_root
    else:
        output_repo_id, input_root, output_root = _resolve_io_paths(
            cfg.repo_id,
            cfg.new_repo_id,
            cfg.root,
            cfg.new_root,
            default_new_repo_id=f"{cfg.repo_id}_resized",
        )

    in_place = _is_in_place(input_root, output_root)

    if in_place and not resize_in_place and not cfg.operation.overwrite:
        raise ValueError(
            f"resize_videos would overwrite the dataset in-place at {input_root}. "
            "Pass --operation.overwrite true to allow in-place modification, "
            "or use --new_repo_id / --new_root to write to a different location. "
            f"Default output repo_id when neither is set: '{cfg.repo_id}_resized'."
        )

    if resize_in_place:
        if not input_root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {input_root}")

        _backup_dataset_if_requested(input_root)
        logging.warning(f"Resizing videos in-place at {input_root}")
    elif in_place:
        logging.warning(
            f"Overwriting dataset videos in-place at {input_root}. The original videos will be lost."
        )
    else:
        logging.info(f"Copying dataset from {input_root} to {output_root}")
        if output_root.exists():
            backup_path = output_root.with_name(output_root.name + "_old")
            logging.warning(f"Output directory {output_root} already exists. Moving to {backup_path}")
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.move(output_root, backup_path)
        shutil.copytree(input_root, output_root)

    if cfg.operation.use_gpu:
        logging.info(
            f"Resizing videos in {output_repo_id} to {cfg.operation.width}x{cfg.operation.height} "
            "using CUDA and NVIDIA NVENC"
        )
    else:
        logging.info(
            f"Resizing videos in {output_repo_id} to {cfg.operation.width}x{cfg.operation.height} "
            "using CPU"
        )

    video_files = list(output_root.glob("videos/**/*.mp4"))
    if not video_files:
        logging.warning("No video files found to resize")
        return

    for video_file in tqdm(video_files, desc="Resizing videos"):
        temp_output_path = video_file.with_name(f"{video_file.stem}_temp{video_file.suffix}")
        try:
            if cfg.operation.use_gpu:
                cmd = [
                    "ffmpeg",
                    "-hwaccel",
                    "cuda",
                    "-hwaccel_output_format",
                    "cuda",
                    "-i",
                    str(video_file),
                    "-vf",
                    f"scale_cuda={cfg.operation.width}:{cfg.operation.height}",
                    "-c:v",
                    "h264_nvenc",
                    "-c:a",
                    "copy",
                    "-y",
                    str(temp_output_path),
                ]
            else:
                cmd = [
                    "ffmpeg",
                    "-i",
                    str(video_file),
                    "-vf",
                    f"scale={cfg.operation.width}:{cfg.operation.height}",
                    "-c:a",
                    "copy",
                    "-y",
                    str(temp_output_path),
                ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                mode = "GPU" if cfg.operation.use_gpu else "CPU"
                logging.error(f"FFmpeg {mode} error for {video_file}: {result.stderr}")
                raise RuntimeError(f"Failed to resize video {video_file} using {mode}")
            os.replace(temp_output_path, video_file)
        except Exception as e:
            if temp_output_path.exists():
                try:
                    temp_output_path.unlink()
                except Exception:
                    pass
            logging.error(f"Error processing video {video_file}: {e}")
            raise

    info_path = output_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset metadata file does not exist: {info_path}")

    info = json.loads(info_path.read_text())
    for feature in info.get("features", {}).values():
        if feature.get("dtype") != "video":
            continue
        shape = feature.get("shape")
        channels = shape[2] if isinstance(shape, list) and len(shape) == 3 else 3
        feature["shape"] = [cfg.operation.height, cfg.operation.width, channels]
        feature_info = feature.setdefault("info", {})
        feature_info["video.height"] = cfg.operation.height
        feature_info["video.width"] = cfg.operation.width

    temp_info_path = info_path.with_suffix(".json.tmp")
    temp_info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp_info_path, info_path)

    logging.info(
        f"All videos and video metadata resized to {cfg.operation.width}x{cfg.operation.height} "
        f"at {output_root}"
    )

    if cfg.push_to_hub:
        logging.info(f"Pushing to hub as {output_repo_id}...")
        dataset = LeRobotDataset(output_repo_id, root=output_root)
        dataset.push_to_hub()


def _get_dataset_size(repo_path):
    import os

    total = 0
    with os.scandir(repo_path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += _get_dataset_size(entry.path)
    return total


def handle_info(cfg: EditDatasetConfig):
    if not isinstance(cfg.operation, InfoConfig):
        raise ValueError("Operation config must be InfoConfig")

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    sys.stdout.write(f"======Info {dataset.meta.repo_id}\n")
    sys.stdout.write(f"Repository ID: {dataset.meta.repo_id} \n")
    sys.stdout.write(f"Total episode: {dataset.meta.total_episodes} \n")
    sys.stdout.write(f"Total task: {dataset.meta.total_tasks} \n")
    sys.stdout.write(f"Total frame(Actual Count): {dataset.meta.total_frames}({len(dataset)}) \n")
    sys.stdout.write(
        f"Average frame per episode: {dataset.meta.total_frames / dataset.meta.total_episodes:.1f}\n"
    )
    sys.stdout.write(
        f"Average episode time(sec): {(dataset.meta.total_frames / dataset.meta.total_episodes) / dataset.meta.fps:.1f}\n"
    )
    sys.stdout.write(f"FPS: {dataset.meta.fps}\n")

    total_file_size = _get_dataset_size(dataset.root)
    sys.stdout.write(f"Size: {total_file_size / (1024 * 1024):.1f} MB\n")
    if cfg.operation.show_features:
        import json

        feature_dump_str = json.dumps(
            dataset.meta.features, ensure_ascii=False, indent=4, sort_keys=True, separators=(",", ": ")
        )
        sys.stdout.write("Features:\n")
        sys.stdout.write(f"{feature_dump_str}\n")


def _validate_config(cfg: EditDatasetConfig) -> None:
    if isinstance(cfg.operation, MergeConfig):
        if not cfg.new_repo_id:
            raise ValueError("--new_repo_id is required for merge operation (the merged dataset identifier)")
    else:
        if not cfg.repo_id:
            raise ValueError(
                f"--repo_id is required for {cfg.operation.type} operation (the input dataset identifier)"
            )


@parser.wrap()
def edit_dataset(cfg: EditDatasetConfig) -> None:
    _validate_config(cfg)
    operation_type = cfg.operation.type

    if operation_type == "delete_episodes":
        handle_delete_episodes(cfg)
    elif operation_type == "split":
        handle_split(cfg)
    elif operation_type == "merge":
        handle_merge(cfg)
    elif operation_type == "remove_feature":
        handle_remove_feature(cfg)
    elif operation_type == "modify_tasks":
        handle_modify_tasks(cfg)
    elif operation_type == "convert_image_to_video":
        handle_convert_image_to_video(cfg)
    elif operation_type == "recompute_stats":
        handle_recompute_stats(cfg)
    elif operation_type == "reencode_videos":
        handle_reencode_videos(cfg)
    elif operation_type == "rename_cameras":
        handle_rename_cameras(cfg)
    elif operation_type == "change_robot_type":
        handle_change_robot_type(cfg)
    elif operation_type == "resize_videos":
        handle_resize_videos(cfg)
    elif operation_type == "info":
        handle_info(cfg)
    else:
        available = ", ".join(OperationConfig.get_known_choices())
        raise ValueError(f"Unknown operation: {operation_type}\nAvailable operations: {available}")


def main() -> None:
    init_logging()
    edit_dataset()


if __name__ == "__main__":
    main()
