#!/usr/bin/env python3
"""Terrain-specific checkpoint evaluator.

Extends the base CheckpointEvaluator with:
- Termination reason breakdown logged to wandb (eval/terminate_reason/*)
- Video keys use motion name instead of index
- Depth videos (_depth.mp4) excluded from video count and wandb upload
"""

import glob
import json
import os
from collections import Counter
from pathlib import Path

import wandb
from gear_sonic.eval_exp import CheckpointEvaluator
from loguru import logger


class TerrainCheckpointEvaluator(CheckpointEvaluator):

    def _log_metrics(self, eval_step: int, metrics_file: Path):
        """Log metrics + per-reason termination stats to wandb."""
        # Extract termination reasons from raw JSON before parent transforms it
        reason_metrics = {}
        try:
            with open(metrics_file) as f:
                raw = json.load(f)
            log_keys = raw.get("log_keys")
            all_dict = raw.get("eval/all_metrics_dict", {})
            terminate_reasons = all_dict.get("terminate_reason", [])
            terminated = all_dict.get("terminated", [])
            if terminate_reasons and terminated:
                reason_counts = Counter(r for r, t in zip(terminate_reasons, terminated) if t)
                total = len(terminated)
                num_failed = sum(1 for t in terminated if t)
                reason_metrics = {
                    f"eval/terminate_reason/{reason}": count
                    for reason, count in reason_counts.items()
                }
                reason_metrics["eval/terminate_reason/total_failed"] = num_failed
                reason_metrics["eval/terminate_reason/total_success"] = total - num_failed
                if log_keys is not None:
                    reason_metrics = {f"{log_keys}/{k}": v for k, v in reason_metrics.items()}
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error extracting termination reasons from {metrics_file}: {e}")

        # Use parent helper (includes 20 MB guard, HTML table gating, log_keys prefixing)
        metrics = self._load_metrics(eval_step, metrics_file)
        if metrics:
            # Merge termination reasons into a single wandb.log() call
            metrics.update(reason_metrics)
            wandb.log(metrics)

    def _log_videos(self, eval_step: int, metrics_file: Path, video_dir: Path):
        """Log render videos to wandb, keyed by motion name, excluding depth videos."""
        if not video_dir.exists():
            return

        log_keys = None
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
                    log_keys = metrics.get("log_keys")
            except Exception as e:
                logger.error(f"Error getting log_keys from metrics file: {e}")

        video_files = sorted(
            f
            for f in video_dir.iterdir()
            if f.is_file() and f.name.endswith(".mp4") and "_depth" not in f.name
        )

        if not video_files:
            return

        prefix = f"videos_hard_{log_keys}" if log_keys else "videos_hard"
        wandb_videos = {
            f"{prefix}/{vf.stem}": wandb.Video(str(vf), format="mp4")
            for vf in reversed(video_files)
        }
        wandb_videos["eval_step"] = eval_step
        wandb.log(wandb_videos)

    def evaluate_checkpoint(
        self,
        checkpoint_path,
        mode="metrics",
        work_dir=None,
        eval_step=None,
        eval_dataset=None,
        num_render_videos=None,
        eval_mode=None,
    ):
        """Override to fix depth video counting in render mode."""
        success = super().evaluate_checkpoint(
            checkpoint_path,
            mode=mode,
            work_dir=work_dir,
            eval_step=eval_step,
            eval_dataset=eval_dataset,
            num_render_videos=num_render_videos,
            eval_mode=eval_mode,
        )

        # Re-check render success: parent counts all mp4s including _depth.
        # We only care about RGB videos.
        if not success and mode == "render" and work_dir:
            all_mp4s = glob.glob(os.path.join(work_dir, "render_results", "*.mp4"))
            rgb_count = len([f for f in all_mp4s if "_depth" not in os.path.basename(f)])
            if rgb_count > 0:
                metrics_file = os.path.join(work_dir, "metrics_eval.json")
                if os.path.exists(metrics_file):
                    logger.info(
                        f"[render] Found {rgb_count} RGB videos (depth excluded), marking as success"
                    )
                    success = True

        return success
