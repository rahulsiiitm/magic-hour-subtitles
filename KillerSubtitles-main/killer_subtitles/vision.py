"""Sampled person, clutter, and motion analysis for smart caption placement."""

from __future__ import annotations

import gc
import time
from pathlib import Path

from .models import FrameAnalysis, VideoInfo, VisionConfig


class VisionAnalyzer:
    """Analyze reduced frames without retaining decoded video or model tensors."""

    _cache: dict[tuple, tuple[list[FrameAnalysis], str, float]] = {}

    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self.device = "unknown"
        self.elapsed_seconds = 0.0

    def analyze(
        self,
        video_path: str | Path,
        video_info: VideoInfo,
    ) -> list[FrameAnalysis]:
        path = Path(video_path)
        stat = path.stat()
        cache_key = (
            str(path.resolve()), stat.st_mtime_ns, stat.st_size,
            video_info.width, video_info.height, self.config,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            frames, self.device, self.elapsed_seconds = cached
            return frames

        try:
            import cv2
            import numpy as np
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Smart placement requires ultralytics, opencv-python-headless, and numpy."
            ) from exc

        started = time.perf_counter()
        requested_device = self.config.device
        if requested_device is None:
            requested_device = 0 if torch.cuda.is_available() else "cpu"
        self.device = (
            f"cuda:{requested_device}" if isinstance(requested_device, int)
            else str(requested_device)
        )
        model = YOLO(self.config.model_name)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            del model
            raise RuntimeError(f"Could not open video for visual analysis: {path}")

        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or video_info.fps or 30.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target_fps = min(max(0.1, self.config.analysis_fps), source_fps)
        frame_step = max(1, int(round(source_fps / target_fps)))
        if total_frames <= 0:
            total_frames = max(1, int(round(video_info.duration * source_fps)))

        analysis_width, analysis_height = _scaled_size(
            video_info.width,
            video_info.height,
            self.config.long_side,
        )
        map_width, map_height = _scaled_size(
            video_info.width,
            video_info.height,
            self.config.map_long_side,
        )
        analyses: list[FrameAnalysis] = []
        previous_gray = None

        try:
            for frame_index in range(0, total_frames, frame_step):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    continue
                reduced = cv2.resize(
                    frame,
                    (analysis_width, analysis_height),
                    interpolation=cv2.INTER_AREA,
                )
                gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
                result = model.predict(
                    source=reduced,
                    classes=[0],
                    conf=self.config.confidence,
                    imgsz=self.config.long_side,
                    device=requested_device,
                    verbose=False,
                )[0]
                person, confidence = _combined_person_mask(
                    result,
                    analysis_width,
                    analysis_height,
                    self.config.person_dilation,
                    cv2,
                    np,
                )
                clutter = cv2.Canny(gray, 100, 200)
                if previous_gray is None:
                    motion = np.zeros_like(gray)
                    scene_cut = False
                else:
                    motion = cv2.absdiff(gray, previous_gray)
                    scene_cut = (
                        float(motion.mean()) / 255.0
                        >= self.config.scene_cut_threshold
                    )
                previous_gray = gray

                timestamp = frame_index / source_fps
                analyses.append(FrameAnalysis(
                    timestamp=timestamp,
                    frame_index=frame_index,
                    map_width=map_width,
                    map_height=map_height,
                    person_map=_reduce_map(person, map_width, map_height, cv2),
                    clutter_map=_reduce_map(clutter, map_width, map_height, cv2),
                    motion_map=_reduce_map(motion, map_width, map_height, cv2),
                    person_confidence=confidence,
                    scene_cut=scene_cut,
                ))
        finally:
            capture.release()
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not analyses:
            raise RuntimeError("Visual analysis produced no sampled frames.")
        self.elapsed_seconds = time.perf_counter() - started
        self._cache[cache_key] = (analyses, self.device, self.elapsed_seconds)
        return analyses


def _scaled_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    scale = min(1.0, max(1, long_side) / max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _combined_person_mask(
    result,
    width: int,
    height: int,
    dilation: int,
    cv2,
    np,
):
    combined = np.zeros((height, width), dtype=np.uint8)
    confidence = 0.0
    if result.masks is not None and len(result.masks.data):
        masks = result.masks.data.detach().cpu().numpy()
        for mask in masks:
            resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
            combined = np.maximum(combined, (resized >= 0.5).astype(np.uint8) * 255)
        if result.boxes is not None and len(result.boxes.conf):
            confidence = float(result.boxes.conf.detach().max().cpu().item())
    if dilation > 0 and combined.any():
        size = max(1, int(dilation))
        kernel = np.ones((size, size), dtype=np.uint8)
        combined = cv2.dilate(combined, kernel, iterations=1)
    return combined, confidence


def _reduce_map(frame_map, width: int, height: int, cv2):
    reduced = cv2.resize(frame_map, (width, height), interpolation=cv2.INTER_AREA)
    return reduced.astype("uint8", copy=False)
