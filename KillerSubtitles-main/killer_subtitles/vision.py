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
                    classes=list(self.config.foreground_class_ids),
                    conf=self.config.confidence,
                    imgsz=self.config.long_side,
                    device=requested_device,
                    verbose=False,
                )[0]
                person, foreground, confidence, foreground_type = (
                    _combined_foreground_masks(
                        result,
                        analysis_width,
                        analysis_height,
                        self.config.person_dilation,
                        set(self.config.foreground_class_ids),
                        self.config.foreground_min_area_ratio,
                        cv2,
                        np,
                    )
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
                    foreground_map=_reduce_map(
                        foreground,
                        map_width,
                        map_height,
                        cv2,
                    ),
                    foreground_type=foreground_type,
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
    person, _foreground, confidence, _kind = _combined_foreground_masks(
        result,
        width,
        height,
        dilation,
        {0},
        0.0,
        cv2,
        np,
    )
    return person, confidence


def _combined_foreground_masks(
    result,
    width: int,
    height: int,
    person_dilation: int,
    foreground_class_ids: set[int],
    min_area_ratio: float,
    cv2,
    np,
):
    """Split one YOLO result into person-only and useful-foreground unions."""
    person = np.zeros((height, width), dtype=np.uint8)
    foreground = np.zeros((height, width), dtype=np.uint8)
    confidence = 0.0
    has_person = False
    has_object = False
    if (
        result.masks is not None
        and len(result.masks.data)
        and result.boxes is not None
        and len(result.boxes.cls)
    ):
        masks = result.masks.data.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy()
        for mask, class_id, detection_confidence in zip(
            masks,
            classes,
            confidences,
        ):
            resized = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
            binary = (resized >= 0.5).astype(np.uint8) * 255
            area_ratio = float(np.count_nonzero(binary)) / max(1, binary.size)
            if class_id == 0:
                person = np.maximum(person, binary)
                confidence = max(confidence, float(detection_confidence))
                has_person = True
            if class_id in foreground_class_ids and (
                class_id == 0 or area_ratio >= min_area_ratio
            ):
                foreground = np.maximum(foreground, binary)
                if class_id == 0:
                    has_person = True
                else:
                    has_object = True
    if person_dilation > 0 and person.any():
        size = max(1, int(person_dilation))
        kernel = np.ones((size, size), dtype=np.uint8)
        person = cv2.dilate(person, kernel, iterations=1)
    foreground = np.maximum(foreground, person)
    foreground_type = (
        "mixed" if has_person and has_object
        else "person" if has_person
        else "object" if has_object
        else "none"
    )
    return person, foreground, confidence, foreground_type


def _reduce_map(frame_map, width: int, height: int, cv2):
    reduced = cv2.resize(frame_map, (width, height), interpolation=cv2.INTER_AREA)
    return reduced.astype("uint8", copy=False)
