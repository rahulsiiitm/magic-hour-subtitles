"""Deterministic caption placement from reduced visual-analysis maps."""

from __future__ import annotations

from math import hypot

import numpy as np

from .models import (
    CaptionPlan,
    FrameAnalysis,
    LayoutConfig,
    Placement,
    PlacementPlan,
    VideoInfo,
)


CANDIDATE_NAMES = (
    "top-left", "top-center", "top-right",
    "middle-left", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
)
PERSON_SAFE_THRESHOLD = 0.30
PERSON_TIE_TOLERANCE = 0.03
CENTER_CANDIDATES = {"top-center", "bottom-center"}


def generate_candidates(
    video: VideoInfo,
    caption_size: tuple[int, int],
    layout: LayoutConfig,
) -> list[Placement]:
    """Create the eight supported rectangles inside configured safe margins."""
    width = min(max(1, int(caption_size[0])), video.width - 2 * layout.margin_x)
    height = min(max(1, int(caption_size[1])), video.height - 2 * layout.margin_y)
    left = layout.margin_x
    center_x = (video.width - width) // 2
    right = video.width - layout.margin_x - width
    top = layout.margin_y
    middle = (video.height - height) // 2
    bottom = video.height - layout.margin_y - height
    coordinates = {
        "top-left": (left, top),
        "top-center": (center_x, top),
        "top-right": (right, top),
        "middle-left": (left, middle),
        "middle-right": (right, middle),
        "bottom-left": (left, bottom),
        "bottom-center": (center_x, bottom),
        "bottom-right": (right, bottom),
    }
    return [Placement(name, *coordinates[name], width, height) for name in CANDIDATE_NAMES]


def region_score(frame_map: np.ndarray, placement: Placement, video: VideoInfo) -> float:
    """Return normalized mean occupancy for a source-coordinate rectangle."""
    if frame_map.size == 0:
        return 0.0
    map_height, map_width = frame_map.shape[:2]
    x0 = max(0, min(map_width - 1, int(placement.x * map_width / video.width)))
    y0 = max(0, min(map_height - 1, int(placement.y * map_height / video.height)))
    x1 = max(x0 + 1, min(map_width, int(np.ceil(
        (placement.x + placement.width) * map_width / video.width
    ))))
    y1 = max(y0 + 1, min(map_height, int(np.ceil(
        (placement.y + placement.height) * map_height / video.height
    ))))
    region = frame_map[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0
    maximum = 255.0 if np.issubdtype(region.dtype, np.integer) else 1.0
    return float(np.clip(region.astype(np.float32).mean() / maximum, 0.0, 1.0))


def movement_penalty(
    current: Placement,
    previous: Placement | None,
    video: VideoInfo,
) -> float:
    if previous is None or current.name == previous.name:
        return 0.0
    current_center = (current.x + current.width / 2, current.y + current.height / 2)
    previous_center = (previous.x + previous.width / 2, previous.y + previous.height / 2)
    distance = hypot(
        current_center[0] - previous_center[0],
        current_center[1] - previous_center[1],
    )
    return min(1.0, distance / max(1.0, hypot(video.width, video.height)))


class PlacementPlanner:
    """Choose one stable measured position for every caption."""

    def __init__(
        self,
        video: VideoInfo,
        layout: LayoutConfig,
        *,
        hysteresis: float = 0.12,
        percentile: float = 75.0,
    ) -> None:
        self.video = video
        self.layout = layout
        self.hysteresis = hysteresis
        self.percentile = percentile

    def plan(
        self,
        captions: list[CaptionPlan],
        analyses: list[FrameAnalysis],
        caption_sizes: list[tuple[int, int]],
    ) -> list[PlacementPlan]:
        if len(captions) != len(caption_sizes):
            raise ValueError("Every caption must have one measured bounding box.")
        if not analyses:
            return []

        previous: Placement | None = None
        planned: list[PlacementPlan] = []
        for caption, size in zip(captions, caption_sizes):
            candidates = generate_candidates(self.video, size, self.layout)
            sampled = self._sample_caption_frames(caption, analyses)
            metrics = {
                candidate.name: self._candidate_metrics(candidate, sampled, previous)
                for candidate in candidates
            }
            by_name = {candidate.name: candidate for candidate in candidates}
            qualities = {
                name: 1.0 - values["penalty"]
                for name, values in metrics.items()
            }
            best_raw_name = max(qualities, key=qualities.get)
            best_name = self._person_safe_choice(metrics, qualities)
            safety_override = best_name != best_raw_name
            hysteresis_applied = False
            hysteresis_reason = "not applicable"
            previous_person_overlap = (
                metrics[previous.name]["person"]
                if previous is not None and previous.name in metrics
                else None
            )

            if previous_person_overlap is not None:
                if previous_person_overlap > PERSON_SAFE_THRESHOLD:
                    hysteresis_reason = (
                        "skipped "
                        f"(previous overlap {previous_person_overlap:.2f})"
                    )
                    safety_override = True
                else:
                    hysteresis_reason = "not needed"
            if previous_person_overlap is not None and (
                previous_person_overlap <= PERSON_SAFE_THRESHOLD
            ):
                previous_quality = qualities[previous.name]
                if (
                    best_name != previous.name
                    and qualities[best_name] <= previous_quality + self.hysteresis
                ):
                    best_name = previous.name
                    hysteresis_applied = True
                    hysteresis_reason = (
                        "applied "
                        f"(previous overlap {previous_person_overlap:.2f})"
                    )

            selected = by_name[best_name]
            planned.append(PlacementPlan(
                caption_plan=caption,
                placement=selected,
                scores={name: round(score, 4) for name, score in qualities.items()},
                person_overlaps={
                    name: round(values["person"], 4)
                    for name, values in metrics.items()
                },
                best_raw_candidate=best_raw_name,
                hysteresis_applied=hysteresis_applied,
                hysteresis_reason=hysteresis_reason,
                safety_override=safety_override,
                previous_person_overlap=previous_person_overlap,
            ))
            previous = selected
        return planned

    @staticmethod
    def _person_safe_choice(
        metrics: dict[str, dict[str, float]],
        qualities: dict[str, float],
    ) -> str:
        safe = [
            name
            for name, values in metrics.items()
            if values["person"] <= PERSON_SAFE_THRESHOLD
        ]
        if safe:
            return max(safe, key=lambda name: qualities[name])

        minimum_overlap = min(values["person"] for values in metrics.values())
        approximately_tied = [
            name
            for name, values in metrics.items()
            if values["person"] <= minimum_overlap + PERSON_TIE_TOLERANCE
        ]
        return max(
            approximately_tied,
            key=lambda name: (
                qualities[name],
                name in CENTER_CANDIDATES,
            ),
        )

    def _candidate_metrics(
        self,
        candidate: Placement,
        frames: list[FrameAnalysis],
        previous: Placement | None,
    ) -> dict[str, float]:
        person = self._robust([
            region_score(frame.person_map, candidate, self.video) for frame in frames
        ])
        clutter = self._robust([
            region_score(frame.clutter_map, candidate, self.video) for frame in frames
        ])
        motion = self._robust([
            region_score(frame.motion_map, candidate, self.video) for frame in frames
        ])
        edge = self._edge_margin_penalty(candidate)
        movement = movement_penalty(candidate, previous, self.video)
        penalty = (
            0.55 * person
            + 0.20 * clutter
            + 0.10 * motion
            + 0.05 * edge
            + 0.10 * movement
        )
        return {
            "person": person,
            "clutter": clutter,
            "motion": motion,
            "edge": edge,
            "movement": movement,
            "penalty": min(1.0, penalty),
        }

    def _sample_caption_frames(
        self,
        caption: CaptionPlan,
        analyses: list[FrameAnalysis],
    ) -> list[FrameAnalysis]:
        start = caption.caption.start
        duration = max(0.0, caption.caption.end - start)
        targets = [start + duration * fraction for fraction in (0.25, 0.50, 0.75)]
        selected: list[FrameAnalysis] = []
        seen: set[int] = set()
        for target in targets:
            nearest = min(analyses, key=lambda frame: abs(frame.timestamp - target))
            if nearest.frame_index not in seen:
                selected.append(nearest)
                seen.add(nearest.frame_index)
        return selected

    def _robust(self, values: list[float]) -> float:
        return float(np.percentile(values, self.percentile)) if values else 0.0

    def _edge_margin_penalty(self, placement: Placement) -> float:
        center_x = placement.x + placement.width / 2
        center_y = placement.y + placement.height / 2
        horizontal = abs(center_x - self.video.width / 2) / max(
            1.0,
            self.video.width / 2,
        )
        vertical = abs(center_y - self.video.height / 2) / max(
            1.0,
            self.video.height / 2,
        )
        return min(1.0, (horizontal + vertical) / 2)
