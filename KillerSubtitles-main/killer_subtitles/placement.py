"""Deterministic caption placement from reduced visual-analysis maps."""

from __future__ import annotations

from math import hypot

import cv2
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
HEAD_SAFE_THRESHOLD = 0.24
PERSON_TIE_TOLERANCE = 0.03
NO_PERSON_CONTEXT_THRESHOLD = 0.03
PLACEMENT_CHANGE_THRESHOLD = 0.08
NO_PERSON_HYSTERESIS = PLACEMENT_CHANGE_THRESHOLD
NO_PERSON_QUALITY_TIE_BAND = 0.025
FOREGROUND_SAFE_THRESHOLD = 0.35
CLUTTER_UNSAFE_THRESHOLD = 0.65
MOTION_UNSAFE_THRESHOLD = 0.60
NO_PERSON_BASELINE_PRIORITY = {
    "bottom-center": 0,
    "top-center": 1,
    "middle-left": 2,
    "middle-right": 2,
    "bottom-left": 3,
    "bottom-right": 3,
    "top-left": 4,
    "top-right": 4,
}


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
        hysteresis: float = PLACEMENT_CHANGE_THRESHOLD,
        percentile: float = 75.0,
        allow_occlusion: bool = False,
        occlusion_min_overlap: float = 0.08,
        occlusion_max_overlap: float = 0.30,
    ) -> None:
        self.video = video
        self.layout = layout
        self.hysteresis = hysteresis
        self.percentile = percentile
        # Retain the old keyword arguments for call-site compatibility only.
        # Occlusion is evaluated after placement and never selects a candidate.

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

        persistent_anchor: str | None = None
        planned: list[PlacementPlan] = []
        for caption, size in zip(captions, caption_sizes):
            candidates = generate_candidates(self.video, size, self.layout)
            by_name = {candidate.name: candidate for candidate in candidates}
            sampled = self._sample_caption_frames(caption, analyses)
            scene_cut = any(
                frame.scene_cut
                and caption.caption.start <= frame.timestamp <= caption.caption.end
                for frame in sampled
            )
            anchor_placement = (
                by_name[persistent_anchor]
                if persistent_anchor is not None and not scene_cut
                else None
            )
            metrics = {
                candidate.name: self._candidate_metrics(
                    candidate,
                    sampled,
                    anchor_placement,
                )
                for candidate in candidates
            }
            qualities = {
                name: 1.0 - values["penalty"]
                for name, values in metrics.items()
            }
            no_person_context = max(
                values["person"] for values in metrics.values()
            ) < NO_PERSON_CONTEXT_THRESHOLD
            portrait = self.video.height > self.video.width
            person_present = not no_person_context
            baseline_name = "bottom-center" if portrait else "scene-aware"
            bottom_center_safe, _ = self._anchor_safety(
                metrics["bottom-center"]
            )
            effective_hysteresis = self.hysteresis
            best_raw_name = max(qualities, key=qualities.get)
            normal_choice = self._safe_choice(metrics, qualities)
            safety_override = normal_choice != best_raw_name
            previous_anchor = persistent_anchor or ""
            baseline_tiebreak_applied = False
            movement_improvement = 0.0
            change_reason = "initial-anchor"
            anchor_retained = False

            if portrait and bottom_center_safe:
                persistent_anchor = "bottom-center"
                anchor_retained = previous_anchor == "bottom-center"
                movement_improvement = (
                    qualities["bottom-center"] - qualities[previous_anchor]
                    if previous_anchor in qualities
                    else 0.0
                )
                change_reason = (
                    "return-to-bottom"
                    if (
                        previous_anchor not in {"", "bottom-center"}
                    )
                    else "baseline-bottom"
                )
            elif persistent_anchor is None or scene_cut or (
                portrait and persistent_anchor == "bottom-center"
            ):
                persistent_anchor, baseline_tiebreak_applied = (
                    self._new_anchor_choice(
                        metrics,
                        qualities,
                        normal_choice,
                        apply_portrait_tiebreak=portrait,
                    )
                )
                if scene_cut:
                    change_reason = "scene-cut"
                elif portrait:
                    change_reason = self._bottom_obstruction_reason(
                        metrics["bottom-center"]
                    )
                else:
                    change_reason = "initial-anchor"
            else:
                anchor_safe, unsafe_reason = self._anchor_safety(
                    metrics[persistent_anchor]
                )
                movement_improvement = (
                    qualities[normal_choice] - qualities[persistent_anchor]
                )
                if not anchor_safe:
                    replacement, baseline_tiebreak_applied = (
                        self._new_anchor_choice(
                            metrics,
                            qualities,
                            normal_choice,
                            apply_portrait_tiebreak=portrait,
                        )
                    )
                    if replacement != persistent_anchor:
                        persistent_anchor = replacement
                        change_reason = self._obstruction_reason(
                            metrics[previous_anchor],
                            fallback=unsafe_reason,
                        )
                        safety_override = True
                    else:
                        anchor_retained = True
                        change_reason = self._retained_reason(
                            person_present,
                            portrait=portrait,
                        )
                elif self._should_change_anchor(movement_improvement):
                    persistent_anchor = normal_choice
                    change_reason = self._obstruction_reason(
                        metrics["bottom-center"],
                        fallback=self._quality_change_reason(
                            metrics[previous_anchor],
                            metrics[persistent_anchor],
                        ),
                    )
                else:
                    anchor_retained = True
                    change_reason = self._retained_reason(
                        person_present,
                        portrait=portrait,
                    )

            assert persistent_anchor is not None
            selected_name = persistent_anchor

            hysteresis_applied = (
                anchor_retained
                and normal_choice != persistent_anchor
                and not scene_cut
            )
            hysteresis_reason = (
                f"applied (improvement {movement_improvement:.3f} < "
                f"{effective_hysteresis:.3f})"
                if hysteresis_applied
                else "not needed"
            )
            previous_person_overlap = (
                metrics[previous_anchor]["person"]
                if previous_anchor in metrics
                else None
            )
            selected = by_name[selected_name]
            planned.append(PlacementPlan(
                caption_plan=caption,
                placement=selected,
                scores={name: round(score, 4) for name, score in qualities.items()},
                person_overlaps={
                    name: round(values["person"], 4)
                    for name, values in metrics.items()
                },
                head_overlaps={
                    name: round(values["head"], 4)
                    for name, values in metrics.items()
                },
                best_raw_candidate=best_raw_name,
                hysteresis_applied=hysteresis_applied,
                hysteresis_reason=hysteresis_reason,
                safety_override=safety_override,
                previous_person_overlap=previous_person_overlap,
                occlusion_opportunity=False,
                opportunity_score=0.0,
                no_person_context=no_person_context,
                effective_hysteresis=effective_hysteresis,
                baseline_tiebreak_applied=baseline_tiebreak_applied,
                foreground_overlaps={
                    name: round(values["foreground"], 4)
                    for name, values in metrics.items()
                },
                foreground_type=_foreground_type(sampled),
                persistent_anchor=persistent_anchor,
                previous_anchor=previous_anchor,
                anchor_retained=anchor_retained,
                movement_improvement=round(movement_improvement, 4),
                move_threshold=effective_hysteresis,
                change_reason=change_reason,
                temporary_placement=False,
                scene_cut=scene_cut,
                baseline_placement=baseline_name,
                person_present=person_present,
                bottom_center_safe=bottom_center_safe,
            ))
        return planned

    def _safe_choice(
        self,
        metrics: dict[str, dict[str, float]],
        qualities: dict[str, float],
    ) -> str:
        eligible = self._eligible_anchor_names(metrics)
        return max(eligible, key=lambda name: qualities[name])

    @staticmethod
    def _eligible_anchor_names(
        metrics: dict[str, dict[str, float]],
    ) -> list[str]:
        head_safe = [
            name
            for name, values in metrics.items()
            if values["head"] <= HEAD_SAFE_THRESHOLD
        ]
        if head_safe:
            eligible = head_safe
        else:
            minimum_head = min(
                values["head"] for values in metrics.values()
            )
            eligible = [
                name
                for name, values in metrics.items()
                if values["head"] <= minimum_head + PERSON_TIE_TOLERANCE
            ]

        person_safe = [
            name
            for name in eligible
            if metrics[name]["person"] <= PERSON_SAFE_THRESHOLD
        ]
        if person_safe:
            eligible = person_safe
        else:
            minimum_person = min(metrics[name]["person"] for name in eligible)
            eligible = [
                name for name in eligible
                if metrics[name]["person"]
                <= minimum_person + PERSON_TIE_TOLERANCE
            ]
        foreground_safe = [
            name for name in eligible
            if metrics[name]["object_foreground"] <= FOREGROUND_SAFE_THRESHOLD
        ]
        if foreground_safe:
            eligible = foreground_safe
        clutter_safe = [
            name for name in eligible
            if metrics[name]["clutter"] <= CLUTTER_UNSAFE_THRESHOLD
        ]
        if clutter_safe:
            eligible = clutter_safe
        motion_safe = [
            name for name in eligible
            if metrics[name]["motion"] <= MOTION_UNSAFE_THRESHOLD
        ]
        return motion_safe or eligible

    def _new_anchor_choice(
        self,
        metrics: dict[str, dict[str, float]],
        qualities: dict[str, float],
        normal_choice: str,
        *,
        apply_portrait_tiebreak: bool,
    ) -> tuple[str, bool]:
        if not apply_portrait_tiebreak:
            return normal_choice, False
        return self._baseline_choice(
            qualities,
            normal_choice,
            candidate_names=self._eligible_anchor_names(metrics),
        )

    @staticmethod
    def _anchor_safety(metrics: dict[str, float]) -> tuple[bool, str]:
        if metrics["head"] > HEAD_SAFE_THRESHOLD:
            return False, "person-safety"
        if metrics["person"] > PERSON_SAFE_THRESHOLD:
            return False, "person-safety"
        if metrics["object_foreground"] > FOREGROUND_SAFE_THRESHOLD:
            return False, "foreground-safety"
        if metrics["clutter"] > CLUTTER_UNSAFE_THRESHOLD:
            return False, "clutter-improvement"
        if metrics["motion"] > MOTION_UNSAFE_THRESHOLD:
            return False, "motion-improvement"
        return True, "retained-anchor"

    @staticmethod
    def _obstruction_reason(
        metrics: dict[str, float],
        *,
        fallback: str,
    ) -> str:
        if (
            metrics["person"] > PERSON_SAFE_THRESHOLD
            or metrics["head"] > HEAD_SAFE_THRESHOLD
        ):
            return "person-obstruction"
        if metrics["object_foreground"] > FOREGROUND_SAFE_THRESHOLD:
            return "foreground-obstruction"
        return fallback

    @classmethod
    def _bottom_obstruction_reason(cls, metrics: dict[str, float]) -> str:
        if metrics["clutter"] > CLUTTER_UNSAFE_THRESHOLD:
            fallback = "clutter-improvement"
        elif metrics["motion"] > MOTION_UNSAFE_THRESHOLD:
            fallback = "motion-improvement"
        else:
            fallback = "foreground-obstruction"
        return cls._obstruction_reason(
            metrics,
            fallback=fallback,
        )

    @staticmethod
    def _retained_reason(person_present: bool, *, portrait: bool) -> str:
        if not portrait:
            return "retained-anchor"
        return (
            "person-aware-retained"
            if person_present
            else "foreground-obstruction"
        )

    def _should_change_anchor(self, improvement: float) -> bool:
        return improvement >= self.hysteresis

    @staticmethod
    def _quality_change_reason(
        current: dict[str, float],
        replacement: dict[str, float],
    ) -> str:
        if current["clutter"] - replacement["clutter"] >= 0.20:
            return "clutter-improvement"
        if current["motion"] - replacement["motion"] >= 0.20:
            return "motion-improvement"
        return "quality-improvement"

    @staticmethod
    def _baseline_choice(
        qualities: dict[str, float],
        normal_choice: str,
        *,
        candidate_names: list[str] | None = None,
    ) -> tuple[str, bool]:
        allowed = candidate_names or list(qualities)
        best_quality = max(qualities[name] for name in allowed)
        comparable = [
            name
            for name in allowed
            if best_quality - qualities[name] <= NO_PERSON_QUALITY_TIE_BAND
        ]
        selected = min(
            comparable,
            key=lambda name: (
                NO_PERSON_BASELINE_PRIORITY[name],
                -qualities[name],
                name,
            ),
        )
        return selected, selected != normal_choice

    def _candidate_metrics(
        self,
        candidate: Placement,
        frames: list[FrameAnalysis],
        previous: Placement | None,
    ) -> dict[str, float]:
        person = self._robust([
            region_score(frame.person_map, candidate, self.video) for frame in frames
        ])
        foreground = self._robust([
            region_score(_foreground_map(frame), candidate, self.video)
            for frame in frames
        ])
        object_foreground = max(0.0, foreground - person)
        head = self._robust([
            region_score(_upper_person_map(frame.person_map), candidate, self.video)
            for frame in frames
        ])
        clutter = self._robust([
            region_score(frame.clutter_map, candidate, self.video) for frame in frames
        ])
        motion = self._robust([
            region_score(frame.motion_map, candidate, self.video) for frame in frames
        ])
        edge = self._edge_margin_penalty(candidate)
        safe_zone = self._portrait_safe_zone_penalty(candidate)
        movement = movement_penalty(candidate, previous, self.video)
        non_person_penalty = (
            0.12 * object_foreground
            + 0.20 * clutter
            + 0.10 * motion
            + 0.05 * edge
            + 0.04 * safe_zone
            + 0.14 * movement
        )
        penalty = (
            0.52 * person
            + 0.12 * head
            + non_person_penalty
        )
        return {
            "person": person,
            "foreground": foreground,
            "object_foreground": object_foreground,
            "head": head,
            "clutter": clutter,
            "motion": motion,
            "edge": edge,
            "safe_zone": safe_zone,
            "movement": movement,
            "non_person_penalty": non_person_penalty,
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

    def _portrait_safe_zone_penalty(self, placement: Placement) -> float:
        if self.video.height <= self.video.width:
            return 0.0
        top_gap = placement.y / max(1.0, self.video.height)
        bottom_gap = (
            self.video.height - placement.y - placement.height
        ) / max(1.0, self.video.height)
        comfortable = 0.08
        nearest = min(top_gap, bottom_gap)
        return max(0.0, min(1.0, (comfortable - nearest) / comfortable))


def _upper_person_map(person_map: np.ndarray) -> np.ndarray:
    """Approximate each visible person's protected upper region."""
    if person_map.size == 0:
        return person_map
    maximum = 255.0 if np.issubdtype(person_map.dtype, np.integer) else 1.0
    binary = np.where(person_map.astype(np.float32) >= maximum * 0.5, 1, 0).astype(
        np.uint8
    )
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    result = np.zeros_like(person_map)
    for label in range(1, component_count):
        x, y, width, height, area = stats[label]
        if area <= 0:
            continue
        protected_height = max(1, int(round(height * 0.40)))
        protected = labels[y:y + protected_height, x:x + width] == label
        horizontal = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        center_weight = 0.65 + 0.35 * (1.0 - np.abs(horizontal))
        source = person_map[y:y + protected_height, x:x + width].astype(
            np.float32
        )
        weighted = np.where(
            protected,
            source * center_weight[None, :],
            0.0,
        )
        target = result[y:y + protected_height, x:x + width]
        np.maximum(
            target,
            np.clip(weighted, 0.0, maximum).astype(person_map.dtype),
            out=target,
        )
    return result


def _foreground_map(frame: FrameAnalysis) -> np.ndarray:
    foreground = frame.foreground_map
    if isinstance(foreground, np.ndarray) and foreground.size:
        return foreground
    return frame.person_map


def _foreground_type(frames: list[FrameAnalysis]) -> str:
    types = {
        frame.foreground_type
        for frame in frames
        if frame.foreground_type in {"person", "object", "mixed"}
    }
    if "mixed" in types or ({"person", "object"} <= types):
        return "mixed"
    if "person" in types:
        return "person"
    if "object" in types:
        return "object"
    if any(np.any(frame.person_map) for frame in frames):
        return "person"
    return "none"
