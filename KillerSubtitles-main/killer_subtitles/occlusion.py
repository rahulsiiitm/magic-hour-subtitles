"""Behind-subject decisions and dense temporal person-mask utilities."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import replace
from math import ceil

import cv2
import numpy as np

from .models import (
    CaptionPlan,
    FrameAnalysis,
    OcclusionDecision,
    PlacementPlan,
    StyleConfig,
    SubtitleState,
    VideoInfo,
)
from .placement import region_score
from .renderer import SubtitleRenderer


MIN_TEXT_INTERSECTION = 0.02
SWEET_SPOT_MIN = 0.08
SWEET_SPOT_MAX = 0.28
MAX_DEMO_OPPORTUNITIES = 6


def decide_occlusion(
    caption_plan: CaptionPlan,
    *,
    has_mask: bool,
    person_overlap: float,
    caption_occlusion: float,
    min_overlap: float,
    max_occlusion: float,
    meaningful_word_count: int | None = None,
    placement_opportunity_score: float = 0.0,
) -> OcclusionDecision:
    """Apply conservative readability gates to one caption."""
    meaningful_count = (
        _meaningful_word_count(caption_plan)
        if meaningful_word_count is None
        else meaningful_word_count
    )
    opportunity_score = occlusion_opportunity_score(
        person_overlap,
        caption_occlusion,
        meaningful_count,
        placement_opportunity_score,
    )
    rejection_code = ""
    if not has_mask:
        enabled = False
        reason = "person mask unavailable"
        rejection_code = "mask_unavailable"
    elif person_overlap < min_overlap:
        enabled = False
        reason = "person overlap below activation threshold"
        rejection_code = "low_overlap"
    elif caption_occlusion < MIN_TEXT_INTERSECTION:
        enabled = False
        reason = "person does not meaningfully intersect text"
        rejection_code = "low_overlap"
    elif caption_occlusion > max_occlusion:
        enabled = False
        reason = f"would hide {caption_occlusion:.0%} of caption"
        rejection_code = "high_occlusion"
    elif meaningful_count < 2 and caption_occlusion < 0.14:
        enabled = False
        reason = "caption too short for a visible controlled overlap"
        rejection_code = "too_short"
    else:
        enabled = True
        reason = (
            "sweet-spot partial person overlap"
            if SWEET_SPOT_MIN <= caption_occlusion <= SWEET_SPOT_MAX
            else "readable partial person overlap"
        )
    return OcclusionDecision(
        caption_plan=caption_plan,
        enabled=enabled,
        person_overlap=float(person_overlap),
        caption_occlusion=float(caption_occlusion),
        reason=reason,
        opportunity_score=opportunity_score,
        rejection_code=rejection_code,
    )


def occlusion_opportunity_score(
    person_overlap: float,
    text_occlusion: float,
    meaningful_word_count: int,
    placement_score: float = 0.0,
) -> float:
    """Score visible, readable partial overlaps without relaxing hard gates."""
    if SWEET_SPOT_MIN <= text_occlusion <= SWEET_SPOT_MAX:
        text_score = 1.0
    elif MIN_TEXT_INTERSECTION <= text_occlusion < SWEET_SPOT_MIN:
        text_score = (
            (text_occlusion - MIN_TEXT_INTERSECTION)
            / (SWEET_SPOT_MIN - MIN_TEXT_INTERSECTION)
        )
    elif SWEET_SPOT_MAX < text_occlusion <= 0.45:
        text_score = 1.0 - (
            (text_occlusion - SWEET_SPOT_MAX) / (0.45 - SWEET_SPOT_MAX)
        )
    else:
        text_score = 0.0

    if 0.10 <= person_overlap <= 0.30:
        person_score = 1.0
    elif person_overlap < 0.10:
        person_score = max(0.0, person_overlap / 0.10)
    else:
        person_score = max(0.0, 1.0 - (person_overlap - 0.30) / 0.30)
    length_score = 1.0 if meaningful_word_count >= 3 else (
        0.85 if meaningful_word_count == 2 else 0.35
    )
    return float(np.clip(
        0.65 * text_score
        + 0.15 * person_score
        + 0.10 * length_score
        + 0.10 * np.clip(placement_score, 0.0, 1.0),
        0.0,
        1.0,
    ))


class TemporalMaskProvider:
    """Interpolate sampled reduced person masks and clean them at output size."""

    def __init__(
        self,
        analyses: list[FrameAnalysis],
        *,
        output_width: int,
        output_height: int,
        dilate: int = 2,
        blur: int = 5,
    ) -> None:
        self.analyses = sorted(analyses, key=lambda frame: frame.timestamp)
        self.timestamps = [frame.timestamp for frame in self.analyses]
        self.output_width = output_width
        self.output_height = output_height
        self.dilate = max(0, int(dilate))
        self.blur = max(0, int(blur))

    def reduced_mask_at(self, timestamp: float) -> np.ndarray | None:
        if not self.analyses:
            return None
        if timestamp <= self.timestamps[0]:
            return _valid_mask(self.analyses[0].person_map)
        if timestamp >= self.timestamps[-1]:
            return _valid_mask(self.analyses[-1].person_map)

        right_index = bisect_right(self.timestamps, timestamp)
        left = self.analyses[right_index - 1]
        right = self.analyses[right_index]
        left_mask = _valid_mask(left.person_map)
        right_mask = _valid_mask(right.person_map)
        if left_mask is None and right_mask is None:
            return None
        if left_mask is None:
            return right_mask
        if right_mask is None:
            return left_mask
        if right_mask.shape != left_mask.shape:
            right_mask = cv2.resize(
                right_mask,
                (left_mask.shape[1], left_mask.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        if right.scene_cut or right.timestamp <= left.timestamp:
            return left_mask if timestamp - left.timestamp <= right.timestamp - timestamp else right_mask

        weight = (timestamp - left.timestamp) / (right.timestamp - left.timestamp)
        blended = (
            left_mask.astype(np.float32) * (1.0 - weight)
            + right_mask.astype(np.float32) * weight
        )
        return np.rint(np.clip(blended, 0, 255)).astype(np.uint8)

    def mask_at(self, timestamp: float) -> np.ndarray | None:
        reduced = self.reduced_mask_at(timestamp)
        if reduced is None:
            return None
        return clean_person_mask(
            reduced,
            self.output_width,
            self.output_height,
            dilate=self.dilate,
            blur=self.blur,
        )


class OcclusionPlanner:
    """Estimate actual rendered-text occlusion for every placed caption."""

    def __init__(
        self,
        video: VideoInfo,
        style: StyleConfig,
        analyses: list[FrameAnalysis],
        *,
        min_overlap: float = 0.10,
        max_occlusion: float = 0.45,
    ) -> None:
        self.video = video
        self.renderer = SubtitleRenderer(video, style)
        self.analyses = analyses
        self.min_overlap = min_overlap
        self.max_occlusion = max_occlusion
        self.provider = TemporalMaskProvider(
            analyses,
            output_width=video.width,
            output_height=video.height,
            dilate=0,
            blur=0,
        )

    def plan(
        self,
        placements: list[PlacementPlan],
        representative_states: list[SubtitleState],
    ) -> list[OcclusionDecision]:
        if len(placements) != len(representative_states):
            raise ValueError("Every placement requires one representative subtitle state.")
        decisions: list[OcclusionDecision] = []
        for placement_plan, state in zip(placements, representative_states):
            decisions.append(self._plan_one(placement_plan, state))
        return self._prefer_meaningful_opportunities(decisions)

    @staticmethod
    def _prefer_meaningful_opportunities(
        decisions: list[OcclusionDecision],
    ) -> list[OcclusionDecision]:
        enabled_indices = [
            index for index, decision in enumerate(decisions) if decision.enabled
        ]
        limit = min(
            MAX_DEMO_OPPORTUNITIES,
            max(1, int(ceil(len(decisions) * 0.30))),
        )
        ranked = sorted(
            enabled_indices,
            key=lambda index: (
                _meaningful_word_count(decisions[index].caption_plan) >= 2,
                decisions[index].opportunity_score,
            ),
            reverse=True,
        )
        selected = set(ranked[:limit])
        return [
            decision
            if not decision.enabled or index in selected
            else replace(
                decision,
                enabled=False,
                reason="stronger readable opportunity preferred",
                rejection_code="lower_ranked",
            )
            for index, decision in enumerate(decisions)
        ]

    def _plan_one(
        self,
        placement_plan: PlacementPlan,
        state: SubtitleState,
    ) -> OcclusionDecision:
        plan = placement_plan.caption_plan
        duration = max(0.0, plan.caption.end - plan.caption.start)
        timestamps = [
            plan.caption.start + duration * fraction
            for fraction in (0.25, 0.50, 0.75)
        ]
        alpha = np.asarray(self.renderer.render(state), dtype=np.uint8)[:, :, 3]
        overlaps: list[float] = []
        text_occlusions: list[float] = []
        has_mask = False

        for timestamp in timestamps:
            person = self.provider.reduced_mask_at(timestamp)
            if person is None or not np.any(person):
                continue
            has_mask = True
            overlaps.append(region_score(person, placement_plan.placement, self.video))
            text_alpha = cv2.resize(
                alpha,
                (person.shape[1], person.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            text_pixels = text_alpha > 8
            text_count = int(np.count_nonzero(text_pixels))
            if text_count:
                hidden = np.count_nonzero(text_pixels & (person >= 128))
                text_occlusions.append(hidden / text_count)

        person_overlap = _robust(overlaps)
        caption_occlusion = _robust(text_occlusions)
        return decide_occlusion(
            plan,
            has_mask=has_mask,
            person_overlap=person_overlap,
            caption_occlusion=caption_occlusion,
            min_overlap=self.min_overlap,
            max_occlusion=self.max_occlusion,
            meaningful_word_count=_meaningful_word_count(plan),
            placement_opportunity_score=placement_plan.opportunity_score,
        )


def clean_person_mask(
    mask: np.ndarray,
    width: int,
    height: int,
    *,
    dilate: int = 2,
    blur: int = 5,
) -> np.ndarray:
    """Resize, close, lightly dilate, and feather a person mask."""
    valid = _valid_mask(mask)
    if valid is None:
        raise ValueError("Person mask must be a non-empty two-dimensional array.")
    resized = cv2.resize(valid, (width, height), interpolation=cv2.INTER_LINEAR)
    binary = np.where(resized >= 128, 255, 0).astype(np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    if dilate > 0:
        size = 2 * int(dilate) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        cleaned = cv2.dilate(cleaned, kernel, iterations=1)
    if blur > 0:
        size = int(blur)
        if size % 2 == 0:
            size += 1
        cleaned = cv2.GaussianBlur(cleaned, (size, size), 0)
    return cleaned.astype(np.uint8, copy=False)


def iter_dense_masks(
    provider: TemporalMaskProvider,
    decisions: list[OcclusionDecision],
    video: VideoInfo,
    *,
    max_fps: float = 30.0,
    stabilization: float = 0.15,
) -> tuple[float, int, Iterator[np.ndarray]]:
    """Return a dense mask iterator aligned to source time, capped at 30 FPS."""
    fps = min(max(1.0, video.fps), max_fps)
    frame_count = max(1, int(ceil(video.duration * fps)))
    enabled = [decision for decision in decisions if decision.enabled]

    def generate() -> Iterator[np.ndarray]:
        previous: np.ndarray | None = None
        for frame_index in range(frame_count):
            timestamp = frame_index / fps
            active = any(
                decision.start <= timestamp <= decision.end
                for decision in enabled
            )
            current = provider.mask_at(timestamp) if active else None
            if current is None:
                previous = None
                yield np.zeros(
                    (provider.output_height, provider.output_width),
                    dtype=np.uint8,
                )
                continue
            if previous is not None and stabilization > 0:
                current = cv2.addWeighted(
                    current,
                    1.0 - stabilization,
                    previous,
                    stabilization,
                    0,
                )
            previous = current
            yield current

    return fps, frame_count, generate()


def _valid_mask(mask) -> np.ndarray | None:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.size == 0:
        return None
    return mask.astype(np.uint8, copy=False)


def _robust(values: list[float]) -> float:
    return float(np.percentile(values, 75.0)) if values else 0.0


def _meaningful_word_count(plan: CaptionPlan) -> int:
    return sum(
        1
        for word in plan.caption.words
        if any(character.isalnum() for character in word.text)
    )
