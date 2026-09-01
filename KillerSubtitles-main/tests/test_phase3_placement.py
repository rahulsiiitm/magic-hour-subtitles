from __future__ import annotations

import unittest

import numpy as np

from killer_subtitles.models import (
    Caption,
    CaptionPlan,
    CaptionStyle,
    FrameAnalysis,
    LayoutConfig,
    Placement,
    Tone,
    VideoInfo,
    Word,
)
from killer_subtitles.placement import (
    CANDIDATE_NAMES,
    PlacementPlanner,
    generate_candidates,
    movement_penalty,
    region_score,
)


VIDEO = VideoInfo(1000, 1000, 30.0, 4.0)
LAYOUT = LayoutConfig(margin_x=50, margin_y=40)
STYLE = CaptionStyle("#fff", "#ff0", "#ff0")


def caption_plan(text: str, start: float, end: float) -> CaptionPlan:
    return CaptionPlan(
        caption=Caption([Word(text, start, end)]),
        tone=Tone.NEUTRAL,
        keyword_indices=(),
        style=STYLE,
    )


def frame(
    timestamp: float,
    index: int,
    *,
    person: np.ndarray | None = None,
    clutter: np.ndarray | None = None,
    motion: np.ndarray | None = None,
) -> FrameAnalysis:
    empty = np.zeros((100, 100), dtype=np.uint8)
    return FrameAnalysis(
        timestamp=timestamp,
        frame_index=index,
        map_width=100,
        map_height=100,
        person_map=empty if person is None else person,
        clutter_map=empty if clutter is None else clutter,
        motion_map=empty if motion is None else motion,
    )


def fill_region(array: np.ndarray, placement: Placement, value: int) -> None:
    x0 = int(placement.x / 10)
    x1 = int(np.ceil((placement.x + placement.width) / 10))
    y0 = int(placement.y / 10)
    y1 = int(np.ceil((placement.y + placement.height) / 10))
    array[y0:y1, x0:x1] = value


class CandidateTests(unittest.TestCase):
    def test_generates_all_supported_candidates_without_middle_center(self):
        candidates = generate_candidates(VIDEO, (240, 120), LAYOUT)
        self.assertEqual(tuple(candidate.name for candidate in candidates), CANDIDATE_NAMES)
        self.assertNotIn("middle-center", CANDIDATE_NAMES)

    def test_candidates_respect_safe_margins(self):
        for candidate in generate_candidates(VIDEO, (240, 120), LAYOUT):
            with self.subTest(candidate=candidate.name):
                self.assertGreaterEqual(candidate.x, LAYOUT.margin_x)
                self.assertGreaterEqual(candidate.y, LAYOUT.margin_y)
                self.assertLessEqual(
                    candidate.x + candidate.width,
                    VIDEO.width - LAYOUT.margin_x,
                )
                self.assertLessEqual(
                    candidate.y + candidate.height,
                    VIDEO.height - LAYOUT.margin_y,
                )


class ScoringTests(unittest.TestCase):
    def test_behind_subject_can_choose_conservative_overlap_opportunity(self):
        planner = PlacementPlanner(
            VIDEO,
            LAYOUT,
            allow_occlusion=True,
            occlusion_min_overlap=0.10,
            occlusion_max_overlap=0.35,
        )
        metrics = {
            "top-left": {"person": 0.0, "head": 0.0, "non_person_penalty": 0.28},
            "middle-right": {"person": 0.18, "head": 0.02, "non_person_penalty": 0.12},
        }
        qualities = {"top-left": 0.80, "middle-right": 0.73}

        selected, score = planner._occlusion_opportunity(
            caption_plan("useful caption moment", 0.0, 1.0),
            metrics,
            qualities,
            "top-left",
        )

        self.assertEqual(selected, "middle-right")
        self.assertGreater(score, 0.58)

    def test_portrait_head_concentration_penalizes_upper_caption_region(self):
        video = VideoInfo(600, 1000, 30.0, 2.0)
        layout = LayoutConfig(margin_x=60, margin_y=60)
        candidates = {
            item.name: item
            for item in generate_candidates(video, (180, 90), layout)
        }
        person = np.zeros((100, 60), dtype=np.uint8)
        person[:45, 15:45] = 255
        planner = PlacementPlanner(video, layout, hysteresis=0.0)
        sampled = [FrameAnalysis(
            timestamp=0.5,
            frame_index=1,
            map_width=60,
            map_height=100,
            person_map=person,
            clutter_map=np.zeros_like(person),
            motion_map=np.zeros_like(person),
        )]

        top = planner._candidate_metrics(candidates["top-center"], sampled, None)
        bottom = planner._candidate_metrics(candidates["bottom-center"], sampled, None)

        self.assertGreater(top["head"], bottom["head"])
        self.assertGreater(top["penalty"], bottom["penalty"])

    def test_person_overlap_uses_candidate_rectangle(self):
        placement = Placement("top-left", 0, 0, 500, 500)
        person = np.zeros((100, 100), dtype=np.uint8)
        person[:50, :50] = 255
        self.assertAlmostEqual(region_score(person, placement, VIDEO), 1.0)

    def test_clutter_score_is_normalized(self):
        placement = Placement("top-left", 0, 0, 500, 500)
        clutter = np.zeros((100, 100), dtype=np.uint8)
        clutter[:50, :50] = 128
        self.assertAlmostEqual(
            region_score(clutter, placement, VIDEO),
            128 / 255,
            places=3,
        )

    def test_movement_penalty_prefers_previous_position(self):
        left, right = generate_candidates(VIDEO, (200, 100), LAYOUT)[0:3:2]
        self.assertEqual(movement_penalty(left, left, VIDEO), 0.0)
        self.assertGreater(movement_penalty(right, left, VIDEO), 0.0)

    def test_hysteresis_keeps_reasonably_good_previous_position(self):
        candidates = {
            item.name: item
            for item in generate_candidates(VIDEO, (200, 100), LAYOUT)
        }
        first_clutter = np.full((100, 100), 100, dtype=np.uint8)
        fill_region(first_clutter, candidates["top-left"], 0)
        second_clutter = np.full((100, 100), 255, dtype=np.uint8)
        fill_region(second_clutter, candidates["top-left"], 80)
        fill_region(second_clutter, candidates["top-right"], 0)
        planner = PlacementPlanner(VIDEO, LAYOUT, hysteresis=0.12)

        plans = planner.plan(
            [caption_plan("one", 0.0, 1.0), caption_plan("two", 2.0, 3.0)],
            [frame(0.5, 1, clutter=first_clutter), frame(2.5, 2, clutter=second_clutter)],
            [(200, 100), (200, 100)],
        )

        self.assertEqual(plans[0].placement.name, "top-left")
        self.assertEqual(plans[1].placement.name, "top-left")
        self.assertTrue(plans[1].hysteresis_applied)
        self.assertLessEqual(plans[1].previous_person_overlap, 0.30)

    def test_hysteresis_cannot_retain_high_person_overlap(self):
        candidates = {
            item.name: item
            for item in generate_candidates(VIDEO, (200, 100), LAYOUT)
        }
        first_clutter = np.full((100, 100), 100, dtype=np.uint8)
        fill_region(first_clutter, candidates["top-left"], 0)
        second_clutter = np.full((100, 100), 255, dtype=np.uint8)
        fill_region(second_clutter, candidates["top-right"], 0)
        second_person = np.zeros((100, 100), dtype=np.uint8)
        fill_region(second_person, candidates["top-left"], 255)
        planner = PlacementPlanner(VIDEO, LAYOUT, hysteresis=0.12)

        plans = planner.plan(
            [caption_plan("one", 0.0, 1.0), caption_plan("two", 2.0, 3.0)],
            [
                frame(0.5, 1, clutter=first_clutter),
                frame(2.5, 2, person=second_person, clutter=second_clutter),
            ],
            [(200, 100), (200, 100)],
        )

        self.assertEqual(plans[0].placement.name, "top-left")
        self.assertEqual(plans[1].placement.name, "top-right")
        self.assertFalse(plans[1].hysteresis_applied)
        self.assertGreater(plans[1].previous_person_overlap, 0.30)
        self.assertTrue(plans[1].safety_override)

    def test_no_person_frames_use_clutter(self):
        candidates = {
            item.name: item
            for item in generate_candidates(VIDEO, (200, 100), LAYOUT)
        }
        clutter = np.full((100, 100), 255, dtype=np.uint8)
        fill_region(clutter, candidates["bottom-right"], 0)
        planner = PlacementPlanner(VIDEO, LAYOUT, hysteresis=0.0)

        result = planner.plan(
            [caption_plan("clear", 0.0, 1.0)],
            [frame(0.5, 1, clutter=clutter)],
            [(200, 100)],
        )[0]

        self.assertEqual(result.placement.name, "bottom-right")
        self.assertTrue(all(value == 0.0 for value in result.person_overlaps.values()))

    def test_all_poor_fallback_checks_all_regions_for_minimum_person_overlap(self):
        candidates = {
            item.name: item
            for item in generate_candidates(VIDEO, (200, 100), LAYOUT)
        }
        person = np.full((100, 100), 220, dtype=np.uint8)
        fill_region(person, candidates["bottom-right"], 80)
        planner = PlacementPlanner(VIDEO, LAYOUT, hysteresis=0.0)

        result = planner.plan(
            [caption_plan("blocked", 0.0, 1.0)],
            [frame(0.5, 1, person=person)],
            [(200, 100)],
        )[0]

        self.assertEqual(result.placement.name, "bottom-right")
        self.assertAlmostEqual(
            result.person_overlaps["bottom-right"],
            min(result.person_overlaps.values()),
        )

    def test_movement_penalty_cannot_outweigh_major_person_difference(self):
        candidates = {
            item.name: item
            for item in generate_candidates(VIDEO, (200, 100), LAYOUT)
        }
        first_clutter = np.full((100, 100), 100, dtype=np.uint8)
        fill_region(first_clutter, candidates["top-left"], 0)
        second_person = np.full((100, 100), 210, dtype=np.uint8)
        fill_region(second_person, candidates["top-left"], 250)
        fill_region(second_person, candidates["bottom-right"], 100)
        planner = PlacementPlanner(VIDEO, LAYOUT, hysteresis=0.12)

        plans = planner.plan(
            [caption_plan("one", 0.0, 1.0), caption_plan("two", 2.0, 3.0)],
            [
                frame(0.5, 1, clutter=first_clutter),
                frame(2.5, 2, person=second_person),
            ],
            [(200, 100), (200, 100)],
        )

        self.assertEqual(plans[0].placement.name, "top-left")
        self.assertEqual(plans[1].placement.name, "bottom-right")
        self.assertLess(
            plans[1].person_overlaps["bottom-right"],
            plans[1].person_overlaps["top-left"] - 0.30,
        )


if __name__ == "__main__":
    unittest.main()
