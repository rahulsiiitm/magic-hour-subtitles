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
    NO_PERSON_HYSTERESIS,
    PLACEMENT_CHANGE_THRESHOLD,
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
    foreground: np.ndarray | None = None,
    foreground_type: str = "none",
    scene_cut: bool = False,
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
        foreground_map=foreground,
        foreground_type=foreground_type,
        scene_cut=scene_cut,
    )


def fill_region(array: np.ndarray, placement: Placement, value: int) -> None:
    x0 = int(placement.x / 10)
    x1 = int(np.ceil((placement.x + placement.width) / 10))
    y0 = int(placement.y / 10)
    y1 = int(np.ceil((placement.y + placement.height) / 10))
    array[y0:y1, x0:x1] = value


def fill_scaled_region(
    array: np.ndarray,
    placement: Placement,
    video: VideoInfo,
    value: int,
) -> None:
    height, width = array.shape
    x0 = int(placement.x * width / video.width)
    x1 = int(np.ceil((placement.x + placement.width) * width / video.width))
    y0 = int(placement.y * height / video.height)
    y1 = int(np.ceil((placement.y + placement.height) * height / video.height))
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
    def test_occlusion_configuration_never_changes_safe_anchor(self):
        video = VideoInfo(600, 1000, 30.0, 2.0)
        layout = LayoutConfig(margin_x=90, margin_y=60)
        candidates = {
            item.name: item
            for item in generate_candidates(video, (180, 90), layout)
        }
        foreground = np.zeros((100, 60), dtype=np.uint8)
        fill_scaled_region(
            foreground,
            candidates["middle-right"],
            video,
            46,
        )
        planner = PlacementPlanner(
            video,
            layout,
            allow_occlusion=True,
            occlusion_min_overlap=0.10,
            occlusion_max_overlap=0.35,
        )

        result = planner.plan(
            [caption_plan("useful caption moment", 0.0, 1.0)],
            [frame(
                0.5,
                0,
                foreground=foreground,
                foreground_type="object",
            )],
            [(180, 90)],
        )[0]

        self.assertEqual(result.placement.name, "bottom-center")
        self.assertFalse(result.temporary_placement)
        self.assertFalse(result.occlusion_opportunity)
        self.assertEqual(result.change_reason, "baseline-bottom")

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

    def test_strong_head_overlap_is_ineligible_even_with_best_raw_score(self):
        video = VideoInfo(600, 1000, 30.0, 2.0)
        layout = LayoutConfig(margin_x=60, margin_y=60)
        candidates = {
            item.name: item
            for item in generate_candidates(video, (180, 90), layout)
        }
        person = np.zeros((100, 60), dtype=np.uint8)
        fill_scaled_region(person, candidates["top-center"], video, 255)
        clutter = np.full_like(person, 180)
        fill_scaled_region(clutter, candidates["top-center"], video, 0)

        result = PlacementPlanner(video, layout, hysteresis=0.0).plan(
            [caption_plan("protect the face", 0.0, 1.0)],
            [frame(0.5, 0, person=person, clutter=clutter)],
            [(180, 90)],
        )[0]

        self.assertGreater(
            result.head_overlaps["top-center"],
            0.24,
        )
        self.assertNotEqual(result.placement.name, "top-center")
        self.assertLessEqual(
            result.head_overlaps[result.placement.name],
            0.24,
        )

    def test_torso_overlap_can_be_safe_when_protected_head_is_clear(self):
        video = VideoInfo(600, 1000, 30.0, 2.0)
        layout = LayoutConfig(margin_x=90, margin_y=60)
        person = np.zeros((100, 60), dtype=np.uint8)
        person[15:95, 28:32] = 255

        result = PlacementPlanner(video, layout).plan(
            [caption_plan("lower body remains readable", 0.0, 1.0)],
            [frame(0.5, 0, person=person)],
            [(180, 90)],
        )[0]

        self.assertEqual(result.head_overlaps["bottom-center"], 0.0)
        self.assertLessEqual(result.person_overlaps["bottom-center"], 0.30)
        self.assertEqual(result.placement.name, "bottom-center")

    def test_two_people_upper_regions_do_not_attract_caption(self):
        video = VideoInfo(600, 1000, 30.0, 2.0)
        layout = LayoutConfig(margin_x=60, margin_y=60)
        person = np.zeros((100, 60), dtype=np.uint8)
        person[8:70, 4:23] = 255
        person[8:70, 37:56] = 255

        result = PlacementPlanner(video, layout).plan(
            [caption_plan("two person podcast", 0.0, 1.0)],
            [frame(0.5, 0, person=person)],
            [(180, 90)],
        )[0]

        self.assertEqual(result.placement.name, "bottom-center")
        self.assertEqual(result.head_overlaps["bottom-center"], 0.0)
        self.assertTrue(any(
            result.head_overlaps[name] > 0.0
            for name in ("top-left", "top-right")
        ))

    def test_no_person_portrait_prefers_bottom_center_with_normal_threshold(self):
        video = VideoInfo(600, 1000, 30.0, 4.0)
        layout = LayoutConfig(margin_x=90, margin_y=60)
        planner = PlacementPlanner(video, layout, hysteresis=0.12)

        result = planner.plan(
            [caption_plan("clean scene", 0.0, 1.0)],
            [FrameAnalysis(
                timestamp=0.5,
                frame_index=1,
                map_width=60,
                map_height=100,
                person_map=np.zeros((100, 60), dtype=np.uint8),
                clutter_map=np.zeros((100, 60), dtype=np.uint8),
                motion_map=np.zeros((100, 60), dtype=np.uint8),
            )],
            [(180, 90)],
        )[0]

        self.assertTrue(result.no_person_context)
        self.assertEqual(result.effective_hysteresis, 0.12)
        self.assertEqual(NO_PERSON_HYSTERESIS, PLACEMENT_CHANGE_THRESHOLD)
        self.assertEqual(result.placement.name, "bottom-center")
        self.assertFalse(result.baseline_tiebreak_applied)
        self.assertEqual(result.change_reason, "baseline-bottom")

    def test_person_context_preserves_normal_hysteresis(self):
        video = VideoInfo(600, 1000, 30.0, 4.0)
        person = np.full((100, 60), 100, dtype=np.uint8)
        planner = PlacementPlanner(
            video,
            LayoutConfig(margin_x=90, margin_y=60),
            hysteresis=0.12,
        )

        result = planner.plan(
            [caption_plan("person scene", 0.0, 1.0)],
            [FrameAnalysis(
                timestamp=0.5,
                frame_index=1,
                map_width=60,
                map_height=100,
                person_map=person,
                clutter_map=np.zeros_like(person),
                motion_map=np.zeros_like(person),
            )],
            [(180, 90)],
        )[0]

        self.assertFalse(result.no_person_context)
        self.assertEqual(result.effective_hysteresis, 0.12)
        self.assertTrue(result.baseline_tiebreak_applied)

    def test_bottom_center_loses_when_clearly_more_cluttered(self):
        video = VideoInfo(600, 1000, 30.0, 4.0)
        layout = LayoutConfig(margin_x=90, margin_y=60)
        candidates = {
            item.name: item
            for item in generate_candidates(video, (180, 90), layout)
        }
        clutter = np.zeros((100, 60), dtype=np.uint8)
        fill_scaled_region(clutter, candidates["bottom-center"], video, 255)
        planner = PlacementPlanner(video, layout, hysteresis=0.12)

        result = planner.plan(
            [caption_plan("busy bottom", 0.0, 1.0)],
            [FrameAnalysis(
                timestamp=0.5,
                frame_index=1,
                map_width=60,
                map_height=100,
                person_map=np.zeros_like(clutter),
                clutter_map=clutter,
                motion_map=np.zeros_like(clutter),
            )],
            [(180, 90)],
        )[0]

        self.assertTrue(result.no_person_context)
        self.assertNotEqual(result.placement.name, "bottom-center")

    def test_identical_no_person_frames_do_not_cycle_positions(self):
        video = VideoInfo(600, 1000, 30.0, 4.0)
        empty = np.zeros((100, 60), dtype=np.uint8)
        planner = PlacementPlanner(
            video,
            LayoutConfig(margin_x=90, margin_y=60),
            hysteresis=0.12,
        )
        captions = [
            caption_plan(f"caption {index}", float(index), float(index + 1))
            for index in range(3)
        ]
        analyses = [
            FrameAnalysis(
                timestamp=index + 0.5,
                frame_index=index,
                map_width=60,
                map_height=100,
                person_map=empty,
                clutter_map=empty,
                motion_map=empty,
            )
            for index in range(3)
        ]

        placements = planner.plan(captions, analyses, [(180, 90)] * 3)

        self.assertEqual(
            [item.placement.name for item in placements],
            ["bottom-center"] * 3,
        )

    def test_no_person_position_changes_when_previous_becomes_clearly_worse(self):
        video = VideoInfo(600, 1000, 30.0, 4.0)
        layout = LayoutConfig(margin_x=90, margin_y=60)
        candidates = {
            item.name: item
            for item in generate_candidates(video, (180, 90), layout)
        }
        empty = np.zeros((100, 60), dtype=np.uint8)
        busy_bottom = empty.copy()
        fill_scaled_region(
            busy_bottom,
            candidates["bottom-center"],
            video,
            255,
        )
        planner = PlacementPlanner(video, layout, hysteresis=0.12)
        captions = [
            caption_plan("clean bottom", 0.0, 1.0),
            caption_plan("bottom becomes busy", 1.0, 2.0),
        ]
        analyses = [
            FrameAnalysis(
                timestamp=0.5,
                frame_index=0,
                map_width=60,
                map_height=100,
                person_map=empty,
                clutter_map=empty,
                motion_map=empty,
            ),
            FrameAnalysis(
                timestamp=1.5,
                frame_index=1,
                map_width=60,
                map_height=100,
                person_map=empty,
                clutter_map=busy_bottom,
                motion_map=empty,
            ),
        ]

        placements = planner.plan(captions, analyses, [(180, 90)] * 2)

        self.assertEqual(placements[0].placement.name, "bottom-center")
        self.assertNotEqual(placements[1].placement.name, "bottom-center")
        self.assertFalse(placements[1].hysteresis_applied)

    def test_unsafe_foreground_object_forces_anchor_change(self):
        video = VideoInfo(600, 1000, 30.0, 3.0)
        layout = LayoutConfig(margin_x=90, margin_y=60)
        candidates = {
            item.name: item
            for item in generate_candidates(video, (180, 90), layout)
        }
        empty = np.zeros((100, 100), dtype=np.uint8)
        foreground = empty.copy()
        fill_scaled_region(
            foreground,
            candidates["bottom-center"],
            video,
            255,
        )
        planner = PlacementPlanner(video, layout)

        plans = planner.plan(
            [caption_plan("one", 0.0, 1.0), caption_plan("two", 1.0, 2.0)],
            [
                frame(0.5, 0),
                frame(
                    1.5,
                    1,
                    foreground=foreground,
                    foreground_type="object",
                ),
            ],
            [(180, 90), (180, 90)],
        )

        self.assertEqual(plans[0].persistent_anchor, "bottom-center")
        self.assertNotEqual(plans[1].persistent_anchor, "bottom-center")
        self.assertEqual(plans[1].change_reason, "foreground-obstruction")

    def test_severe_clutter_forces_anchor_change_with_reason(self):
        video = VideoInfo(600, 1000, 30.0, 3.0)
        layout = LayoutConfig(margin_x=90, margin_y=60)
        candidates = {
            item.name: item
            for item in generate_candidates(video, (180, 90), layout)
        }
        clutter = np.zeros((100, 100), dtype=np.uint8)
        fill_scaled_region(clutter, candidates["bottom-center"], video, 255)
        planner = PlacementPlanner(video, layout)

        plans = planner.plan(
            [caption_plan("one", 0.0, 1.0), caption_plan("two", 1.0, 2.0)],
            [frame(0.5, 0), frame(1.5, 1, clutter=clutter)],
            [(180, 90), (180, 90)],
        )

        self.assertNotEqual(plans[1].persistent_anchor, "bottom-center")
        self.assertEqual(plans[1].change_reason, "clutter-improvement")

    def test_severe_motion_forces_anchor_change_with_reason(self):
        video = VideoInfo(600, 1000, 30.0, 3.0)
        layout = LayoutConfig(margin_x=90, margin_y=60)
        candidates = {
            item.name: item
            for item in generate_candidates(video, (180, 90), layout)
        }
        motion = np.zeros((100, 100), dtype=np.uint8)
        fill_scaled_region(motion, candidates["bottom-center"], video, 255)
        planner = PlacementPlanner(video, layout)

        plans = planner.plan(
            [caption_plan("one", 0.0, 1.0), caption_plan("two", 1.0, 2.0)],
            [frame(0.5, 0), frame(1.5, 1, motion=motion)],
            [(180, 90), (180, 90)],
        )

        self.assertNotEqual(plans[1].persistent_anchor, "bottom-center")
        self.assertEqual(plans[1].change_reason, "motion-improvement")

    def test_scene_cut_reconsiders_anchor_without_movement_inertia(self):
        video = VideoInfo(600, 1000, 30.0, 3.0)
        layout = LayoutConfig(margin_x=90, margin_y=60)
        candidates = {
            item.name: item
            for item in generate_candidates(video, (180, 90), layout)
        }
        clutter = np.zeros((100, 100), dtype=np.uint8)
        fill_scaled_region(clutter, candidates["bottom-center"], video, 255)
        planner = PlacementPlanner(video, layout)

        plans = planner.plan(
            [caption_plan("one", 0.0, 1.0), caption_plan("two", 1.0, 2.0)],
            [
                frame(0.5, 0),
                frame(1.5, 1, clutter=clutter, scene_cut=True),
            ],
            [(180, 90), (180, 90)],
        )

        self.assertEqual(plans[0].persistent_anchor, "bottom-center")
        self.assertNotEqual(plans[1].persistent_anchor, "bottom-center")
        self.assertTrue(plans[1].scene_cut)
        self.assertEqual(plans[1].change_reason, "scene-cut")

    def test_object_overlap_elsewhere_never_creates_temporary_placement(self):
        video = VideoInfo(600, 1000, 30.0, 4.0)
        layout = LayoutConfig(margin_x=90, margin_y=60)
        candidates = {
            item.name: item
            for item in generate_candidates(video, (180, 90), layout)
        }
        foreground = np.zeros((100, 100), dtype=np.uint8)
        fill_scaled_region(foreground, candidates["middle-right"], video, 46)
        planner = PlacementPlanner(
            video,
            layout,
            allow_occlusion=True,
            occlusion_min_overlap=0.10,
            occlusion_max_overlap=0.30,
        )
        captions = [
            caption_plan("first stable caption", 0.0, 1.0),
            caption_plan("cinematic overlap moment", 1.0, 2.0),
            caption_plan("back to baseline", 2.0, 3.0),
        ]

        plans = planner.plan(
            captions,
            [
                frame(0.5, 0),
                frame(
                    1.5,
                    1,
                    foreground=foreground,
                    foreground_type="object",
                ),
                frame(2.5, 2),
            ],
            [(180, 90)] * 3,
        )

        self.assertEqual(plans[0].persistent_anchor, "bottom-center")
        self.assertEqual(plans[1].persistent_anchor, "bottom-center")
        self.assertFalse(plans[1].temporary_placement)
        self.assertEqual(plans[1].placement.name, "bottom-center")
        self.assertEqual(plans[1].change_reason, "baseline-bottom")
        self.assertEqual(plans[2].persistent_anchor, "bottom-center")
        self.assertEqual(plans[2].placement.name, "bottom-center")
        self.assertEqual(plans[2].change_reason, "baseline-bottom")

    def test_person_away_from_bottom_keeps_portrait_baseline(self):
        video = VideoInfo(600, 1000, 30.0, 2.0)
        layout = LayoutConfig(margin_x=90, margin_y=60)
        candidates = {
            item.name: item
            for item in generate_candidates(video, (180, 90), layout)
        }
        person = np.zeros((100, 100), dtype=np.uint8)
        fill_scaled_region(person, candidates["top-left"], video, 255)

        result = PlacementPlanner(video, layout).plan(
            [caption_plan("person elsewhere", 0.0, 1.0)],
            [frame(0.5, 0, person=person)],
            [(180, 90)],
        )[0]

        self.assertTrue(result.person_present)
        self.assertTrue(result.bottom_center_safe)
        self.assertEqual(result.placement.name, "bottom-center")
        self.assertEqual(result.change_reason, "baseline-bottom")

    def test_person_obstruction_relocates_stays_then_returns_to_bottom(self):
        video = VideoInfo(600, 1000, 30.0, 5.0)
        layout = LayoutConfig(margin_x=90, margin_y=60)
        candidates = {
            item.name: item
            for item in generate_candidates(video, (180, 90), layout)
        }
        person = np.zeros((100, 100), dtype=np.uint8)
        fill_scaled_region(person, candidates["bottom-center"], video, 255)
        planner = PlacementPlanner(video, layout)
        captions = [
            caption_plan("baseline", 0.0, 1.0),
            caption_plan("person enters", 1.0, 2.0),
            caption_plan("person remains", 2.0, 3.0),
            caption_plan("person leaves", 3.0, 4.0),
        ]

        plans = planner.plan(
            captions,
            [
                frame(0.5, 0),
                frame(1.5, 1, person=person),
                frame(2.5, 2, person=person),
                frame(3.5, 3),
            ],
            [(180, 90)] * 4,
        )

        relocated = plans[1].placement.name
        self.assertEqual(plans[0].placement.name, "bottom-center")
        self.assertNotEqual(relocated, "bottom-center")
        self.assertEqual(plans[1].change_reason, "person-obstruction")
        self.assertEqual(plans[2].placement.name, relocated)
        self.assertEqual(plans[2].change_reason, "person-aware-retained")
        self.assertEqual(plans[3].placement.name, "bottom-center")
        self.assertEqual(plans[3].change_reason, "return-to-bottom")
        self.assertFalse(plans[3].person_present)

    def test_object_away_from_bottom_does_not_move_caption(self):
        video = VideoInfo(600, 1000, 30.0, 2.0)
        layout = LayoutConfig(margin_x=90, margin_y=60)
        candidates = {
            item.name: item
            for item in generate_candidates(video, (180, 90), layout)
        }
        foreground = np.zeros((100, 100), dtype=np.uint8)
        fill_scaled_region(foreground, candidates["top-right"], video, 255)

        result = PlacementPlanner(video, layout).plan(
            [caption_plan("object elsewhere", 0.0, 1.0)],
            [frame(
                0.5,
                0,
                foreground=foreground,
                foreground_type="object",
            )],
            [(180, 90)],
        )[0]

        self.assertFalse(result.person_present)
        self.assertTrue(result.bottom_center_safe)
        self.assertEqual(result.placement.name, "bottom-center")

    def test_person_elsewhere_never_creates_temporary_placement(self):
        video = VideoInfo(600, 1000, 30.0, 4.0)
        layout = LayoutConfig(margin_x=90, margin_y=60)
        candidates = {
            item.name: item
            for item in generate_candidates(video, (180, 90), layout)
        }
        person = np.zeros((100, 100), dtype=np.uint8)
        fill_scaled_region(person, candidates["middle-right"], video, 38)
        planner = PlacementPlanner(
            video,
            layout,
            allow_occlusion=True,
            occlusion_min_overlap=0.10,
            occlusion_max_overlap=0.30,
        )

        plans = planner.plan(
            [
                caption_plan("normal caption", 0.0, 1.0),
                caption_plan("person overlap moment", 1.0, 2.0),
                caption_plan("normal again", 2.0, 3.0),
            ],
            [
                frame(0.5, 0),
                frame(
                    1.5,
                    1,
                    person=person,
                    foreground=person,
                    foreground_type="person",
                ),
                frame(2.5, 2),
            ],
            [(180, 90)] * 3,
        )

        self.assertEqual(plans[1].persistent_anchor, "bottom-center")
        self.assertFalse(plans[1].temporary_placement)
        self.assertEqual(plans[1].placement.name, "bottom-center")
        self.assertEqual(plans[1].change_reason, "baseline-bottom")
        self.assertEqual(plans[2].placement.name, "bottom-center")
        self.assertEqual(plans[2].change_reason, "baseline-bottom")

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
        self.assertTrue(plans[1].anchor_retained)
        self.assertEqual(plans[1].change_reason, "retained-anchor")
        self.assertLessEqual(plans[1].previous_person_overlap, 0.30)

    def test_anchor_stays_for_point_zero_two_improvement(self):
        planner = PlacementPlanner(VIDEO, LAYOUT)
        self.assertFalse(planner._should_change_anchor(0.02))

    def test_anchor_stays_for_point_zero_five_improvement(self):
        planner = PlacementPlanner(VIDEO, LAYOUT)
        self.assertFalse(planner._should_change_anchor(0.05))

    def test_anchor_moves_above_point_zero_eight_improvement(self):
        planner = PlacementPlanner(VIDEO, LAYOUT)
        self.assertTrue(planner._should_change_anchor(0.081))

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
