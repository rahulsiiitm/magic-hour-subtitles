from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from magic_hour_subtitles.compositor import restore_foreground_pixels
from magic_hour_subtitles.models import (
    Caption,
    CaptionPlan,
    CaptionStyle,
    FrameAnalysis,
    Placement,
    PlacementPlan,
    StyleConfig,
    SubtitleState,
    Tone,
    VideoInfo,
    Word,
)
from magic_hour_subtitles.occlusion import (
    OCCLUSION_HEAD_TOLERANCE,
    TemporalMaskProvider,
    OcclusionPlanner,
    clean_person_mask,
    decide_occlusion,
    occlusion_opportunity_score,
)


STYLE = CaptionStyle("#fff", "#ff0", "#ff0")
FONT_PATH = (
    Path(__file__).resolve().parents[1]
    / "magic_hour_subtitles"
    / "fonts"
    / "Montserrat-ExtraBold.ttf"
)
PLAN = CaptionPlan(
    caption=Caption([Word("foreground", 0.0, 1.0)]),
    tone=Tone.NEUTRAL,
    keyword_indices=(),
    style=STYLE,
)


def analysis(
    timestamp: float,
    index: int,
    mask,
    *,
    foreground=None,
    foreground_type: str = "none",
) -> FrameAnalysis:
    empty = np.zeros((4, 4), dtype=np.uint8)
    return FrameAnalysis(
        timestamp=timestamp,
        frame_index=index,
        map_width=4,
        map_height=4,
        person_map=mask,
        clutter_map=empty,
        motion_map=empty,
        foreground_map=foreground,
        foreground_type=foreground_type,
    )


class OcclusionDecisionTests(unittest.TestCase):
    def test_zero_mask_means_normal_caption(self):
        decision = decide_occlusion(
            PLAN,
            has_mask=False,
            person_overlap=0.0,
            caption_occlusion=0.0,
            min_overlap=0.10,
            max_occlusion=0.45,
        )
        self.assertFalse(decision.enabled)
        self.assertEqual(decision.reason, "no-foreground")

    def test_moderate_overlap_enables_effect(self):
        decision = decide_occlusion(
            PLAN,
            has_mask=True,
            person_overlap=0.18,
            caption_occlusion=0.22,
            min_overlap=0.10,
            max_occlusion=0.45,
        )
        self.assertTrue(decision.enabled)
        self.assertEqual(decision.reason, "natural-overlap-sweet-spot")
        self.assertGreater(decision.opportunity_score, 0.8)

    def test_face_overlap_hard_rejects_behind_person(self):
        decision = decide_occlusion(
            PLAN,
            has_mask=True,
            person_overlap=0.18,
            caption_occlusion=0.16,
            min_overlap=0.10,
            max_occlusion=0.45,
            head_overlap=OCCLUSION_HEAD_TOLERANCE + 0.01,
        )

        self.assertFalse(decision.enabled)
        self.assertFalse(decision.head_safe)
        self.assertEqual(decision.reason, "protected-head-region")
        self.assertEqual(decision.rejection_code, "head_overlap")

    def test_shoulder_overlap_remains_eligible_when_head_is_clear(self):
        decision = decide_occlusion(
            PLAN,
            has_mask=True,
            person_overlap=0.18,
            caption_occlusion=0.16,
            min_overlap=0.10,
            max_occlusion=0.45,
            head_overlap=0.0,
        )

        self.assertTrue(decision.enabled)
        self.assertTrue(decision.head_safe)
        self.assertEqual(decision.reason, "natural-overlap-sweet-spot")

    def test_excessive_occlusion_disables_effect(self):
        decision = decide_occlusion(
            PLAN,
            has_mask=True,
            person_overlap=0.50,
            caption_occlusion=0.63,
            min_overlap=0.10,
            max_occlusion=0.45,
        )
        self.assertFalse(decision.enabled)
        self.assertIn("63%", decision.reason)
        self.assertEqual(decision.rejection_code, "high_occlusion")

    def test_sweet_spot_scores_above_barely_visible_overlap(self):
        sweet = occlusion_opportunity_score(0.18, 0.16, 3, 0.8)
        subtle = occlusion_opportunity_score(0.18, 0.03, 3, 0.8)
        self.assertGreater(sweet, subtle)

    def test_tiny_caption_is_not_preferred_over_meaningful_caption(self):
        tiny_plan = CaptionPlan(
            Caption([Word("days.", 0.0, 0.5)]),
            Tone.NEUTRAL,
            (),
            STYLE,
        )
        meaningful_plan = CaptionPlan(
            Caption([
                Word("code", 0.5, 0.8),
                Word("contributors", 0.8, 1.2),
                Word("arrive", 1.2, 1.5),
            ]),
            Tone.NEUTRAL,
            (),
            STYLE,
        )
        decisions = [
            decide_occlusion(
                tiny_plan,
                has_mask=True,
                person_overlap=0.18,
                caption_occlusion=0.20,
                min_overlap=0.10,
                max_occlusion=0.45,
            ),
            decide_occlusion(
                meaningful_plan,
                has_mask=True,
                person_overlap=0.18,
                caption_occlusion=0.20,
                min_overlap=0.10,
                max_occlusion=0.45,
            ),
        ]

        preferred = OcclusionPlanner._prefer_meaningful_opportunities(decisions)

        self.assertFalse(preferred[0].enabled)
        self.assertTrue(preferred[1].enabled)

    def test_subtle_one_word_caption_is_rejected_as_too_short(self):
        decision = decide_occlusion(
            PLAN,
            has_mask=True,
            person_overlap=0.18,
            caption_occlusion=0.08,
            min_overlap=0.10,
            max_occlusion=0.45,
        )
        self.assertFalse(decision.enabled)
        self.assertEqual(decision.rejection_code, "too_short")

    def test_readable_object_occlusion_can_activate(self):
        decision = decide_occlusion(
            CaptionPlan(
                Caption([
                    Word("useful", 0.0, 0.4),
                    Word("object", 0.4, 0.8),
                    Word("overlap", 0.8, 1.2),
                ]),
                Tone.NEUTRAL,
                (),
                STYLE,
            ),
            has_mask=True,
            person_overlap=0.0,
            foreground_overlap=0.18,
            foreground_type="object",
            caption_occlusion=0.16,
            min_overlap=0.10,
            max_occlusion=0.45,
        )
        self.assertTrue(decision.enabled)
        self.assertEqual(decision.person_overlap, 0.0)
        self.assertEqual(decision.foreground_overlap, 0.18)
        self.assertEqual(decision.foreground_type, "object")

    def test_unsafe_object_occlusion_is_rejected(self):
        decision = decide_occlusion(
            PLAN,
            has_mask=True,
            person_overlap=0.0,
            foreground_overlap=0.60,
            foreground_type="object",
            caption_occlusion=0.60,
            min_overlap=0.10,
            max_occlusion=0.45,
        )
        self.assertFalse(decision.enabled)
        self.assertEqual(decision.rejection_code, "high_occlusion")

    def test_natural_overlap_is_evaluated_at_unchanged_chosen_placement(self):
        video = VideoInfo(100, 100, 30.0, 1.0)
        chosen = Placement("bottom-center", 20, 70, 60, 20)
        plan = CaptionPlan(
            Caption([
                Word("natural", 0.0, 0.3),
                Word("shoulder", 0.3, 0.6),
                Word("overlap", 0.6, 0.9),
            ]),
            Tone.NEUTRAL,
            (),
            STYLE,
        )
        placement_plan = PlacementPlan(
            plan,
            chosen,
            person_overlaps={"bottom-center": 1 / 6},
            head_overlaps={"bottom-center": 0.0},
            foreground_overlaps={"bottom-center": 1 / 6},
            foreground_type="person",
            persistent_anchor="bottom-center",
            temporary_placement=False,
        )
        foreground = np.zeros((10, 10), dtype=np.uint8)
        foreground[7:9, 2:3] = 255
        empty = np.zeros_like(foreground)
        frames = [FrameAnalysis(
            timestamp=0.45,
            frame_index=0,
            map_width=10,
            map_height=10,
            person_map=foreground,
            clutter_map=empty,
            motion_map=empty,
            foreground_map=foreground,
            foreground_type="person",
        )]
        planner = OcclusionPlanner(
            video,
            StyleConfig(font_path=str(FONT_PATH), font_size=12),
            frames,
        )
        rendered = np.zeros((100, 100, 4), dtype=np.uint8)
        rendered[70:90, 20:80, 3] = 255
        planner.renderer.render = lambda _state: Image.fromarray(rendered, "RGBA")

        decision = planner.plan(
            [placement_plan],
            [SubtitleState(start=0.0, end=0.9)],
        )[0]

        self.assertIs(placement_plan.placement, chosen)
        self.assertFalse(placement_plan.temporary_placement)
        self.assertTrue(decision.enabled)
        self.assertAlmostEqual(decision.caption_occlusion, 1 / 6, places=2)


class MaskTests(unittest.TestCase):
    def test_mask_resize(self):
        mask = np.zeros((2, 2), dtype=np.uint8)
        mask[0, 0] = 255
        resized = clean_person_mask(mask, 8, 6, dilate=0, blur=0)
        self.assertEqual(resized.shape, (6, 8))
        self.assertEqual(resized.dtype, np.uint8)

    def test_mask_feathering_produces_soft_boundary(self):
        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[3:6, 3:6] = 255
        feathered = clean_person_mask(mask, 9, 9, dilate=0, blur=5)
        self.assertTrue(np.any((feathered > 0) & (feathered < 255)))

    def test_unavailable_mask_lookup_returns_none(self):
        provider = TemporalMaskProvider([], output_width=8, output_height=6)
        self.assertIsNone(provider.mask_at(0.5))

    def test_first_and_last_frame_mask_lookup(self):
        first = np.zeros((4, 4), dtype=np.uint8)
        last = np.full((4, 4), 255, dtype=np.uint8)
        provider = TemporalMaskProvider(
            [analysis(1.0, 1, first), analysis(2.0, 2, last)],
            output_width=4,
            output_height=4,
            dilate=0,
            blur=0,
        )
        np.testing.assert_array_equal(provider.reduced_mask_at(0.0), first)
        np.testing.assert_array_equal(provider.reduced_mask_at(3.0), last)

    def test_temporal_lookup_interpolates_between_sampled_masks(self):
        first = np.zeros((4, 4), dtype=np.uint8)
        last = np.full((4, 4), 255, dtype=np.uint8)
        provider = TemporalMaskProvider(
            [analysis(0.0, 0, first), analysis(1.0, 1, last)],
            output_width=4,
            output_height=4,
            dilate=0,
            blur=0,
        )
        middle = provider.reduced_mask_at(0.5)
        self.assertTrue(np.all((middle >= 127) & (middle <= 128)))

    def test_foreground_mask_is_preferred_without_changing_person_map(self):
        person = np.zeros((4, 4), dtype=np.uint8)
        foreground = np.full((4, 4), 255, dtype=np.uint8)
        frame = analysis(
            0.0,
            0,
            person,
            foreground=foreground,
            foreground_type="object",
        )
        provider = TemporalMaskProvider(
            [frame],
            output_width=4,
            output_height=4,
            dilate=0,
            blur=0,
        )

        np.testing.assert_array_equal(provider.reduced_mask_at(0.0), foreground)
        np.testing.assert_array_equal(frame.person_map, person)


class CompositionTests(unittest.TestCase):
    def test_person_pixels_are_restored_over_caption(self):
        original = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        captioned = np.array([[[200, 210, 220], [220, 210, 200]]], dtype=np.uint8)
        mask = np.array([[255, 0]], dtype=np.uint8)

        result = restore_foreground_pixels(original, captioned, mask)

        np.testing.assert_array_equal(result[0, 0], original[0, 0])

    def test_non_person_pixels_retain_caption(self):
        original = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        captioned = np.array([[[200, 210, 220], [220, 210, 200]]], dtype=np.uint8)
        mask = np.array([[255, 0]], dtype=np.uint8)

        result = restore_foreground_pixels(original, captioned, mask)

        np.testing.assert_array_equal(result[0, 1], captioned[0, 1])


if __name__ == "__main__":
    unittest.main()
