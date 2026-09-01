from __future__ import annotations

import unittest

import numpy as np

from killer_subtitles.compositor import restore_foreground_pixels
from killer_subtitles.models import (
    Caption,
    CaptionPlan,
    CaptionStyle,
    FrameAnalysis,
    Tone,
    VideoInfo,
    Word,
)
from killer_subtitles.occlusion import (
    TemporalMaskProvider,
    OcclusionPlanner,
    clean_person_mask,
    decide_occlusion,
    occlusion_opportunity_score,
)


STYLE = CaptionStyle("#fff", "#ff0", "#ff0")
PLAN = CaptionPlan(
    caption=Caption([Word("foreground", 0.0, 1.0)]),
    tone=Tone.NEUTRAL,
    keyword_indices=(),
    style=STYLE,
)


def analysis(timestamp: float, index: int, mask) -> FrameAnalysis:
    empty = np.zeros((4, 4), dtype=np.uint8)
    return FrameAnalysis(
        timestamp=timestamp,
        frame_index=index,
        map_width=4,
        map_height=4,
        person_map=mask,
        clutter_map=empty,
        motion_map=empty,
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
        self.assertIn("unavailable", decision.reason)

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
        self.assertEqual(decision.reason, "sweet-spot partial person overlap")
        self.assertGreater(decision.opportunity_score, 0.8)

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
