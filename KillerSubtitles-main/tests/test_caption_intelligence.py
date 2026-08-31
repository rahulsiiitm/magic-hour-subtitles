from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from killer_subtitles.caption_analysis import (
    STOPWORDS,
    analyze_captions,
    classify_tone,
    normalize_word,
    select_keyword_indices,
)
from killer_subtitles.caption_chunker import chunk_words
from killer_subtitles.compositor import _normalise_timeline
from killer_subtitles.layout import LayoutEngine
from killer_subtitles.models import (
    Caption,
    LayoutConfig,
    Placement,
    PlacementPlan,
    StyleConfig,
    Tone,
    VideoInfo,
    Word,
)
from killer_subtitles.renderer import SubtitleRenderer


ROOT = Path(__file__).resolve().parents[1]
FONT_PATH = ROOT / "killer_subtitles" / "fonts" / "Montserrat-ExtraBold.ttf"


def timed_words(texts: list[str], *, gap_after: dict[int, float] | None = None):
    words: list[Word] = []
    cursor = 0.0
    gap_after = gap_after or {}
    for index, text in enumerate(texts):
        words.append(Word(text, cursor, cursor + 0.2))
        cursor += 0.2 + gap_after.get(index, 0.0)
    return words


class CaptionChunkerTests(unittest.TestCase):
    def test_strong_punctuation_breaks_caption(self):
        words = timed_words(["Hello", "world.", "This", "is", "next"])
        captions = chunk_words(words)
        self.assertEqual(
            [[word.text for word in caption.words] for caption in captions],
            [["Hello", "world."], ["This", "is", "next"]],
        )

    def test_pause_breaks_caption(self):
        words = timed_words(
            ["one", "two", "three", "four", "five", "six"],
            gap_after={2: 0.6},
        )
        captions = chunk_words(words)
        self.assertEqual([len(caption.words) for caption in captions], [3, 3])

    def test_trailing_single_word_is_rebalanced(self):
        words = timed_words(["one", "two", "three", "four", "five", "six", "seven"])
        captions = chunk_words(words)
        self.assertEqual([len(caption.words) for caption in captions], [7])

    def test_rebalance_does_not_cross_strong_punctuation(self):
        words = timed_words(["This", "is", "done.", "Next"])
        captions = chunk_words(words)
        self.assertEqual([len(caption.words) for caption in captions], [3, 1])

    def test_original_words_and_timestamps_are_preserved(self):
        words = timed_words(["keep", "these", "timestamps.", "exactly"])
        original_times = [(word.start, word.end) for word in words]
        captions = chunk_words(words)
        flattened = [word for caption in captions for word in caption.words]

        self.assertEqual(flattened, words)
        self.assertTrue(all(actual is original for actual, original in zip(flattened, words)))
        self.assertEqual([(word.start, word.end) for word in flattened], original_times)


class CaptionAnalysisTests(unittest.TestCase):
    def test_question_tone_has_highest_priority(self):
        caption = Caption(timed_words(["How", "amazing", "is", "this?"]))
        self.assertEqual(classify_tone(caption), Tone.QUESTION)

    def test_excited_tone(self):
        caption = Caption(timed_words(["This", "is", "an", "incredible", "breakthrough!"]))
        self.assertEqual(classify_tone(caption), Tone.EXCITED)

    def test_serious_tone(self):
        caption = Caption(timed_words(["This", "warning", "is", "critical."]))
        self.assertEqual(classify_tone(caption), Tone.SERIOUS)

    def test_added_serious_cues_with_punctuation(self):
        for cue in ("scared", "death", "gun", "danger", "trapped", "terrifying"):
            with self.subTest(cue=cue):
                caption = Caption(timed_words(["This", "is", f"{cue}."]))
                self.assertEqual(classify_tone(caption), Tone.SERIOUS)

    def test_keywords_exclude_stopwords(self):
        caption = Caption(timed_words(["the", "amazing", "change", "is"]))
        frequencies = Counter(normalize_word(word.text) for word in caption.words)
        indices = select_keyword_indices(caption, frequencies)
        selected = {normalize_word(caption.words[index].text) for index in indices}
        self.assertFalse(selected & STOPWORDS)
        self.assertEqual(selected, {"amazing"})

    def test_keyword_analysis_strips_only_surrounding_punctuation(self):
        original = "\u201cgame-changing!\u201d"
        caption = Caption(timed_words([original, "oh", "really"]))
        frequencies = Counter(normalize_word(word.text) for word in caption.words)

        indices = select_keyword_indices(caption, frequencies)

        self.assertEqual(normalize_word(original), "game-changing")
        self.assertEqual(indices, (0,))
        self.assertEqual(caption.words[0].text, original)

    def test_common_filler_words_are_not_keywords(self):
        caption = Caption(timed_words([
            "oh", "well", "did", "all", "these", "those", "about", "there",
            "here", "really",
        ]))
        frequencies = Counter(normalize_word(word.text) for word in caption.words)
        self.assertEqual(select_keyword_indices(caption, frequencies), ())

    def test_keyword_limits(self):
        short = Caption(timed_words(["bright", "future", "arrives", "today"]))
        long = Caption(timed_words([
            "bright", "future", "technology", "arrives", "globally", "today",
        ]))
        plans = analyze_captions([short, long])
        self.assertLessEqual(len(plans[0].keyword_indices), 1)
        self.assertLessEqual(len(plans[1].keyword_indices), 2)
        self.assertEqual(len(plans[1].keyword_indices), 2)

    def test_dynamic_layout_keeps_word_positions_stable(self):
        caption = Caption(timed_words(["This", "changes", "everything!"]))
        plan = analyze_captions([caption])[0]
        engine = LayoutEngine(
            VideoInfo(640, 360, 30.0, 2.0),
            StyleConfig(font_path=str(FONT_PATH), font_size=40),
            LayoutConfig(mode="karaoke", margin_x=40, margin_y=20),
        )

        states = engine.build_dynamic_states([plan])

        self.assertEqual(len(states), len(caption.words))
        positions = [
            [(word.x, word.y) for word in state.rendered_words]
            for state in states
        ]
        self.assertTrue(all(position == positions[0] for position in positions))
        self.assertEqual(
            [(round(state.start, 3), round(state.end, 3)) for state in states],
            [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6)],
        )
        image = SubtitleRenderer(engine.video, engine.style).render(states[0])
        self.assertIsNotNone(image.getbbox())

    def test_dynamic_layout_applies_one_caption_placement(self):
        caption = Caption(timed_words(["Stable", "placed", "caption."]))
        plan = analyze_captions([caption])[0]
        engine = LayoutEngine(
            VideoInfo(640, 360, 30.0, 2.0),
            StyleConfig(font_path=str(FONT_PATH), font_size=40),
            LayoutConfig(margin_x=40, margin_y=20),
        )
        width, height = engine.measure_dynamic_caption(plan)
        placement = Placement("top-left", 45, 25, width, height)

        states = engine.build_dynamic_states(
            [plan],
            [PlacementPlan(plan, placement)],
        )

        positions = [
            [(word.x, word.y) for word in state.rendered_words]
            for state in states
        ]
        self.assertTrue(all(position == positions[0] for position in positions))
        self.assertGreaterEqual(min(word.x for word in states[0].rendered_words), 45)
        self.assertGreaterEqual(min(word.y for word in states[0].rendered_words), 25)

    def test_dynamic_layout_honors_uppercase_without_mutating_words(self):
        caption = Caption(timed_words(["Keep", "source", "text."]))
        plan = analyze_captions([caption])[0]
        engine = LayoutEngine(
            VideoInfo(640, 360, 30.0, 2.0),
            StyleConfig(font_path=str(FONT_PATH), font_size=40, uppercase=True),
            LayoutConfig(margin_x=40, margin_y=20),
        )

        states = engine.build_dynamic_states([plan])

        self.assertEqual(
            [word.text for word in states[0].rendered_words],
            ["KEEP", "SOURCE", "TEXT."],
        )
        self.assertEqual([word.text for word in caption.words], ["Keep", "source", "text."])

    def test_zero_duration_dynamic_state_is_visible_and_clamped(self):
        captions = [
            Caption([Word("zero", 1.0, 1.0)]),
            Caption([Word("next", 1.07, 1.3)]),
        ]
        engine = LayoutEngine(
            VideoInfo(640, 360, 30.0, 2.0),
            StyleConfig(font_path=str(FONT_PATH), font_size=40),
            LayoutConfig(margin_x=40, margin_y=20),
        )

        states = engine.build_dynamic_states(analyze_captions(captions))

        self.assertGreater(states[0].end, states[0].start)
        self.assertAlmostEqual(states[0].end, captions[1].start)
        self.assertLessEqual(states[0].end, states[1].start)

    def test_final_zero_duration_dynamic_state_gets_minimum_duration(self):
        caption = Caption([Word("zero", 1.0, 1.0)])
        engine = LayoutEngine(
            VideoInfo(640, 360, 30.0, 2.0),
            StyleConfig(font_path=str(FONT_PATH), font_size=40),
            LayoutConfig(margin_x=40, margin_y=20),
        )

        state = engine.build_dynamic_states(analyze_captions([caption]))[0]

        self.assertAlmostEqual(state.end - state.start, 0.10)

    def test_short_valid_dynamic_state_keeps_original_duration(self):
        caption = Caption([Word("brief", 1.0, 1.05)])
        engine = LayoutEngine(
            VideoInfo(640, 360, 30.0, 2.0),
            StyleConfig(font_path=str(FONT_PATH), font_size=40),
            LayoutConfig(margin_x=40, margin_y=20),
        )

        state = engine.build_dynamic_states(analyze_captions([caption]))[0]

        self.assertAlmostEqual(state.end, 1.05)

    def test_long_transcript_remains_complete_through_normalization(self):
        words: list[Word] = []
        cursor = 0.0
        for index in range(600):
            suffix = "." if index % 17 == 16 else ""
            duration = 0.08 + (index % 5) * 0.03
            words.append(Word(f"word{index}{suffix}", cursor, cursor + duration))
            cursor += duration + (0.55 if index % 53 == 52 else 0.02)

        captions = chunk_words(words)
        chunked = [word for caption in captions for word in caption.words]
        plans = analyze_captions(captions)
        engine = LayoutEngine(
            VideoInfo(1080, 1920, 30.0, cursor + 1.0),
            StyleConfig(font_path=str(FONT_PATH), font_size=72),
            LayoutConfig(margin_x=108, margin_y=96),
        )
        states = engine.build_dynamic_states(plans)
        normalized = _normalise_timeline(states, cursor + 1.0)

        self.assertEqual(len(chunked), len(words))
        self.assertEqual(
            [(word.text, word.start, word.end) for word in chunked],
            [(word.text, word.start, word.end) for word in words],
        )
        self.assertEqual(sum(len(plan.caption.words) for plan in plans), len(words))
        self.assertEqual(len(states), len(words))
        self.assertEqual(len(normalized), len(words))
        self.assertEqual(states[0].start, words[0].start)
        self.assertEqual(states[-1].end, words[-1].end)
        self.assertEqual(normalized[-1].end, words[-1].end)


if __name__ == "__main__":
    unittest.main()
