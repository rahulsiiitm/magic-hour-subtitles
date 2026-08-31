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
from killer_subtitles.layout import LayoutEngine
from killer_subtitles.models import (
    Caption,
    LayoutConfig,
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

    def test_keywords_exclude_stopwords(self):
        caption = Caption(timed_words(["the", "amazing", "change", "is"]))
        frequencies = Counter(normalize_word(word.text) for word in caption.words)
        indices = select_keyword_indices(caption, frequencies)
        selected = {normalize_word(caption.words[index].text) for index in indices}
        self.assertFalse(selected & STOPWORDS)
        self.assertEqual(selected, {"amazing"})

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


if __name__ == "__main__":
    unittest.main()
