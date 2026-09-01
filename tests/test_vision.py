from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from magic_hour_subtitles.models import VisionConfig
from magic_hour_subtitles.vision import VisionAnalyzer, _combined_foreground_masks


class FakeTensor:
    def __init__(self, values) -> None:
        self.values = np.asarray(values)

    def __len__(self) -> int:
        return len(self.values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def fake_result(masks, classes, confidences):
    return SimpleNamespace(
        masks=SimpleNamespace(data=FakeTensor(masks)),
        boxes=SimpleNamespace(
            cls=FakeTensor(classes),
            conf=FakeTensor(confidences),
        ),
    )


class ForegroundMaskTests(unittest.TestCase):
    def test_large_whitelisted_object_joins_foreground_but_not_person_map(self):
        masks = np.zeros((3, 10, 10), dtype=np.float32)
        masks[0, :3, :3] = 1.0       # person
        masks[1, 4:8, 4:8] = 1.0    # car
        masks[2, 7:10, :3] = 1.0    # dog (not whitelisted)

        person, foreground, confidence, kind = _combined_foreground_masks(
            fake_result(masks, [0, 2, 16], [0.9, 0.8, 0.95]),
            10,
            10,
            0,
            set(VisionConfig().foreground_class_ids),
            0.01,
            cv2,
            np,
        )

        self.assertEqual(person[1, 1], 255)
        self.assertEqual(person[5, 5], 0)
        self.assertEqual(foreground[1, 1], 255)
        self.assertEqual(foreground[5, 5], 255)
        self.assertEqual(foreground[8, 1], 0)
        self.assertAlmostEqual(confidence, 0.9)
        self.assertEqual(kind, "mixed")

    def test_tiny_whitelisted_object_is_ignored(self):
        mask = np.zeros((1, 10, 10), dtype=np.float32)
        mask[0, 0, 0] = 1.0
        person, foreground, _confidence, kind = _combined_foreground_masks(
            fake_result(mask, [56], [0.9]),
            10,
            10,
            0,
            set(VisionConfig().foreground_class_ids),
            0.02,
            cv2,
            np,
        )
        self.assertFalse(np.any(person))
        self.assertFalse(np.any(foreground))
        self.assertEqual(kind, "none")

    def test_non_whitelisted_class_is_ignored(self):
        mask = np.ones((1, 10, 10), dtype=np.float32)
        _person, foreground, _confidence, kind = _combined_foreground_masks(
            fake_result(mask, [16], [0.9]),
            10,
            10,
            0,
            set(VisionConfig().foreground_class_ids),
            0.01,
            cv2,
            np,
        )
        self.assertFalse(np.any(foreground))
        self.assertEqual(kind, "none")

    def test_vision_analyzer_contains_one_model_prediction_call(self):
        source = inspect.getsource(VisionAnalyzer.analyze)
        self.assertEqual(source.count("model.predict("), 1)

    def test_foreground_whitelist_is_conservative(self):
        self.assertEqual(
            VisionConfig().foreground_class_ids,
            (0, 1, 2, 3, 5, 7, 56),
        )


if __name__ == "__main__":
    unittest.main()
