import unittest
from types import SimpleNamespace

from gesture_logic import (
    brush_size_for_finger_count,
    count_extended_fingers,
    is_closed_hand,
    is_erase_gesture,
    normalized_to_pixel,
)


def make_landmarks(
    index_up: bool,
    middle_up: bool,
    ring_up: bool = False,
    pinky_up: bool = False,
) -> list[SimpleNamespace]:
    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
    for tip, pip, extended in (
        (8, 6, index_up),
        (12, 10, middle_up),
        (16, 14, ring_up),
        (20, 18, pinky_up),
    ):
        landmarks[pip].y = 0.5
        landmarks[tip].y = 0.3 if extended else 0.7
    return landmarks


class GestureLogicTests(unittest.TestCase):
    def test_counting_raised_fingers(self) -> None:
        for count in range(5):
            landmarks = make_landmarks(
                index_up=count >= 1,
                middle_up=count >= 2,
                ring_up=count >= 3,
                pinky_up=count >= 4,
            )
            self.assertEqual(count_extended_fingers(landmarks), count)

    def test_brush_gets_thicker_for_one_to_three_raised_fingers(self) -> None:
        sizes = [brush_size_for_finger_count(count) for count in range(1, 4)]
        self.assertEqual(sizes, [4, 8, 12])
        self.assertEqual(sizes, sorted(sizes))

    def test_brush_size_rejects_invalid_finger_count(self) -> None:
        for count in (0, 4):
            with self.subTest(count=count), self.assertRaises(ValueError):
                brush_size_for_finger_count(count)

    def test_closed_hand_is_clear_gesture(self) -> None:
        self.assertTrue(is_closed_hand(make_landmarks(index_up=False, middle_up=False)))

    def test_open_hand_triggers_erase_not_clear(self) -> None:
        landmarks = make_landmarks(
            index_up=True, middle_up=True, ring_up=True, pinky_up=True
        )
        self.assertEqual(count_extended_fingers(landmarks), 4)
        self.assertTrue(is_erase_gesture(landmarks))
        self.assertFalse(is_closed_hand(landmarks))

    def test_partial_hand_does_not_trigger_erase(self) -> None:
        landmarks = make_landmarks(
            index_up=True, middle_up=True, ring_up=True, pinky_up=False
        )
        self.assertFalse(is_erase_gesture(landmarks))

    def test_normalized_coordinates_are_bounded(self) -> None:
        self.assertEqual(normalized_to_pixel(0.5, 0.25, 640, 480), (320, 120))
        self.assertEqual(normalized_to_pixel(2.0, -1.0, 640, 480), (639, 0))


if __name__ == "__main__":
    unittest.main()
