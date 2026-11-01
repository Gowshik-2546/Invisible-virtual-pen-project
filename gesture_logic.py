"""Small, dependency-free helpers for interpreting hand landmarks."""

from __future__ import annotations

from typing import Protocol, Sequence


INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18


class LandmarkPoint(Protocol):
    y: float


def finger_is_up(
    landmarks: Sequence[LandmarkPoint], tip_index: int, pip_index: int
) -> bool:
    """Return whether a finger tip is above its PIP joint in image coordinates."""
    return landmarks[tip_index].y < landmarks[pip_index].y


def count_extended_fingers(landmarks: Sequence[LandmarkPoint]) -> int:
    """Count raised non-thumb fingers for drawing and brush-size selection."""
    return sum(
        finger_is_up(landmarks, tip_index, pip_index)
        for tip_index, pip_index in (
            (INDEX_TIP, INDEX_PIP),
            (MIDDLE_TIP, MIDDLE_PIP),
            (RING_TIP, RING_PIP),
            (PINKY_TIP, PINKY_PIP),
        )
    )


def is_erase_gesture(landmarks: Sequence[LandmarkPoint]) -> bool:
    """Erase when all four non-thumb fingers are raised in an open palm."""
    return count_extended_fingers(landmarks) == 4


def brush_size_for_finger_count(finger_count: int) -> int:
    """Return a progressively thicker brush for one through three raised fingers."""
    brush_sizes = (4, 8, 12)
    if finger_count < 1 or finger_count > len(brush_sizes):
        raise ValueError("finger_count must be between 1 and 4")
    return brush_sizes[finger_count - 1]


def is_closed_hand(landmarks: Sequence[LandmarkPoint]) -> bool:
    """Return whether all four non-thumb fingers are folded."""
    return all(
        not finger_is_up(landmarks, tip_index, pip_index)
        for tip_index, pip_index in (
            (INDEX_TIP, INDEX_PIP),
            (MIDDLE_TIP, MIDDLE_PIP),
            (RING_TIP, RING_PIP),
            (PINKY_TIP, PINKY_PIP),
        )
    )


def normalized_to_pixel(
    x: float, y: float, frame_width: int, frame_height: int
) -> tuple[int, int]:
    """Convert normalized landmark coordinates to bounded frame coordinates."""
    pixel_x = round(x * frame_width)
    pixel_y = round(y * frame_height)
    return (
        min(max(pixel_x, 0), frame_width - 1),
        min(max(pixel_y, 0), frame_height - 1),
    )
