"""Draw in the air with a webcam and MediaPipe hand tracking."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from gesture_logic import (
    brush_size_for_finger_count,
    count_extended_fingers,
    is_closed_hand,
    is_erase_gesture,
    normalized_to_pixel,
)


WINDOW_NAME = "Invisible Virtual Pen"
DRAWINGS_DIR = Path("drawings")
COLORS = {
    "1": ("White", (255, 255, 255)),
    "2": ("Red", (0, 0, 255)),
    "3": ("Green", (0, 255, 0)),
    "4": ("Blue", (255, 0, 0)),
    "5": ("Yellow", (0, 255, 255)),
}
ERASER_SIZE = 36


def draw_hud(
    frame: np.ndarray,
    selected_color: str,
    erasing: bool,
    keyboard_eraser_enabled: bool,
    finger_count: int,
    brush_size: int,
) -> None:
    """Render the on-screen controls and selected brush state."""
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 112), (30, 30, 30), -1)
    cv2.putText(
        frame,
        "1-3 fingers: draw thicker | Open palm: erase | Fist: clear",
        (16, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    brush_label = (
        f"ERASER {ERASER_SIZE}px"
        if erasing
        else f"{finger_count} fingers / {brush_size}px"
    )
    cv2.putText(
        frame,
        f"Brush: {brush_label} | E-key eraser: "
        f"{'ON' if keyboard_eraser_enabled else 'OFF'}",
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "1-5 color | C clear | S save | Q quit",
        (16, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    if not erasing:
        cv2.circle(
            frame,
            (frame.shape[1] - 30, 30),
            12,
            COLORS[selected_color][1],
            -1,
            cv2.LINE_AA,
        )


def save_drawing(canvas: np.ndarray) -> Path:
    """Save the transparent drawing layer and return its path."""
    DRAWINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = DRAWINGS_DIR / f"drawing_{timestamp}.png"
    output = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
    output[:, :, 3] = np.where(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0, 255, 0)
    if not cv2.imwrite(str(output_path), output):
        raise OSError(f"Could not save drawing to {output_path}")
    return output_path


def run(camera_index: int = 0) -> None:
    """Open the webcam and run the virtual-pen interaction loop."""
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera {camera_index}. Check that a webcam is connected "
            "and available."
        )

    hand_api = mp.solutions.hands
    drawing_api = mp.solutions.drawing_utils
    canvas: np.ndarray | None = None
    previous_point: tuple[int, int] | None = None
    selected_color = "1"
    erasing = False
    closed_hand_active = False

    try:
        with hand_api.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as hands:
            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("The webcam stopped returning frames.")

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                if canvas is None or canvas.shape[:2] != (height, width):
                    canvas = np.zeros((height, width, 3), dtype=np.uint8)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = hands.process(rgb_frame)
                rgb_frame.flags.writeable = True

                draw_active = False
                finger_count = 0
                brush_size = 0
                erase_gesture_active = False
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmarks = hand_landmarks.landmark
                    drawing_api.draw_landmarks(
                        frame, hand_landmarks, hand_api.HAND_CONNECTIONS
                    )

                    fist_detected = is_closed_hand(landmarks)
                    if fist_detected:
                        if not closed_hand_active:
                            canvas.fill(0)
                        previous_point = None
                    else:
                        finger_count = count_extended_fingers(landmarks)
                        erase_gesture_active = is_erase_gesture(landmarks)
                        index_tip = landmarks[8]
                        point = normalized_to_pixel(
                            index_tip.x, index_tip.y, width, height
                        )
                        draw_active = finger_count > 0

                        if draw_active:
                            use_eraser = erasing or erase_gesture_active
                            brush_size = (
                                ERASER_SIZE
                                if use_eraser
                                else brush_size_for_finger_count(finger_count)
                            )
                            brush_color = (
                                (0, 0, 0)
                                if use_eraser
                                else COLORS[selected_color][1]
                            )
                            if previous_point is not None:
                                cv2.line(
                                    canvas,
                                    previous_point,
                                    point,
                                    brush_color,
                                    brush_size,
                                    cv2.LINE_8 if use_eraser else cv2.LINE_AA,
                                )
                            cv2.circle(
                                canvas,
                                point,
                                brush_size // 2,
                                brush_color,
                                -1,
                                cv2.LINE_8 if use_eraser else cv2.LINE_AA,
                            )
                            cv2.circle(
                                frame,
                                point,
                                brush_size // 2,
                                (255, 255, 255) if use_eraser else brush_color,
                                2,
                                cv2.LINE_AA,
                            )
                            previous_point = point
                        else:
                            previous_point = None
                    closed_hand_active = fist_detected
                else:
                    previous_point = None
                    closed_hand_active = False

                assert canvas is not None
                visible_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0
                frame[visible_canvas] = canvas[visible_canvas]
                draw_hud(
                    frame,
                    selected_color,
                    erasing or erase_gesture_active,
                    erasing,
                    finger_count,
                    brush_size,
                )
                if not draw_active:
                    cv2.putText(
                        frame,
                        "1-3 fingers: draw thicker | Open palm: erase | Fist: clear",
                        (16, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(
                    WINDOW_NAME, cv2.WND_PROP_VISIBLE
                ) < 1:
                    break
                if chr(key) in COLORS:
                    selected_color = chr(key)
                    erasing = False
                elif key == ord("e"):
                    erasing = not erasing
                elif key == ord("c"):
                    canvas.fill(0)
                    previous_point = None
                elif key == ord("s"):
                    print(f"Saved drawing to {save_drawing(canvas)}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
