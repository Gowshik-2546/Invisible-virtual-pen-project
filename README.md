# Invisible Virtual Pen

Draw in the air using a webcam. The app uses MediaPipe's pretrained hand-landmark
model to find your index fingertip; it does not train a new model or need a custom
dataset.

## Setup

Use Python 3.10, 3.11, or 3.12 (validated here on Python 3.12), then run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```powershell
python virtual_pen.py
```

Allow webcam access when prompted. The mirrored camera preview makes movement feel
natural. Hold one hand in view and use these gestures:

- **Raise one, two, or three fingers:** draw, with each additional raised finger
  selecting a thicker brush.
- **Open your hand (all four fingers up):** erase as you move.
- **Close your hand into a fist:** clear the canvas once. Open your hand before
  making another fist to clear again.

The MediaPipe dependency is capped below 0.10.22 because this app uses the
MediaPipe Solutions Hands API.

| Key | Action |
| --- | --- |
| `1`-`5` | Select white, red, green, blue, or yellow |
| `E` | Toggle keyboard eraser (as an alternative to the open-palm gesture) |
| `C` | Clear the canvas |
| `S` | Save the drawing layer as a PNG in `drawings/` |
| `Q` | Quit |

Saved PNGs contain a transparent background, so they can be composited over other
images. If the webcam cannot be opened, close apps that may already be using it
and try again.

## Test

The gesture rules can be tested without installing the computer-vision packages:

```powershell
python -m unittest discover -s tests -v
```

## How the ML fits in

MediaPipe performs the machine-learning inference: it estimates 21 hand landmarks
per frame. This app applies geometric rules to those landmarks to recognize drawing,
erasing, and clearing gestures. A future extension could train a gesture classifier
from labeled landmark sequences for additional gestures such as pause, undo, or
changing brush size.
