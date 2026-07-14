"""
MediCare AI - OpenCV / YOLO-style Interface
=============================================
Replaces the Gradio web UI with a native OpenCV window. Results are rendered
directly on-screen the way YOLO detection demos overlay boxes/labels on a
live cv2.imshow feed - a colored border + a side console panel with the
prediction, confidence, a hand-drawn probability bar chart, and the
clinical description.

Controls (focus the OpenCV window first):
  1 / 2 / 3   - switch image type (MRI / Retina / Fracture)
  S           - load the built-in sample image for the current type
  O           - open an image file (native file dialog)
  W           - toggle webcam mode (live feed)
  SPACE       - capture + predict current webcam frame
  Q / ESC     - quit
"""

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from prediction.predict_mri import predict_mri
from prediction.predict_retina import predict_retina
from prediction.predict_fracture import predict_fracture

ASSETS_DIR = PROJECT_ROOT / "assets"

IMAGE_TYPES = {
    "1": {
        "name": "Brain Tumor MRI",
        "predict_fn": predict_mri,
        "sample": ASSETS_DIR / "brain_mri.jpg",
        "color": (168, 59, 46),   # BGR accent, echoes the old #2E86AB
    },
    "2": {
        "name": "Diabetic Retina Screening",
        "predict_fn": predict_retina,
        "sample": ASSETS_DIR / "retina.jpeg",
        "color": (46, 125, 46),
    },
    "3": {
        "name": "Bone Fracture Detection",
        "predict_fn": predict_fracture,
        "sample": ASSETS_DIR / "fracturexray.png",
        "color": (43, 57, 192),
    },
}

WINDOW_NAME = "MediCare AI"
PANEL_WIDTH = 340
DISPLAY_H = 640


def pick_file() -> str:
    """Native OS file dialog (same approach your CustomTkinter app uses)."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select medical image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
    )
    root.destroy()
    return path


def cv2_to_pil(frame_bgr) -> Image.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def wrap_text(text: str, max_chars: int = 42):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_bars(panel, probabilities: dict, predicted_class: str, accent_bgr, origin_y: int = 170) -> int:
    """Hand-drawn horizontal bar chart using raw cv2 primitives (no matplotlib)."""
    x0 = 20
    bar_max_w = PANEL_WIDTH - 110
    bar_h = 22
    gap = 14
    y = origin_y

    cv2.putText(panel, "PROBABILITIES", (x0, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    for cls, val in probabilities.items():
        w = int(bar_max_w * (val / 100.0))
        color = accent_bgr if cls == predicted_class else (90, 90, 90)
        cv2.rectangle(panel, (x0, y), (x0 + bar_max_w, y + bar_h), (50, 50, 50), -1)
        cv2.rectangle(panel, (x0, y), (x0 + max(w, 2), y + bar_h), color, -1)
        cv2.putText(panel, cls, (x0, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{val:.1f}%", (x0 + bar_max_w + 8, y + bar_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        y += bar_h + gap

    return y


def build_panel(image_type_key: str, result: Optional[dict], status: str) -> np.ndarray:
    info = IMAGE_TYPES[image_type_key]
    panel = np.full((DISPLAY_H, PANEL_WIDTH, 3), 26, dtype=np.uint8)

    cv2.putText(panel, "MediCare AI", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.line(panel, (20, 48), (PANEL_WIDTH - 20, 48), (70, 70, 70), 1)

    cv2.putText(panel, info["name"], (20, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 190, 255), 1, cv2.LINE_AA)

    if result is None:
        for i, line in enumerate(wrap_text(status, 40)):
            cv2.putText(panel, line, (20, 110 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
    else:
        label = result["predicted_class"]
        conf = result["confidence"]
        cv2.putText(panel, label.upper(), (20, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, info["color"], 2, cv2.LINE_AA)
        cv2.putText(panel, f"Confidence: {conf:.2f}%", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        y_after_bars = draw_bars(panel, result["probabilities"], label, info["color"])

        cv2.line(panel, (20, y_after_bars + 5), (PANEL_WIDTH - 20, y_after_bars + 5), (70, 70, 70), 1)
        cv2.putText(panel, "DESCRIPTION", (20, y_after_bars + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        yy = y_after_bars + 55
        for line in wrap_text(result["description"])[:10]:
            cv2.putText(panel, line, (20, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
            yy += 20

    footer_y = DISPLAY_H - 90
    cv2.line(panel, (20, footer_y), (PANEL_WIDTH - 20, footer_y), (70, 70, 70), 1)
    controls = ["[1/2/3] Type", "[O] Open   [S] Sample", "[W] Webcam  [SPACE] Capture", "[Q] Quit"]
    yy = footer_y + 22
    for c in controls:
        cv2.putText(panel, c, (20, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)
        yy += 20

    return panel


def fit_image(img_bgr, target_h: int):
    h, w = img_bgr.shape[:2]
    scale = target_h / h
    return cv2.resize(img_bgr, (int(w * scale), target_h))


def run_prediction(image_type_key: str, img_bgr):
    info = IMAGE_TYPES[image_type_key]
    pil_img = cv2_to_pil(img_bgr)
    try:
        return info["predict_fn"](pil_img), None
    except FileNotFoundError as exc:
        return None, f"Model not found: {exc}"
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return None, f"Error: {exc}"


def compose_frame(img_bgr, image_type_key: str, result: Optional[dict], status: str):
    display_img = fit_image(img_bgr, DISPLAY_H)
    panel = build_panel(image_type_key, result, status)

    if result is not None:
        info = IMAGE_TYPES[image_type_key]
        cv2.rectangle(display_img, (0, 0),
                       (display_img.shape[1] - 1, display_img.shape[0] - 1),
                       info["color"], 4)

    canvas = np.zeros((DISPLAY_H, display_img.shape[1] + PANEL_WIDTH, 3), dtype=np.uint8)
    canvas[:, :display_img.shape[1]] = display_img
    canvas[:, display_img.shape[1]:] = panel
    return canvas


def main():
    image_type_key = "1"
    current_img_bgr: Optional[np.ndarray] = None
    result: Optional[dict] = None
    status = "Press [S] to load sample or [O] to open an image."
    webcam_mode = False
    cap = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    while True:
        if webcam_mode and cap is not None:
            ok, frame = cap.read()
            if ok:
                current_img_bgr = frame
            frame_out = compose_frame(current_img_bgr, image_type_key, result,
                                       "Webcam live - press [SPACE] to predict this frame")
        elif current_img_bgr is not None:
            frame_out = compose_frame(current_img_bgr, image_type_key, result, status)
        else:
            blank = np.full((DISPLAY_H, 480, 3), 15, dtype=np.uint8)
            frame_out = compose_frame(blank, image_type_key, None, status)

        cv2.imshow(WINDOW_NAME, frame_out)
        key = cv2.waitKey(1 if webcam_mode else 20) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key in (ord("1"), ord("2"), ord("3")):
            image_type_key = chr(key)
            result = None
            status = f"Switched to {IMAGE_TYPES[image_type_key]['name']}."
        elif key == ord("s"):
            sample_path = IMAGE_TYPES[image_type_key]["sample"]
            if sample_path.exists():
                current_img_bgr = cv2.imread(str(sample_path))
                result, err = run_prediction(image_type_key, current_img_bgr)
                status = err or "Sample loaded."
            else:
                status = f"Sample not found: {sample_path}"
        elif key == ord("o"):
            path = pick_file()
            if path:
                img = cv2.imread(path)
                if img is None:
                    status = "Could not read image."
                else:
                    current_img_bgr = img
                    result, err = run_prediction(image_type_key, current_img_bgr)
                    status = err or f"Loaded {Path(path).name}."
        elif key == ord("w"):
            webcam_mode = not webcam_mode
            if webcam_mode:
                cap = cv2.VideoCapture(0)
                result = None
                status = "Webcam started."
            else:
                if cap is not None:
                    cap.release()
                    cap = None
                status = "Webcam stopped."
        elif key == ord(" ") and webcam_mode and current_img_bgr is not None:
            result, err = run_prediction(image_type_key, current_img_bgr)
            status = err or "Frame captured & analyzed."

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()