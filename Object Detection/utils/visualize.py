"""Drawing helpers for detections."""
from __future__ import annotations

import colorsys

import cv2
import numpy as np


def _palette(n: int) -> list[tuple[int, int, int]]:
    """Generate n visually distinct BGR colors."""
    colors = []
    for i in range(n):
        h = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    return colors


_DEFAULT_PALETTE = _palette(80)


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_names: list[str],
    palette: list[tuple[int, int, int]] | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw xyxy boxes with class label and score onto a copy of the image."""
    out = image.copy()
    palette = palette or _DEFAULT_PALETTE

    for box, score, cid in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        cid = int(cid)
        color = palette[cid % len(palette)]

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        label = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        text = f"{label} {score:.2f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(y1 - 4, th + 4)
        cv2.rectangle(
            out,
            (x1, y_text - th - bl),
            (x1 + tw + 2, y_text),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            out,
            text,
            (x1 + 1, y_text - bl),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def put_fps(image: np.ndarray, fps: float, detector_name: str = "") -> np.ndarray:
    """Top-left HUD with FPS and detector name."""
    text = f"{detector_name}  {fps:5.1f} FPS" if detector_name else f"{fps:5.1f} FPS"
    cv2.putText(
        image,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return image
