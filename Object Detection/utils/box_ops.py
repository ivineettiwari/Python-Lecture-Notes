"""Bounding-box geometry: format conversions, IoU, and NMS.

All boxes are numpy arrays of shape (N, 4) unless noted otherwise.
The canonical internal format is xyxy = (x_min, y_min, x_max, y_max).
"""
from __future__ import annotations

import numpy as np


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """(x, y, w, h) → (x1, y1, x2, y2)."""
    boxes = np.asarray(boxes, dtype=np.float32)
    out = boxes.copy()
    out[..., 2] = boxes[..., 0] + boxes[..., 2]
    out[..., 3] = boxes[..., 1] + boxes[..., 3]
    return out


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """(cx, cy, w, h) → (x1, y1, x2, y2). YOLO's native output format."""
    boxes = np.asarray(boxes, dtype=np.float32)
    out = np.empty_like(boxes)
    out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2.0
    out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2.0
    out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2.0
    out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2.0
    return out


def clip_boxes(boxes: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """Clamp xyxy boxes inside [0, w] × [0, h]."""
    boxes = boxes.copy()
    boxes[..., 0::2] = np.clip(boxes[..., 0::2], 0, img_w - 1)
    boxes[..., 1::2] = np.clip(boxes[..., 1::2], 0, img_h - 1)
    return boxes


def iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU.

    Args:
        boxes_a: (N, 4) xyxy
        boxes_b: (M, 4) xyxy
    Returns:
        (N, M) IoU matrix.
    """
    a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)

    area_a = np.maximum(a[:, 2] - a[:, 0], 0) * np.maximum(a[:, 3] - a[:, 1], 0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)

    # Broadcast to (N, M, 4)
    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
) -> list[int]:
    """Pure-numpy Non-Maximum Suppression.

    Args:
        boxes: (N, 4) xyxy
        scores: (N,)
        iou_threshold: drop boxes whose IoU with a kept box exceeds this.
    Returns:
        Indices of boxes to keep, ordered by descending score.
    """
    if len(boxes) == 0:
        return []

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        ovr = inter / (areas[i] + areas[rest] - inter + 1e-9)

        order = rest[ovr <= iou_threshold]

    return keep
