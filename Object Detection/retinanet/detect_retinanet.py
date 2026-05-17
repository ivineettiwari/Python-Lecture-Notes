"""RetinaNet object detection with OpenCV's DNN module (ONNX backend).

Required files in models/retinanet/:
    retinanet-9.onnx    https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/retinanet

How RetinaNet works (one-stage detector):
    - Backbone: ResNet-50/101.
    - Neck: Feature Pyramid Network (FPN) with levels P3..P7.
    - Two heads (class & box) shared across pyramid levels.
    - Key innovation: **Focal Loss**
          FL(p_t) = -α (1 - p_t)^γ log(p_t)
      Down-weights the loss of easy background anchors so the network
      focuses on hard examples — closes the accuracy gap with two-stage
      detectors while staying fast.

The ONNX model from the model zoo expects a 1x3x480x640 BGR image (no
mean subtraction, raw uint8→float). It produces three outputs:
    boxes   (N, 4)  in xyxy pixel coords
    labels  (N,)    int64 class ids (1-based, 80 classes)
    scores  (N,)    float
NMS has already been applied internally.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import COCO_80_CLASSES, draw_detections, put_fps  # noqa: E402

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "retinanet"

# RetinaNet ONNX model expects this exact input size.
INPUT_H, INPUT_W = 480, 640


class RetinaNetDetector:
    def __init__(
        self,
        weights: str | Path,
        conf_threshold: float = 0.5,
        use_cuda: bool = False,
    ) -> None:
        self.conf_threshold = conf_threshold

        self.net = cv2.dnn.readNetFromONNX(str(weights))
        if use_cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.class_names = COCO_80_CLASSES

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, float]:
        h, w = image.shape[:2]
        resized = cv2.resize(image, (INPUT_W, INPUT_H))
        # ImageNet-style mean subtraction (RGB order).
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - mean) / std
        # HWC → CHW → NCHW
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]
        scale_x = w / INPUT_W
        scale_y = h / INPUT_H
        return blob, scale_x, scale_y

    def detect(self, image: np.ndarray):
        blob, sx, sy = self._preprocess(image)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        # Output ordering depends on the export — find by shape.
        boxes_out, labels_out, scores_out = None, None, None
        for o in outputs:
            arr = np.asarray(o)
            if arr.ndim == 2 and arr.shape[1] == 4:
                boxes_out = arr
            elif arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
                labels_out = arr
            elif arr.ndim == 1 and np.issubdtype(arr.dtype, np.floating):
                scores_out = arr
        if boxes_out is None or labels_out is None or scores_out is None:
            empty = np.zeros((0, 4), dtype=np.float32)
            return empty, np.array([], dtype=np.float32), np.array([], dtype=np.int32)

        mask = scores_out >= self.conf_threshold
        boxes_out = boxes_out[mask]
        scores_out = scores_out[mask]
        labels_out = labels_out[mask]

        # Rescale boxes from 640x480 → original.
        boxes_out = boxes_out.astype(np.float32).copy()
        boxes_out[:, [0, 2]] *= sx
        boxes_out[:, [1, 3]] *= sy

        # ONNX-zoo model uses 1-based 80-class labels.
        class_ids = (labels_out.astype(np.int32) - 1).clip(0, len(self.class_names) - 1)

        return boxes_out, scores_out.astype(np.float32), class_ids


def _run(args: argparse.Namespace) -> None:
    detector = RetinaNetDetector(
        weights=args.weights,
        conf_threshold=args.conf,
        use_cuda=args.cuda,
    )

    source = args.source
    is_image = Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}

    if is_image:
        img = cv2.imread(source)
        if img is None:
            raise FileNotFoundError(source)
        t0 = time.time()
        boxes, scores, ids = detector.detect(img)
        dt = time.time() - t0
        annotated = draw_detections(img, boxes, scores, ids, detector.class_names)
        annotated = put_fps(annotated, 1.0 / max(dt, 1e-6), "RetinaNet")
        out_path = Path("outputs") / f"retinanet_{Path(source).name}"
        out_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"[RetinaNet] {len(boxes)} detections in {dt*1000:.1f} ms → {out_path}")
        cv2.imshow("RetinaNet", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        boxes, scores, ids = detector.detect(frame)
        fps = 1.0 / max(time.time() - t0, 1e-6)
        annotated = draw_detections(frame, boxes, scores, ids, detector.class_names)
        annotated = put_fps(annotated, fps, "RetinaNet")
        cv2.imshow("RetinaNet", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RetinaNet with OpenCV DNN (ONNX)")
    p.add_argument("--source", required=True)
    p.add_argument(
        "--weights",
        default=str(DEFAULT_MODEL_DIR / "retinanet-9.onnx"),
    )
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--cuda", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    _run(parse_args())
