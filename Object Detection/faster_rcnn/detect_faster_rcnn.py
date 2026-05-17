"""Faster R-CNN object detection with OpenCV's DNN module.

Required files in models/faster_rcnn/:
    frozen_inference_graph.pb
    faster_rcnn_inception_v2_coco_2018_01_28.pbtxt

Get them from the OpenCV wiki:
    https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API

How Faster R-CNN works (two-stage detector):
    1. Backbone (Inception/ResNet) extracts a feature map.
    2. Region Proposal Network (RPN) slides over the feature map and emits
       ~2000 candidate boxes ranked by objectness.
    3. ROI Pooling crops each proposal to a fixed 7x7 feature.
    4. Classification head outputs class probabilities + box refinement.
    5. NMS gives final detections.

OpenCV's TF model output format: (1, 1, N, 7) where each row is
    [batch_id, class_id (1-based), confidence, x1, y1, x2, y2]
with x1..y2 normalised to [0, 1].
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import COCO_80_CLASSES, COCO_91_TO_80, draw_detections, put_fps  # noqa: E402

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "faster_rcnn"


class FasterRCNNDetector:
    def __init__(
        self,
        weights: str | Path,
        config: str | Path,
        conf_threshold: float = 0.5,
        input_size: int = 800,
        use_cuda: bool = False,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.input_size = input_size

        self.net = cv2.dnn.readNetFromTensorflow(str(weights), str(config))
        if use_cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.class_names = COCO_80_CLASSES

    def detect(self, image: np.ndarray):
        h, w = image.shape[:2]
        # TF object-detection models expect raw BGR uint8, no scaling.
        blob = cv2.dnn.blobFromImage(
            image,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        detections = self.net.forward()  # (1, 1, N, 7)

        boxes: list[list[float]] = []
        scores: list[float] = []
        class_ids: list[int] = []

        for det in detections[0, 0]:
            score = float(det[2])
            if score < self.conf_threshold:
                continue
            tf_class = int(det[1])  # 1-based, 91-style
            cid = COCO_91_TO_80.get(tf_class)
            if cid is None:
                continue
            x1 = float(det[3]) * w
            y1 = float(det[4]) * h
            x2 = float(det[5]) * w
            y2 = float(det[6]) * h
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            class_ids.append(cid)

        if not boxes:
            empty = np.zeros((0, 4), dtype=np.float32)
            return empty, np.array([], dtype=np.float32), np.array([], dtype=np.int32)
        return (
            np.array(boxes, dtype=np.float32),
            np.array(scores, dtype=np.float32),
            np.array(class_ids, dtype=np.int32),
        )


def _run(args: argparse.Namespace) -> None:
    detector = FasterRCNNDetector(
        weights=args.weights,
        config=args.config,
        conf_threshold=args.conf,
        input_size=args.size,
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
        annotated = put_fps(annotated, 1.0 / max(dt, 1e-6), "Faster R-CNN")
        out_path = Path("outputs") / f"faster_rcnn_{Path(source).name}"
        out_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"[Faster R-CNN] {len(boxes)} detections in {dt*1000:.1f} ms → {out_path}")
        cv2.imshow("Faster R-CNN", annotated)
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
        annotated = put_fps(annotated, fps, "Faster R-CNN")
        cv2.imshow("Faster R-CNN", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Faster R-CNN with OpenCV DNN")
    p.add_argument("--source", required=True)
    p.add_argument(
        "--weights",
        default=str(DEFAULT_MODEL_DIR / "frozen_inference_graph.pb"),
    )
    p.add_argument(
        "--config",
        default=str(
            DEFAULT_MODEL_DIR / "faster_rcnn_inception_v2_coco_2018_01_28.pbtxt"
        ),
    )
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--size", type=int, default=800)
    p.add_argument("--cuda", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    _run(parse_args())
