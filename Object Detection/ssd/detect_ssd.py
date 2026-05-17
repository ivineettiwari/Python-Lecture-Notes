"""SSD MobileNet object detection with OpenCV's DNN module.

Required files in models/ssd/:
    frozen_inference_graph.pb
    ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt   (or v2)

Get them from:
    https://github.com/opencv/opencv_extra/tree/master/testdata/dnn

How SSD works (one-stage detector):
    1. Backbone (e.g. MobileNet/VGG) → multi-scale feature maps.
    2. Each feature map cell is tiled with K "default boxes" of various
       aspect ratios. Smaller maps detect larger objects, larger maps detect
       smaller objects.
    3. Two parallel 3x3 convs per scale predict (a) box offsets per default
       box, (b) class scores per default box.
    4. Filter by confidence + per-class NMS.

OpenCV's TF-SSD output is identical to Faster R-CNN's:
    (1, 1, N, 7) → [batch, class_id, score, x1, y1, x2, y2] (normalised)
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

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "ssd"


class SSDDetector:
    def __init__(
        self,
        weights: str | Path,
        config: str | Path,
        conf_threshold: float = 0.5,
        input_size: int = 320,
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
        blob = cv2.dnn.blobFromImage(
            image,
            size=(self.input_size, self.input_size),
            mean=(127.5, 127.5, 127.5),
            scalefactor=1 / 127.5,  # MobileNet normalisation: pixels → [-1, 1]
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
            tf_class = int(det[1])
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
    detector = SSDDetector(
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
        annotated = put_fps(annotated, 1.0 / max(dt, 1e-6), "SSD")
        out_path = Path("outputs") / f"ssd_{Path(source).name}"
        out_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"[SSD] {len(boxes)} detections in {dt*1000:.1f} ms → {out_path}")
        cv2.imshow("SSD", annotated)
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
        annotated = put_fps(annotated, fps, "SSD")
        cv2.imshow("SSD", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SSD MobileNet with OpenCV DNN")
    p.add_argument("--source", required=True)
    p.add_argument(
        "--weights",
        default=str(DEFAULT_MODEL_DIR / "frozen_inference_graph.pb"),
    )
    p.add_argument(
        "--config",
        default=str(
            DEFAULT_MODEL_DIR / "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
        ),
    )
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--cuda", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    _run(parse_args())
