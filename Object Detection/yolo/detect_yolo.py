"""YOLOv3 / YOLOv4 object detection with OpenCV's DNN module.

Required files in models/yolo/:
    yolov3.weights              https://pjreddie.com/media/files/yolov3.weights
    yolov3.cfg                  https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
    coco.names                  https://github.com/pjreddie/darknet/blob/master/data/coco.names

For YOLOv4, swap to yolov4.weights / yolov4.cfg from AlexeyAB/darknet.

How YOLO works (one-stage detector):
    1. Image → 416x416 blob (BGR→RGB, scale 1/255).
    2. Single forward pass through Darknet backbone + FPN-like neck.
    3. Three output heads predict boxes at 3 scales (13x13, 26x26, 52x52).
       Each grid cell predicts 3 anchors → (cx, cy, w, h, objectness, 80 classes).
    4. Filter by confidence, run NMS.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import draw_detections, put_fps  # noqa: E402

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "yolo"


class YoloDetector:
    def __init__(
        self,
        weights: str | Path,
        config: str | Path,
        names: str | Path,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: int = 416,
        use_cuda: bool = False,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        self.net = cv2.dnn.readNetFromDarknet(str(config), str(weights))
        if use_cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        with open(names, "r") as f:
            self.class_names = [line.strip() for line in f if line.strip()]

        layer_names = self.net.getLayerNames()
        out_idx = self.net.getUnconnectedOutLayers().flatten()
        self.output_layers = [layer_names[i - 1] for i in out_idx]

    def detect(self, image: np.ndarray):
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1 / 255.0,
            size=(self.input_size, self.input_size),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes_xywh: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []

        for output in outputs:
            # output shape: (num_boxes, 5 + num_classes)
            scores_all = output[:, 5:]
            class_id = scores_all.argmax(axis=1)
            confidence = scores_all[np.arange(len(scores_all)), class_id]
            mask = confidence > self.conf_threshold
            if not mask.any():
                continue

            preds = output[mask]
            class_id = class_id[mask]
            confidence = confidence[mask]

            cx = preds[:, 0] * w
            cy = preds[:, 1] * h
            bw = preds[:, 2] * w
            bh = preds[:, 3] * h
            x = (cx - bw / 2).astype(int)
            y = (cy - bh / 2).astype(int)

            for i in range(len(preds)):
                boxes_xywh.append([int(x[i]), int(y[i]), int(bw[i]), int(bh[i])])
                confidences.append(float(confidence[i]))
                class_ids.append(int(class_id[i]))

        if not boxes_xywh:
            empty = np.zeros((0, 4), dtype=np.float32)
            return empty, np.array([], dtype=np.float32), np.array([], dtype=np.int32)

        keep = cv2.dnn.NMSBoxes(
            boxes_xywh, confidences, self.conf_threshold, self.nms_threshold
        )
        if len(keep) == 0:
            empty = np.zeros((0, 4), dtype=np.float32)
            return empty, np.array([], dtype=np.float32), np.array([], dtype=np.int32)
        keep = np.array(keep).flatten()

        out_boxes = []
        for i in keep:
            x, y, bw, bh = boxes_xywh[i]
            out_boxes.append([x, y, x + bw, y + bh])
        out_boxes = np.array(out_boxes, dtype=np.float32)
        out_scores = np.array([confidences[i] for i in keep], dtype=np.float32)
        out_ids = np.array([class_ids[i] for i in keep], dtype=np.int32)
        return out_boxes, out_scores, out_ids


def _run(args: argparse.Namespace) -> None:
    detector = YoloDetector(
        weights=args.weights,
        config=args.config,
        names=args.names,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
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
        annotated = put_fps(annotated, 1.0 / max(dt, 1e-6), "YOLO")
        out_path = Path("outputs") / f"yolo_{Path(source).name}"
        out_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"[YOLO] {len(boxes)} detections in {dt*1000:.1f} ms → {out_path}")
        cv2.imshow("YOLO", annotated)
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
        annotated = put_fps(annotated, fps, "YOLO")
        cv2.imshow("YOLO", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO detection with OpenCV DNN")
    p.add_argument("--source", required=True, help="image / video path or webcam index")
    p.add_argument("--weights", default=str(DEFAULT_MODEL_DIR / "yolov3.weights"))
    p.add_argument("--config", default=str(DEFAULT_MODEL_DIR / "yolov3.cfg"))
    p.add_argument("--names", default=str(DEFAULT_MODEL_DIR / "coco.names"))
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--nms", type=float, default=0.4)
    p.add_argument("--size", type=int, default=416, choices=[320, 416, 608])
    p.add_argument("--cuda", action="store_true", help="Use CUDA backend if built")
    return p.parse_args()


if __name__ == "__main__":
    _run(parse_args())
