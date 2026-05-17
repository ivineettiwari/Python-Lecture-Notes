"""Unified entry point — pick a detector and run it on any image / video / webcam.

Examples:
    python main.py --detector yolo         --source image.jpg
    python main.py --detector ssd          --source 0
    python main.py --detector faster_rcnn  --source 00225f53-67614580.mov
    python main.py --detector retinanet    --source image.jpg --conf 0.4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from faster_rcnn.detect_faster_rcnn import FasterRCNNDetector
from retinanet.detect_retinanet import RetinaNetDetector
from ssd.detect_ssd import SSDDetector
from utils import draw_detections, put_fps
from yolo.detect_yolo import YoloDetector

MODELS = Path(__file__).resolve().parent / "models"


def build_detector(name: str, conf: float, use_cuda: bool):
    if name == "yolo":
        d = MODELS / "yolo"
        return YoloDetector(
            weights=d / "yolov3.weights",
            config=d / "yolov3.cfg",
            names=d / "coco.names",
            conf_threshold=conf,
            use_cuda=use_cuda,
        )
    if name == "faster_rcnn":
        d = MODELS / "faster_rcnn"
        return FasterRCNNDetector(
            weights=d / "frozen_inference_graph.pb",
            config=d / "faster_rcnn_inception_v2_coco_2018_01_28.pbtxt",
            conf_threshold=conf,
            use_cuda=use_cuda,
        )
    if name == "ssd":
        d = MODELS / "ssd"
        return SSDDetector(
            weights=d / "frozen_inference_graph.pb",
            config=d / "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt",
            conf_threshold=conf,
            use_cuda=use_cuda,
        )
    if name == "retinanet":
        d = MODELS / "retinanet"
        return RetinaNetDetector(
            weights=d / "retinanet-9.onnx",
            conf_threshold=conf,
            use_cuda=use_cuda,
        )
    raise ValueError(f"Unknown detector: {name}")


def run_image(detector, label: str, source: str) -> None:
    img = cv2.imread(source)
    if img is None:
        raise FileNotFoundError(source)
    t0 = time.time()
    boxes, scores, ids = detector.detect(img)
    dt = time.time() - t0
    annotated = draw_detections(img, boxes, scores, ids, detector.class_names)
    annotated = put_fps(annotated, 1.0 / max(dt, 1e-6), label)
    out_path = Path("outputs") / f"{label.lower()}_{Path(source).name}"
    out_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    print(f"[{label}] {len(boxes)} detections in {dt*1000:.1f} ms → {out_path}")
    cv2.imshow(label, annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_stream(detector, label: str, source: str) -> None:
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
        annotated = put_fps(annotated, fps, label)
        cv2.imshow(label, annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(description="OpenCV Object Detection — unified runner")
    p.add_argument(
        "--detector",
        required=True,
        choices=["yolo", "faster_rcnn", "ssd", "retinanet"],
    )
    p.add_argument("--source", required=True, help="image / video path or webcam index")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--cuda", action="store_true", help="Use CUDA backend if built")
    args = p.parse_args()

    detector = build_detector(args.detector, args.conf, args.cuda)
    label = {
        "yolo": "YOLO",
        "faster_rcnn": "Faster R-CNN",
        "ssd": "SSD",
        "retinanet": "RetinaNet",
    }[args.detector]

    is_image = Path(args.source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    if is_image:
        run_image(detector, label, args.source)
    else:
        run_stream(detector, label, args.source)


if __name__ == "__main__":
    main()
