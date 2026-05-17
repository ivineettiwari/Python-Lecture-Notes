# Object Detection with OpenCV

A learning-focused object detection project that demonstrates four classic detector families using OpenCV's DNN module:

| Detector      | Family         | Speed     | Accuracy  | Notes                              |
|---------------|----------------|-----------|-----------|------------------------------------|
| YOLOv3 / v4   | One-stage      | Very fast | Good      | Single forward pass, real-time     |
| Faster R-CNN  | Two-stage      | Slow      | Very high | Region Proposal Network + ROI head |
| SSD           | One-stage      | Fast      | Good      | Multi-scale feature maps           |
| RetinaNet     | One-stage      | Medium    | High      | Focal loss handles class imbalance |

## Project layout

```
Object Detection/
├── docs/
│   └── terminology.md          # IoU, bounding box, NMS, mAP, anchors, etc.
├── utils/
│   ├── __init__.py
│   ├── box_ops.py              # IoU, NMS, format conversions
│   ├── visualize.py            # Draw boxes, labels, FPS overlays
│   └── coco_labels.py          # 80 COCO class names
├── yolo/
│   └── detect_yolo.py          # YOLOv3 / YOLOv4 with OpenCV DNN
├── faster_rcnn/
│   └── detect_faster_rcnn.py   # Faster R-CNN (TensorFlow frozen graph)
├── ssd/
│   └── detect_ssd.py           # MobileNet-SSD
├── retinanet/
│   └── detect_retinanet.py     # RetinaNet (ONNX)
├── models/                     # Place downloaded weights/configs here
├── outputs/                    # Annotated images / videos go here
├── main.py                     # Unified CLI: pick detector + input
├── requirements.txt
└── README.md
```

## Quick start

```powershell
pip install -r requirements.txt
python main.py --detector yolo --source image.jpg
python main.py --detector ssd  --source 0          # webcam
python main.py --detector faster_rcnn --source 00225f53-67614580.mov
```

Each detector script is also runnable standalone:

```powershell
python yolo/detect_yolo.py --source image.jpg --conf 0.5
```

## Where to get the weights

The repo ships only the code. Each detector's script header lists the exact files
to download and where to drop them inside [models/](models/). Common sources:

- **YOLOv3:** `yolov3.weights`, `yolov3.cfg`, `coco.names` from pjreddie.com
- **YOLOv4:** `yolov4.weights`, `yolov4.cfg` from AlexeyAB/darknet
- **Faster R-CNN:** `frozen_inference_graph.pb` + `.pbtxt` from OpenCV's wiki
- **SSD MobileNet:** `frozen_inference_graph.pb` + `.pbtxt` (TensorFlow model zoo)
- **RetinaNet:** `retinanet-9.onnx` from the ONNX Model Zoo

## Concepts covered

See [docs/terminology.md](docs/terminology.md) for a deep dive into:

- Bounding box formats (xywh, xyxy, cxcywh, normalized)
- Intersection over Union (IoU) and GIoU/DIoU
- Non-Maximum Suppression (NMS) and Soft-NMS
- Anchors / priors / default boxes
- Confidence vs class score vs objectness
- Mean Average Precision (mAP), Precision-Recall curves
- One-stage vs two-stage detectors
- Backbone, neck (FPN), head
- Focal Loss
- Region Proposal Network (RPN), ROI Pooling / ROI Align
