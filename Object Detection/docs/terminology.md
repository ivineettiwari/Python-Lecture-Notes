# Object Detection Terminology

A practical glossary for everything you'll touch in this project. Each term
includes the math/intuition and how it shows up in code.

---

## 1. Bounding Box

A rectangle that localises an object. Common encodings:

| Format    | Tuple                          | Used by                         |
|-----------|--------------------------------|---------------------------------|
| `xyxy`    | (x_min, y_min, x_max, y_max)   | OpenCV `rectangle`, evaluation  |
| `xywh`    | (x_min, y_min, width, height)  | COCO annotations                |
| `cxcywh`  | (center_x, center_y, w, h)     | YOLO outputs                    |
| Normalised| Divide by image (w, h) → [0,1] | SSD, Faster R-CNN OpenCV output |

Convert freely with helpers in [utils/box_ops.py](../utils/box_ops.py).

---

## 2. Intersection over Union (IoU)

Measures overlap between predicted box `P` and ground-truth box `G`:

```
IoU = area(P ∩ G) / area(P ∪ G)        ∈ [0, 1]
```

- IoU = 1.0 → perfect match
- IoU ≥ 0.5 → typically counted as a true positive in PASCAL VOC mAP
- COCO uses IoU thresholds 0.5, 0.55, ..., 0.95 averaged

Variants:
- **GIoU** — penalises distance between non-overlapping boxes
- **DIoU / CIoU** — used as YOLOv4/v5 regression losses

---

## 3. Non-Maximum Suppression (NMS)

A detector typically emits many overlapping boxes for the same object. NMS
keeps the most confident one and removes the rest:

```
1. Sort boxes by confidence (high → low)
2. Take the top box, add it to "keep"
3. Remove every remaining box whose IoU with it ≥ nms_threshold
4. Repeat with next remaining top box
```

OpenCV exposes `cv2.dnn.NMSBoxes(...)`. **Soft-NMS** decays neighbour scores
instead of dropping them — better when objects truly overlap.

---

## 4. Anchors / Priors / Default Boxes

Pre-defined reference rectangles tiled over the image (or feature map). The
network learns *offsets* from each anchor to the true box, not absolute
coordinates. This makes regression easier.

- **YOLO** uses 3 anchors per scale (9 total in YOLOv3) chosen by k-means on
  the training set.
- **SSD** uses default boxes with multiple aspect ratios per feature-map cell.
- **Faster R-CNN** uses 9 anchors (3 scales × 3 ratios) at each RPN location.

---

## 5. Objectness vs Class Score vs Confidence

YOLO outputs three numbers per box:

```
objectness   = P(an object is here)
class_scores = P(class_i | object) for each class
confidence   = objectness × max(class_scores)
```

Faster R-CNN / SSD typically merge these: a single softmax over `K + 1`
classes (extra slot = background).

---

## 6. Mean Average Precision (mAP)

The standard accuracy metric.

1. Sort all detections (across all images) by confidence.
2. Walk down the list; each detection is TP (matches an unmatched GT box at
   IoU ≥ τ) or FP.
3. Compute precision & recall at every threshold → Precision-Recall curve.
4. **AP** = area under the PR curve (per class).
5. **mAP** = mean of AP over all classes.

Notation:
- `mAP@0.5` (PASCAL VOC) — IoU threshold 0.5
- `mAP@[.5:.95]` (COCO) — averaged over 10 IoU thresholds

---

## 7. One-Stage vs Two-Stage Detectors

| Stage     | Pipeline                                                       | Examples                |
|-----------|----------------------------------------------------------------|-------------------------|
| Two-stage | (1) propose regions → (2) classify + refine each region        | Faster R-CNN, Mask R-CNN|
| One-stage | Predict class + box in a single forward pass over a dense grid | YOLO, SSD, RetinaNet    |

Two-stage is usually more accurate but slower. One-stage is real-time-friendly.

---

## 8. Backbone / Neck / Head

Modern detectors are modular:

- **Backbone** — feature extractor (ResNet, Darknet, MobileNet, EfficientNet)
- **Neck** — multi-scale fusion (Feature Pyramid Network / FPN, PAN)
- **Head** — task-specific layers that produce class + box outputs

---

## 9. Region Proposal Network (RPN)

Faster R-CNN's first stage. A small conv network slides over the backbone
feature map and outputs:
- Objectness for each of 9 anchors
- Bounding-box refinement for each anchor

Top-N proposals (after NMS) are fed to the second stage.

---

## 10. ROI Pooling / ROI Align

Each region proposal has a different size, but the second-stage classifier
needs a fixed-size feature. ROI Pooling crops + max-pools to e.g. 7×7.
**ROI Align** (Mask R-CNN) replaces the quantised pooling with bilinear
interpolation — preserves sub-pixel accuracy and is critical for masks.

---

## 11. Focal Loss (RetinaNet)

Standard cross-entropy is dominated by the huge number of easy background
anchors in dense one-stage detectors. Focal loss down-weights easy examples:

```
FL(p_t) = -α (1 - p_t)^γ log(p_t)
```

`γ = 2` is typical. This was the key insight that let RetinaNet match
two-stage accuracy at one-stage speed.

---

## 12. Confidence Threshold vs NMS Threshold

Two knobs you'll tune in every script:

- **`conf_threshold`** — discard predictions with score < this value (e.g. 0.5)
- **`nms_threshold`**  — IoU above which overlapping boxes are suppressed (e.g. 0.4)

Lower confidence → more detections (more recall, more false positives).
Lower NMS threshold → more aggressive suppression (fewer overlapping boxes).

---

## 13. Letterbox / Resize

Most networks expect a fixed input size (e.g. 416×416 for YOLO). Naively
resizing distorts aspect ratio. **Letterboxing** scales preserving aspect
ratio and pads with grey borders, then un-pads predicted boxes back to the
original image. OpenCV's `cv2.dnn.blobFromImage` does plain resize; for
serious accuracy you implement letterbox manually.

---

## 14. COCO 80 vs 91 classes

The COCO dataset annotates 80 object categories but the original ID space
runs to 91 (with gaps). TensorFlow models typically output 91-style IDs while
YOLO uses 80-style. Mismatches here are the #1 source of "wrong labels".
This project uses the 80-class list in
[utils/coco_labels.py](../utils/coco_labels.py).
