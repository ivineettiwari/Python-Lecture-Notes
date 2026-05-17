# Model weights

The detector code looks for files in these subfolders. Drop the downloads in
exactly these locations:

```
models/
├── yolo/
│   ├── yolov3.weights
│   ├── yolov3.cfg
│   └── coco.names
├── faster_rcnn/
│   ├── frozen_inference_graph.pb
│   └── faster_rcnn_inception_v2_coco_2018_01_28.pbtxt
├── ssd/
│   ├── frozen_inference_graph.pb
│   └── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
└── retinanet/
    └── retinanet-9.onnx
```

## Download links

### YOLOv3
- weights: <https://pjreddie.com/media/files/yolov3.weights>
- cfg:     <https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg>
- names:   <https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names>

### YOLOv4 (drop-in replacement)
- weights: <https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights>
- cfg:     <https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg>

### Faster R-CNN (Inception v2, COCO)
See the OpenCV wiki for the matching `.pb` and `.pbtxt`:
<https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API>

### SSD MobileNet v3 (COCO)
- pb / pbtxt: <https://github.com/opencv/opencv_extra/tree/master/testdata/dnn>

### RetinaNet (ONNX)
- model: <https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/retinanet>

## PowerShell download helpers

```powershell
mkdir models\yolo -Force
Invoke-WebRequest -Uri https://pjreddie.com/media/files/yolov3.weights -OutFile models\yolo\yolov3.weights
Invoke-WebRequest -Uri https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -OutFile models\yolo\yolov3.cfg
Invoke-WebRequest -Uri https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -OutFile models\yolo\coco.names
```
