from yolort.runtime.ort_helper import export_onnx
import cv2
import onnx
import onnxruntime
from yolort.models import YOLOv5
from yolort.v5 import attempt_download
from yolort.utils import get_image_from_url, read_image_to_tensor
from yolort.utils.image_utils import to_numpy


img_size = 800
size = (img_size, img_size)  # Used for pre-processing
size_divisible = 64
score_thresh = 0.5
nms_thresh = 0.5
opset_version = 11


model = YOLOv5.load_from_yolov5(
    "/home/talha/oneTB/yolov5-face/YOLOFACE/exp6/weights/best.pt",
    size=size,
    size_divisible=size_divisible,
    score_thresh=score_thresh,
    nms_thresh=nms_thresh,
)

model = model.eval()



export_onnx(model=model, onnx_path="best_nms.onnx", opset_version=opset_version)