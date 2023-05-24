# -*- coding: utf-8 -*-
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from .temporal.tracker import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from .models.common import DetectMultiBackend
from .utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from .utils.general import (
    LOGGER,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_coords,
    strip_optimizer,
    xyxy2xywh,
)
from .utils.plots import Annotator, colors, save_one_box
from .utils.torch_utils import select_device, time_sync
from .utils.augmentations import letterbox


@torch.no_grad()
def run(
    weights=ROOT / "yolov5s.pt",  # model.pt path(s)
    source=ROOT / "data/images",  # file/dir/URL/glob, 0 for webcam
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    temporal=None,  # temporal analysis technique used after detection
    area_thresh=0.05,  # suppression threshold when temporal analysis technique is tracker
    window_size=20,  # sliding window size for temporal analysis technique
    task="test",  # perform validation or test
    persistence_thresh=0.5,  # persistence threshold when temporal analysis technique is persistence
    image=None,
):
    t1 = time.time()
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = (
        model.stride,
        model.names,
        model.pt,
        model.jit,
        model.onnx,
        model.engine,
    )
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (
        pt or jit or onnx or engine
    ) and device.type != "cpu"  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    print("\nFire detector without temporal analysis technique\n")
    # output_name += '.csv'
    threshold = None
    window_size = None

    nframes = 1
    detected = False
    time_list = list()
    first_frame = None

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    # model.warmup(imgsz=(1 if pt else bs, 3, imgsz, imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    # cap = cv2.VideoCapture(0)
    # while True:
    #     # 从splitcam读取一帧
    #     ret, frame = cap.read()
    #     # 如果读取成功，显示在窗口上
    #     if ret:
    # image = frame
    im0 = image  # BGR
    assert im0 is not None, f"Image Not Found"
    # Padded resize
    img = letterbox(im0, imgsz, stride=stride, auto=pt)[0]
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    dataset = [[None, img, im0, None, ""]]

    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        visualize = False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
        )
        dt[2] += time_sync() - t3
        print(f"dt={dt}")
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            result_data = []

            seen += 1
            # if webcam:  # batch_size >= 1
            #     p, im0, frame = path[i], im0s[i].copy(), dataset.count
            #     s += f'{i}: '
            # else:
            #     p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            im0 = im0s.copy()

            # p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # im.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                result_data = []
                for *xyxy, conf, cls in reversed(det):
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    xywh = (
                        (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                        .view(-1)
                        .tolist()
                    )  # normalized xywh
                    line = (int(cls.item()), *xywh, conf.item())  # label format
                    result_data.append(line)

                    # if save_img or save_crop or view_img:  # Add bbox to image
                    #     c = int(cls)  # integer class
                    #     label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    #     annotator.box_label(xyxy, label, color=colors(c, True))
                    #     if save_crop:
                    #         save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))

            if det.shape[0] > 0 and detected is False:
                first_frame = nframes
                detected = True
            # Print time (inference-only)
            time_list.append(t3 - t2)
            LOGGER.info(f"{s}Done. ({t3 - t2:.3f}s)")
            nframes += 1
            print(nframes)

            # Stream results
            im0 = annotator.result()
            # cv2.imshow('', im0)
            # cv2.waitKey(1)
            # if view_img:
            #     cv2.imshow(str(p), im0)
            #     cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #     else:  # 'video' or 'stream'
            #         if vid_path[i] != save_path:  # new video
            #             vid_path[i] = save_path
            #             if isinstance(vid_writer[i], cv2.VideoWriter):
            #                 vid_writer[i].release()  # release previous video writer
            #             if vid_cap:  # video
            #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             else:  # stream
            #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
            #                 save_path += '.mp4'
            #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #         vid_writer[i].write(im0)

    # Print results
    # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # if update:
    #     strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    # results = pd.DataFrame(columns = ['video', 'technique', 'threshold', 'window_size', 'detected', 'first_frame', 'time_avg'])
    # results.loc[0] = [video_name, temporal, threshold, window_size, detected, first_frame, np.mean(time_list)]
    # print("Results")
    # print(results)
    t2 = time.time()
    print("run function", t2 - t1)
    return im0, result_data


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default=ROOT / "yolov5s.pt",
        help="model path(s)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=ROOT / "data/images",
        help="file/dir/URL/glob, 0 for webcam",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=ROOT / "data/coco128.yaml",
        help="(optional) dataset.yaml path",
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640],
        help="inference size h,w",
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--max-det", type=int, default=1000, help="maximum detections per image"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--save-crop", action="store_true", help="save cropped prediction boxes"
    )
    parser.add_argument(
        "--nosave", action="store_true", help="do not save images/videos"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --classes 0, or --classes 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument(
        "--project", default=ROOT / "runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--line-thickness", default=3, type=int, help="bounding box thickness (pixels)"
    )
    parser.add_argument(
        "--hide-labels", default=False, action="store_true", help="hide labels"
    )
    parser.add_argument(
        "--hide-conf", default=False, action="store_true", help="hide confidences"
    )
    parser.add_argument(
        "--half", action="store_true", help="use FP16 half-precision inference"
    )
    parser.add_argument(
        "--dnn", action="store_true", help="use OpenCV DNN for ONNX inference"
    )
    parser.add_argument(
        "--temporal", type=str, help="temporal analysis technique used after detection"
    )
    parser.add_argument(
        "--area-thresh",
        type=float,
        default=0.05,
        help="suppression threshold when temporal analysis technique is tracker",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="sliding window size for temporal analysis technique",
    )
    parser.add_argument(
        "--task", type=str, default="test", help="perform validation or test"
    )
    parser.add_argument(
        "--persistence-thresh",
        type=float,
        default=0.50,
        help="suppression threshold when temporal analysis technique is persistence",
    )
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


class yolo_detector:
    def __init__(
        self,
        weights="yolov5s.pt",  # 用train.py训练出的.pt文件
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        half=False,
    ):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.device = (
            select_device("0") if torch.cuda.is_available() else select_device("cpu")
        )
        self.model = DetectMultiBackend(weights, device=self.device)  # 加载模型
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        self.stride, self.names, self.pt = stride, names, pt
        self.imgsz = check_img_size(imgsz, s=stride)  # check image size
        half &= (
            pt and self.device.type != "cpu"
        )  # half precision only supported by PyTorch on CUDA
        self.half = half
        if pt:
            self.model.model.half() if half else self.model.model.float()
        self.view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model.warmup(imgsz=(1, 3, *self.imgsz), half=self.half)

    def run(self, frame):
        # Read image
        im0 = frame  # BGR
        assert im0 is not None, f"Image Not Found"

        # Padded resize

        img = letterbox(im0, self.imgsz, stride=self.stride, auto=self.pt)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        # # (h, w, c) to (c, h, w)
        # b, g, r = cv2.split(frame)
        # im0 = np.array([b, g, r])

        im = torch.from_numpy(img).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred = self.model(im)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, max_det=5)

        results = []
        annotator = Annotator(im0, line_width=3, example=str(self.names))
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in reversed(det):
                line = (cls, xyxy, conf)
                print(line)
                results.append(line)

                c = int(cls)  # integer class
                label = f"{self.names[c]} {conf:.2f}"
                annotator.box_label(xyxy, label, color=colors(c, True))

            im0 = annotator.result()

            # if det.numel():
            #     x1, y1, x2, y2 = int(det[0, 0].item()), int(det[0, 1].item()), int(det[0, 2].item()), int(
            #         det[0, 3].item())
            #     lu = (x1, y1)
            #     rd = (x2, y2)
            #     results.append((lu, rd))
        return im0, results


def main(opt, image):
    t1 = time.time()
    check_requirements(exclude=("tensorboard", "thop"))
    a, b = run(**vars(opt), image=image)
    # a, b = run(image=image)
    t2 = time.time()
    print("main func", t2 - t1)
    return a, b


if __name__ == "__main__":
    opt = parse_opt()
    main(opt, None)
