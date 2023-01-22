import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect(source, weights, device, img_size, iou_thres, conf_thres):
    
    webcam = source.isnumeric()
    
    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu' # half precision only supported on CUDA
    
    # Local Model
    model = attempt_load(weights, map_location=device) #load FP32 model
    stride = int(model.stride.max()) #model stride
    imgsz = check_img_size(img_size, s=stride) #check img_size
    
    if half:
        model.half() #to FP16
        
    # Set Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True #set True to speed up constant image size inference
        # IDEA: set frame width, frame height, and fps in utils/datasets.py LoadSteams class
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    
    # Get Names and Colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0,255) for _ in range(3)] for _ in names]
    
    # Run Inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))) #run once
    
    old_img_w = old_img_h = img_size
    old_img_b = 1
    
    t0 = time.perf_counter()
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float() #uint8 to fp16/32
        img /= 255.0 #0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqeeze(0)
            
        # Wramup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
        
        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img)[0]
        t2 = time_synchronized()
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        t3 = time_synchronized()
        
        
        # Process Detections
        for i, det in enumerate(pred): #detections per image
            if webcam: #batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                
            p = Path(p) #to Path
            
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] #normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Print Results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum() #detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, " #add to string
                
                # Write Results
                for *xyxy, conf, cls in reversed(det):
                    
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    
                    
        cv2.imshow(str(p), im0)
        
    print(f'Done. ({time.perf_counter() - t0:.3f})')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(device)
    
    with torch.no_grad():
        # yolov7tiny-freezing28 IDEA: Use this lightweight model for this time because of limitations of test environment
        detect("0", "./train-results/yolov7tiny-freezing28/weights/best.pt", device, img_size=640, iou_thres=0.45, conf_thres=0.8)
        
        # yolo7-freezing50
        # detect("0", "./train-results/yolov7-freezing50/weights/best.pt", device, img_size=640, iou_thres=0.45, conf_thres=0.8)
