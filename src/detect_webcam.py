import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def draw_text(img, text,
                font=cv2.FONT_HERSHEY_TRIPLEX,
                pos=(50, 50),
                font_scale=1,
                font_thickness=2,
                text_color=(0, 0, 255),
                text_color_bg=(51, 195, 236)
                ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w + 5, y + text_h + 5), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size


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
    # colors = [[random.randint(0,255) for _ in range(3)] for _ in names]
    colors = [[random.randint(0,130) for _ in range(3)] for _ in names] #except red color

    # Run Inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))) #run once
    old_img_w = old_img_h = img_size
    old_img_b = 1

    ####################################################################################################
    # Set HSV color range / cut_height / thres for siren light detection
    hsv_lower1 = np.array([0, 20, 225])
    hsv_upper1 = np.array([30, 255, 255])
    hsv_lower2 = np.array([160, 20, 225])
    hsv_upper2 = np.array([180, 255, 255])
    cut_height_fire = 11
    cut_height_police = 8
    cut_height_ambul = 5
    hsv_thres_fire = 0.07
    hsv_thres_police = 0.03
    hsv_thres_ambul = 0.01
    ####################################################################################################

    t0 = time.perf_counter()
    num_obj = 1
    frame_count = 0
    for path, img, im0s, vid_cap in dataset:
        frame_count += 1
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
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
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
                    obj_class = names[int(cls)]
                    label = f'{obj_class} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    '''
                    Crop the 1/10 detected object
                    And check if any region in the cropped object falls in the hsv color range
                    '''
                    if obj_class != "Normal Car":
                        for k in range(len(det)):
                            x,y,w,h=int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])                   
                            img_ = im0.astype(np.uint8)
                            
                            # crop the siren light part at the top
                            if obj_class == "Fire Engine" or obj_class == "Police Car":
                                crop_img=img_[y:y + h//cut_height_fire, x:x + w]
                            elif obj_class == "Police Car":
                                crop_img=img_[y:y + h//cut_height_police, x:x + w]
                            else:
                                crop_img=img_[y:y + h//cut_height_ambul, x:x + w]

                            crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                            crop_img_hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
                            
                            # make a mask to detect siren light
                            hsv_mask1 = cv2.inRange(crop_img_hsv, hsv_lower1, hsv_upper1)
                            hsv_mask2 = cv2.inRange(crop_img_hsv, hsv_lower2, hsv_upper2)
                            hsv_mask = hsv_mask1 + hsv_mask2
                            
                            # number of pixels falls in the hsv color range
    #                         num_pixels = cv2.countNonZero(hsv_mask)
    #                         print(f"Number of pixels falls in the hsv color range / obj{num_obj}: {num_pixels}")
                            total_pixels = crop_img_hsv.shape[0]*crop_img_hsv.shape[1]
    #                         print(f"Number of pixels: {total_pixels}")
                            pixels = cv2.countNonZero(hsv_mask)
    #                         print(f"Number of pixels falls into the range: {pixels}")
                            percentage = pixels/total_pixels
    #                         print(f"Percentage: {percentage}")
                            
                            # emergency state if the percentage of the hsv histogram of the ROI falls in the siren light hsv range is bigger than 0.1
                            if obj_class == "Fire Engine" and percentage > hsv_thres_fire:
    #                             cv2.putText(im0, "Move Over or Slow Down for Emergency Vehicles", org = (50,50),  fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 0, 255), thickness = 3)
                                draw_text(im0, "Slow Down and Move Over for Emergency Vehicles")
                            if obj_class == "Police Car" and percentage > hsv_thres_police:
                                draw_text(im0, "Slow Down and Move Over for Emergency Vehicles")
                            if obj_class == "Ambulance" and percentage > hsv_thres_ambul:
    #                             cv2.putText(im0, "Move Over or Slow Down for Emergency Vehicles", org = (50,50),  fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 0, 255), thickness = 3)
                                draw_text(im0, "Slow Down and Move Over for Emergency Vehicles")
                        
                            # uncomment these lines if you want to save the mask
    #                         result = cv2.bitwise_and(crop_img_rgb, crop_img_rgb, mask=hsv_mask)
    #                         filename = f'mask{num_obj}.png'
    #                         filepath = str(save_dir / filename)
    #                         cv2.imwrite(filepath, hsv_mask)


                        '''
                        Crop and save the detected object
                        uncomment these lines if you want to save the cropped image
                        
                        code from (https://github.com/ultralytics/yolov5/issues/803 and 2608)
                        '''
                        # save_obj = True
                        # if save_obj:
                        #     for k in range(len(det)):
                        #         x,y,w,h=int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])                   
                        #         img_ = im0.astype(np.uint8)
                        #         #IDEA: crop the siren light part
                        #         if obj_class == "Fire Engine":
                        #             crop_img=img_[y:y + h//cut_height_fire, x:x + w]
                        #         elif obj_class == "Police Car":
                        #             crop_img=img_[y:y + h//cut_height_police, x:x + w]
                        #         else:
                        #             crop_img=img_[y:y + h//cut_height_ambul, x:x + w]                                 

                        #         #!!rescale image !!!
                        #         filename = f'cropped{num_obj}.png'
                        #         filepath = str(save_dir / filename)
                        #         cv2.imwrite(filepath, crop_img)

                        # else:
                        #     print("There is no detected object")
                        #     continue

                        # num_obj+=1
#
#
        cv2.imshow(str(p), im0)

    # calculate the average fps
    avg_fps = frame_count / (time.time()-t0)
    # print("Printing Average FPS for inference + nms + postprocess + saving_results")
    # print(f"Average FPS: {avg_fps:.2f}")

    print(f'Done. ({time.perf_counter() - t0:.3f})')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(device)
    
    with torch.no_grad():
        # yolov7tiny-freezing28 IDEA: Use this lightweight model for this time because of limitations of test environment
        # detect("0", "./train-results/yolov7tiny-freezing28-normalcar/weights/best.pt", device, img_size=640, iou_thres=0.45, conf_thres=0.8)
        # iphone camera
        detect("1", "./train-results/yolov7tiny-freezing28-normalcar/weights/best.pt", device, img_size=640, iou_thres=0.45, conf_thres=0.8)
        
        # yolo7-freezing50
        # detect("0", "./train-results/yolov7-freezing50/weights/best.pt", device, img_size=640, iou_thres=0.45, conf_thres=0.8)
