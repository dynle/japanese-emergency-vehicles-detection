# Japanese Active Emergency Vehicle Detection

<p align="center">
<img width="800" src="https://user-images.githubusercontent.com/36508771/228735246-0e89c703-1415-488e-a7a7-ec501b540b5c.png">
</p>

As the number of people on Earth increases, roads become more congested. The problem is that sometimes normal car drivers cannot respond quickly for emergency vehicles when they are not aware of them approaching. Emergency vehicles, whose quick response is important, arrive at their destinations late due to this problem. In order to solve this problem using only computer vision technology, many studies have been conducted to detect emergency vehicles using data from CCTV or autonomous car. However, we must consider an important information on whether or not those emergency vehicles are active.

I try to solve this problem using only a cost-effective RGB camera with computer vision technology. YOLOv7 trained by transfer learning technique is used as the emergency vehicle detection model, and the activeness of the beacon light is checked by comparing the HSV color of the detected beacon light and the HSV color range of flashing beacon lights at each time frame. If the ratio of the HSV values of the beacon light part of the emergency vehicle falling into the averaged HSV range is larger than the specified threshold value, a message ‘Slow Down and Move Over for Emergency Vehicles’ is displayed on the screen at each time frame. To verify transfer learning and YOLOv7-tiny model, which is suitable for real- time applications, I perform a performance comparison among YOLOv7 and YOLOv7-tiny models with transfer learning, and the two models without it. Using mAP and FPS metrics, YOLOv7-tiny trained by transfer learning (YOLOv7-tiny-EV-TL) is the best model, 97.6% and 47 FPS, thus selected as model for the emergency vehicle detection system. The proposed system has classified active and inactive emergency vehicles to some extent, but there are still many points that need to be solved to actually use.

Keywords: Emergency Vehicle, Deep Learning, Transfer Learning, Object Detection, YOLOv7


## Built With
* OpenCV
* PyTorch
* YOLOv7
* Selenium
* Numpy
* Matplotlib

## License
[MIT](https://choosealicense.com/licenses/mit/)
