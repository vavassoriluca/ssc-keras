# Object Detection: Classification and Localization of objects within an image or video

Object Detection is a useful tool in today's applications. It is widely used in computer vision task such as face detection, face recognition, video object co-segmentation. It is also used in tracking objects, for example tracking a ball during a football match, tracking movement of a cricket bat, tracking a person in a video. 

Object Detecion algorithms can be classified in 2 categories:
- Two-Stage Approaches
- Single-Stage Approaches

## Two-Stage Approaches

1. R-CNN
2. Fast R-CNN
3. Faster R-CNN

The history and explanation of the basics of these algorithms are well presented in this [blog post](https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4)

## Single-Stage Approaches

1. SSD
2. YOLO
3. RetinaNet

The explanation of these algorithms are well presented in this [blog post](https://medium.com/@jonathan_hui/what-do-we-learn-from-single-shot-object-detectors-ssd-yolo-fpn-focal-loss-3888677c5f4d)

## Module content

In this module the Keras implementation of SSD by Pierluigi Ferrari is provided, along with a mAP evaluator.

## Performance table

FPS(Speed) index is related to the hardware spec(e.g. CPU, GPU, RAM, etc), so it is hard to make an equal comparison. 

|   Detector   | VOC07 (mAP@IoU=0.5) | VOC12 (mAP@IoU=0.5) | COCO (mAP@IoU=0.5:0.95) | Published In |
|:------------:|:-------------------:|:-------------------:|:----------:|:------------:| 
|     R-CNN    |         58.5        |          -          |      -     |    CVPR'14   |
|  Fast R-CNN  |     70.0 (07+12)    |     68.4 (07++12)   |    19.7    |    ICCV'15   |
| Faster R-CNN |     73.2 (07+12)    |     70.4 (07++12)   |    21.9    |    NIPS'15   |
|    YOLO v1   |     66.4 (07+12)    |     57.9 (07++12)   |      -     |    CVPR'16   |
|      SSD     |     76.8 (07+12)    |    74.9 (07++12)    |    31.2    |    ECCV'16   |
|    YOLO v2   |     78.6 (07+12)    |    73.4 (07++12)    |      -     |    CVPR'17   |
|   RetinaNet  |          -          |          -          |    39.1    |    ICCV'17   |

For a discussion about the comparison check [this blog post.](https://medium.com/@jonathan_hui/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359)