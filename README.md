# Super resolution and Object_detection on Remote Sensing Images

### CSCI 5561 project overview

This project focuses on enhancing remote sensing image quality and improving object detection accuracy in aerial imagery. Using the DOTA v1 dataset, low-resolution images are simulated and restored with Super-Resolution Generative Adversarial Network (SRGAN) to enhance sharpness and fine details.

Object detection is performed with YOLO-v11 and YOLO-v11 OBB (Oriented Bounding Box), enabling accurate detection of objects at varying orientations. The method identifies eight object classes and demonstrates significant improvements in image quality and detection accuracy through the integration of super-resolution and oriented bounding boxes.

Comparative results on low-resolution vs. super-resolved images highlight the effectiveness of this approach for applications in remote sensing and aerial surveillance.
 
DOTA v1 dataset can be found [here](https://captain-whu.github.io/DOTA/dataset.html).

Link to download SRGAN checkpoint train on DOTA v1 dataset: [link](https://drive.google.com/file/d/10eBCHZLtl8HBMqCL90cezlKF04cOB1Bh/view?usp=sharing)

Link to download YOLOv11 and YOLOv11-OBB models: [link](https://drive.google.com/drive/folders/18TEWaciGL6Be6P3vVEDOk55vjX7dM6x4?usp=sharing)


# Streamlit GUI

We developed a Streamlit-based GUI for interacting with the super-resolution and object detection pipeline. Choose a few test images from the DOTA V1 dataset for the GUI demo. The GUI offers the following features:
#### 1. Image Upload & Model Selection: Users can upload images, select object detection models (YOLOv11 or YOLOv11-OBB), and super-resolution models (SRGAN or SRResNet) to observe differences.


#### 2. Super-Resolution Results: Displays super-resolved images with PSNR and SSIM metrics compared to the high-resolution ground truth.
![gui1](https://github.com/anwesha-umn/Super_resolution_object_detection/blob/main/imgs/gui1.png)


#### 3. Object Detection on Super-Resolved Images: Shows detected objects with bounding boxes and class labels using the chosen YOLO model.

![gui2](https://github.com/anwesha-umn/Super_resolution_object_detection/blob/main/imgs/gui2.png)

#### 4. Comparison of Low-Resolution vs. Super-Resolved Detections: 
        + Number of detections (confidence ≥ 0.5).
        + Average confidence scores for detections.
![gui3](https://github.com/anwesha-umn/Super_resolution_object_detection/blob/main/imgs/gui3.png)

This GUI allows users to visually assess how SRGAN improves object detection on real-world test data.





