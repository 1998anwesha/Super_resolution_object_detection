# Super resolution and Object_detection on Remote Sensing Images

### CSCI 5561 project overview

This project focuses on enhancing remote sensing image quality and improving object detection accuracy in aerial imagery. Using the DOTA v1 dataset, low-resolution images are simulated and restored with Super-Resolution Generative Adversarial Network (SRGAN) to enhance sharpness and fine details.

Object detection is performed with YOLO-v11 and YOLO-v11 OBB (Oriented Bounding Box), enabling accurate detection of objects at varying orientations. The method identifies eight object classes and demonstrates significant improvements in image quality and detection accuracy through the integration of super-resolution and oriented bounding boxes.

Comparative results on low-resolution vs. super-resolved images highlight the effectiveness of this approach for applications in remote sensing and aerial surveillance.
 
DOTA v1 dataset can be found [here](https://captain-whu.github.io/DOTA/dataset.html).

Link to download SRGAN checkpoint train on DOTA v1 dataset: [link](https://drive.google.com/file/d/10eBCHZLtl8HBMqCL90cezlKF04cOB1Bh/view?usp=sharing)

Link to download YOLOv11 and YOLOv11-OBB models: [link](https://drive.google.com/drive/folders/18TEWaciGL6Be6P3vVEDOk55vjX7dM6x4?usp=sharing)


# Streamlit GUI

Choose a few test images from the DOTA V1 dataset for the GUI demo. The object detection results on these images highlight the improvements achieved by using SRGAN for super-resolution compared to low-resolution inputs. 
We observe that there is an improvement in the average confidence scores of OBB detections and there is an increase in the total number of small objects detected on Super resolved images.

