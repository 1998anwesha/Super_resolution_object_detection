from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import *
import torch
import torch.nn as nn
from ultralytics import YOLO
from skimage.metrics import structural_similarity as compare_ssim
import torch.nn as nn
from torchvision import transforms
import streamlit as st
from matplotlib.patches import Polygon
from collections import Counter
import matplotlib.patches as patches
from shapely.geometry import Polygon as poly
import pandas as pd
from matplotlib.patches import Rectangle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoints
srgan_checkpoint = "./checkpoint_srgan.pth.tar"
srresnet_checkpoint = "./checkpoint_srresnet.pth.tar"


def calculate_psnr(img1, img2):
    """
    Calculates PSNR (Peak Signal-to-Noise Ratio) between two images.
    Assumes img1 and img2 are PIL images with the same dimensions.
    
    :param img1: PIL Image
    :param img2: PIL Image
    :return: PSNR value
    """
    # Convert images to numpy arrays
    img1 = np.array(img1).astype(np.float32)
    img2 = np.array(img2).astype(np.float32)
    
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """
    Calculate SSIM between two images.

    :param img1: First image (PIL Image).
    :param img2: Second image (PIL Image).
    :return: SSIM value.
    """
    img1 = np.array(img1)
    img2 = np.array(img2)
    # Ensure the images are of the same dimensions
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions for SSIM calculation.")

    # Dynamically determine the window size
    win_size = min(7, img1.shape[0], img1.shape[1])  # Choose 7 or smaller dimensions
    if win_size < 3:
        win_size = 3  # Set to the smallest odd value
    
    # Check if the images are multichannel
    channel_axis = -1 if len(img1.shape) == 3 else None

    # Calculate SSIM
    ssim_value, _ = compare_ssim(
        img1, img2, full=True, channel_axis=channel_axis, win_size=win_size
    )
    return ssim_value

# Super-resolution Visualization
def visualize_sr_plot(hr_img,sr_model, halve=False):
    """
    Visualizes the super-resolved images from SRResNet and SRGAN for comparison
    with the bicubic-upsampled image and the original high-resolution (HR) image.

    :param img_path: File path of the HR image.
    :param halve: Whether to halve the HR image dimensions for visualization.
    """
    # Load HR image
    #hr_img = Image.open(img_path).convert('RGB')
    if halve:
        hr_img = hr_img.resize((hr_img.width // 2, hr_img.height // 2), Image.LANCZOS)
    
    # Generate low-resolution (LR) image
    blurred_img = hr_img.filter(ImageFilter.GaussianBlur(radius=1))
    lr_img = blurred_img.resize((hr_img.width // 4, hr_img.height // 4), Image.BOX)

    # Super-resolution (SR) with SRGAN / SRRESNET
    sr_img = sr_model(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img = sr_img.squeeze(0).cpu().detach()
    sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
    
    # Calculate PSNR for SRGAN / SRRESNET
    #hr_img_resized = hr_img.resize((sr_img.width, sr_img.height), Image.LANCZOS)
    psnr_sr = calculate_psnr(sr_img, hr_img)

    # Calculate SSIM
    ssim_sr = calculate_ssim(sr_img, hr_img)

    # Set up the Matplotlib figure and axes (1 row x 3 columns)
    fig, axs = plt.subplots(1, 3, figsize=(30, 15))

    # Display images and titles
    axs[0].imshow(lr_img)
    axs[0].set_title(f"LR Input: {lr_img.size}", fontsize=24) 
    axs[0].axis('off')

    axs[1].imshow(sr_img)
    axs[1].set_title(f"Super-Resolved Image: {sr_img.size} \nPSNR: {psnr_sr:.2f} dB | SSIM: {ssim_sr:.3f}", fontsize=24) 
    axs[1].axis('off')

    axs[2].imshow(hr_img)
    axs[2].set_title(f"GT High Res Image: {hr_img.size}", fontsize=24) 
    axs[2].axis('off')

    return fig, sr_img, lr_img



def plot_detections(img, detections, move, color):
    """
    Display the low-resolution image, super-resolved image, and YOLO detection output with OBBs.

    Parameters:
        lr_img (PIL.Image.Image): Low-resolution input image.
        detections (list): Detection results from the YOLO model.
    """
    detection_output_path = "temp_detection_result.png"
    detections[0].save(detection_output_path)  # Save the detection result
    detection_img = Image.open(detection_output_path)  # Load the saved image

    # Create a plot with two panels (1x2 layout)
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))

    # Show low-resolution image
    axs[0].imshow(img)
    axs[0].set_title(f"Super resolved Image Input: {img.size}", fontsize=24) 
    axs[0].axis('off')

    # Show super-resolved image
    axs[1].imshow(detection_img)
    axs[1].set_title("Detections on image", fontsize=24) 
    axs[1].axis('off')

    axs[2].imshow(img)
    axs[2].set_title("Bounding boxes without labels", fontsize=24) 
    axs[2].axis('off')

    # Image dimensions
    img_height, img_width = np.array(img).shape[:2]

    # Draw detections on the super-resolved image (axs[1])
    for result in detections:

        if result.obb:  # Check if OBB results are available
            # result.show()
            obb_data = result.obb
            xyxyxyxyn = obb_data.xyxyxyxyn.cpu().numpy()  # Get normalized 8-point polygon coordinates
            conf = obb_data.conf.cpu().numpy()  # Get confidence scores
            cls = obb_data.cls.cpu().numpy()  # Get class labels

            for i, points in enumerate(xyxyxyxyn):
                # Only process detections above the confidence threshold
                if conf[i] < 0.5:
                    continue

                # Scale normalized points to image dimensions
                points[:, 0] *= img_width  # Scale x-coordinates
                points[:, 1] *= img_height  # Scale y-coordinates

                label = int(cls[i])  # Class label
                confidence = conf[i]  # Confidence score

                # Draw the polygon
                polygon = Polygon(points, linewidth=3, edgecolor=color, facecolor='none')
                axs[2].add_patch(polygon)

                # Add label text
                centroid = points.mean(axis=0)  # Compute centroid for text placement
                # axs[2].text(
                #     centroid[0]+move,
                #     centroid[1],
                #     f"{result.names[label]}: {confidence:.2f}",
                #     color='white',
                #     fontsize=10,
                #     bbox=dict(facecolor=color, alpha=0.35, edgecolor='none')
                # )
        else:
            # Loop through bounding boxes and draw polygons
            for box in result.boxes:
                # Extract bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Create a rectangle as a polygon
                bbox_polygon = Polygon(
                    [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],  # Rectangle corners
                    closed=True,
                    edgecolor="red",  # Color of the bounding box
                    linewidth=3,        # Thickness of the edges
                    fill=False          # Do not fill the rectangle
                )
                axs[2].add_patch(bbox_polygon)


    # Adjust layout and show the plot
    plt.tight_layout()
    #plt.show()

    return fig

def compare_detections_yolo(detections_1, detections_2, lr_image, sr_image, iou_threshold=0.5):
    """
    Compare two sets of YOLOv8 axis-aligned bounding box (AABB) detections.
    Outputs:
    - Class distribution comparison
    - Confidence scores comparison
    - Bounding Box IoU comparison
    - Number of detections
    """
    # Extract information from detections
    cls_1 = detections_1[0].boxes.cls.cpu().numpy()  # Class labels
    conf_1 = detections_1[0].boxes.conf.cpu().numpy()  # Confidence scores
    xyxy_1 = detections_1[0].boxes.xyxy.cpu().numpy()  # BBoxes (x1, y1, x2, y2)

    cls_2 = detections_2[0].boxes.cls.cpu().numpy()
    conf_2 = detections_2[0].boxes.conf.cpu().numpy()
    xyxy_2 = detections_2[0].boxes.xyxy.cpu().numpy()

    # Prepare results dictionary
    results = {}

    # 1. Average confidence scores
    avg_conf_1 = np.mean(conf_1)
    avg_conf_2 = np.mean(conf_2)
    results['Average Confidence (Low Res)'] = avg_conf_1
    results['Average Confidence (Super Res)'] = avg_conf_2

    # 2. Confidence above threshold
    conf_above_thresh_1 = sum(conf_1 >= 0.5)
    conf_above_thresh_2 = sum(conf_2 >= 0.5)
    results['Detections on Low Res with conf >= 0.5'] = conf_above_thresh_1
    results['Detections on Super Res with conf >= 0.5'] = conf_above_thresh_2

    # 3. Total number of detections
    # results['Total Detections (Low Res)'] = len(cls_1)
    # results['Total Detections (Super Res)'] = len(cls_2)

    # Convert results to pandas DataFrame for display in Streamlit
    df_results = pd.DataFrame(list(results.items()), columns=["Metric", "Value"])

    # Display the results table in Streamlit
    st.write("Comparison Results:")
    st.dataframe(df_results)

    # Visualization for bounding boxes
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for Model 1 (Low resolution detections)
    axs[0].imshow(lr_image)
    for bbox, conf in zip(xyxy_1, conf_1):
        if conf >= 0.45:  # Plot if confidence is above threshold
            x1, y1, x2, y2 = bbox
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='blue', facecolor='none')
            axs[0].add_patch(rect)
    axs[0].set_title('Low Resolution Detections')

    # Plot for Model 2 (High resolution detections)
    axs[1].imshow(sr_image)
    for bbox, conf in zip(xyxy_2, conf_2):
        if conf >= 0.45:
            x1, y1, x2, y2 = bbox
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='red', facecolor='none')
            axs[1].add_patch(rect)
    axs[1].set_title('Super Resolution Detections')

    # Show the plot in Streamlit
    st.pyplot(fig)


def compare_detections_obb(detections_1, detections_2, lr_image, sr_image, iou_threshold=0.5):
    """
    Compare two sets of OBB detections and output results:
    - Class distribution
    - Confidence scores comparison
    - Bounding Box IoU comparison for polygons
    - Number of detections
    """
    # Extract information from detections
    cls_1 = detections_1[0].obb.cls.cpu().numpy()
    conf_1 = detections_1[0].obb.conf.cpu().numpy()
    xy_1 = detections_1[0].obb.xyxyxyxy.cpu().numpy()  # Shape (n_boxes, 4, 2)

    cls_2 = detections_2[0].obb.cls.cpu().numpy()
    conf_2 = detections_2[0].obb.conf.cpu().numpy()
    xy_2 = detections_2[0].obb.xyxyxyxy.cpu().numpy()  # Shape (n_boxes, 4, 2)

    # Prepare results
    results = {}

    # 3. Compare average confidence scores
    avg_conf_1 = np.mean(conf_1)
    avg_conf_2 = np.mean(conf_2)
    results['Average Confidence Low-res image'] = avg_conf_1
    results['Average Confidence Super-res image'] = avg_conf_2

    # 4. Compare confidence above threshold
    conf_above_thresh1 = sum([1 for a in conf_1 if a >= 0.5])
    conf_above_thresh2 = sum([1 for a in conf_2 if a >= 0.5])
    results['Low Res Detections (conf >= 0.5)'] = conf_above_thresh1
    results['Super Res Detections (conf >= 0.5)'] = conf_above_thresh2

    # 5. Compare number of detections
    num_detections_1 = len(cls_1)
    num_detections_2 = len(cls_2)
    # results['Total Low Res Detections'] = num_detections_1
    # results['Total High Res Detections'] = num_detections_2

    # Convert results to pandas DataFrame for display in Streamlit
    df_results = pd.DataFrame(list(results.items()), columns=["Metric", "Value"])
    
    # Display the results table in Streamlit
    st.write("Comparison Results:")
    st.dataframe(df_results)

    # Visual Comparison 
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for Model 1 (Low resolution detections)
    axs[0].imshow(lr_image)
    for bbox, conf in zip(xy_1, conf_1):
        if conf >= 0.5:  # Only plot if confidence is above the threshold
            poly = patches.Polygon(bbox, linewidth=1, edgecolor='blue', facecolor='none')
            axs[0].add_patch(poly)
    axs[0].set_title('Low resolution Detections')

    # Plot for Model 2 (High resolution detections)
    axs[1].imshow(sr_image)
    for bbox, conf in zip(xy_2, conf_2):
        if conf >= 0.5:  # Only plot if confidence is above the threshold
            poly = patches.Polygon(bbox, linewidth=1, edgecolor='red', facecolor='none')
            axs[1].add_patch(poly)
    axs[1].set_title('Super resolution Detections')

    # Show plot in Streamlit
    st.pyplot(fig)

# Streamlit App
st.set_page_config(page_title="Super-Resolution & Object Detection", layout="wide")
st.title("🌍Super-Resolution & Object Detection on Satellite Imagery📡")


# Sidebar for model selection
with st.sidebar:
    st.header("Model Selection")
    sr_model_type = st.selectbox("Choose a Super-Resolution Model:", ["SRGAN","SRResNet"])
    yolo_model_type = st.selectbox("Choose an Object Detection Model:", ["Yolov11-OBB", "Yolov11"])

    # Load models based on selection
    if sr_model_type == "SRResNet":
        sr_model = torch.load(srresnet_checkpoint, weights_only=False)["model"].to(device)
        sr_model.eval()
    elif sr_model_type == "SRGAN":
        sr_model = torch.load(srgan_checkpoint, weights_only=False)["generator"].to(device)
        sr_model.eval()

    if yolo_model_type == "Yolov11-OBB":
        yolo_model = YOLO("YOLO/obb.pt").to(device)
    elif yolo_model_type == "Yolov11":
        yolo_model = YOLO("YOLO/yolov11-best.pt").to(device)
    yolo_model.eval()

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    hr_image = Image.open(uploaded_file)

    # Display the images with added titles and a little style
    with st.expander("Super-Resolution Results"):
        fig1, sr_img, lr_img = visualize_sr_plot(hr_image, sr_model,halve=False)
        st.pyplot(fig1)

    # YOLO Detection and Comparison
    with st.expander("Object Detection Results"):
        detections = yolo_model(hr_image)
        fig2 = plot_detections(hr_image, detections, move=10, color='red')
        st.pyplot(fig2)

    # Compare low-res and high-res detections
    with st.expander("Detection Comparison - Low Resolution vs Super Resolution"):
        detections_1 = yolo_model(lr_img)
        detections_2 = yolo_model(sr_img)
        if yolo_model_type == "Yolov11-OBB":
            compare_detections_obb(detections_1, detections_2, lr_img, sr_img)
        else:
            compare_detections_yolo(detections_1, detections_2, lr_img, sr_img)