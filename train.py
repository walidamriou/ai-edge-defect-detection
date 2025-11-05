# -------------------------------------------------------------------
# Script: train.py
# Description: This script trains a YOLOv8 classification model on a custom defect detection dataset.
# Author: Walid A.
# Date: 06/11/2025
# Usage: python3 ./train.py
# -------------------------------------------------------------------

from ultralytics import YOLO

# Load YOLOv8 classification model
model = YOLO("yolov8n-cls.pt")

# Train on your dataset
model.train(
    data='dataset/xxxxxxxxxxxxxxxxxx_split', # Path to your dataset YAML or folder split
    epochs=20, # Number of times the model will iterate over the entire dataset
    imgsz=224, # Resize all training images to 224x224 pixels
    batch=8, # Number of images processed in one training step
    project='runs/classify', # Folder to save training results
    name='train', # Name for this specific training run
    pretrained=True # Start from pretrained weights for faster convergence
)