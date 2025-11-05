# -------------------------------------------------------------------
# Script: test2.py
# Description: This script test a single image and quickly check if the prediction matches expectations.   
# Author: Walid A.
# Date: 06/11/2025
# Usage: python3 ./test2.py
# -------------------------------------------------------------------
from ultralytics import YOLO

# Load your trained classification model
model = YOLO("runs/classify/trainxx/weights/best.pt")

# Predict a single image
results = model.predict("dataset/xxxxxxxxxxxxxxxxxx/XXX/XXX.bmp")

# Get the first (and only) image prediction
pred = results[0]

# Use the Probs attributes
class_idx = int(pred.probs.top1)           # index of the highest probability class
class_name = model.names[class_idx]        # 'NOK' or 'OK'
class_prob = float(pred.probs.top1conf)    # probability of the predicted class

# Print clean result
print(f"Predicted: {class_name} with probability {class_prob:.2f}")

# Optional OK/NOK check
if class_name == "OK":
    print("Image is OK ✅")
else:
    print("Image is NOT OK ❌")