# -------------------------------------------------------------------
# Script: test1.py
# Description: This script test a single image and generate a detailed report of predictions. 
# Author: Walid A.
# Date: 06/11/2025
# Usage: python3 ./test1.py
# -------------------------------------------------------------------
from ultralytics import YOLO

model = YOLO("runs/classify/trainxx/weights/best.pt")

result = model.predict("dataset/xxxxxxxxxxxxxxxxxx/XXX/XXX.bmp")
print(result)