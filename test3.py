# -------------------------------------------------------------------
# Script: test3.py
# Description: This script test a batch or group of images and summarize results for all of them.   
# Author: Walid A.
# Date: 06/11/2025
# Usage: python3 ./test3.py
# -------------------------------------------------------------------
from ultralytics import YOLO
from pathlib import Path

model = YOLO("runs/classify/trainxx/weights/best.pt")

image_folder = Path("dataset/xxxxxxxxxxxxxxxxxx/XXX")  # يمكن تغييره لمجلد آخر أو NOK+OK
image_paths = list(image_folder.glob("*.bmp"))  # كل ملفات bmp في المجلد

for img_path in image_paths:
    results = model.predict(str(img_path))
    pred = results[0]

    class_idx = int(pred.probs.top1)
    class_name = model.names[class_idx]
    class_prob = float(pred.probs.top1conf)

    print(f"{img_path.name}: Predicted {class_name} with probability {class_prob:.2f}")

    if class_name == "OK":
        print("✅ Image is OK")
    else:
        print("❌ Image is NOT OK")