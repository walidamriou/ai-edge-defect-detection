# -------------------------------------------------------------------
# Script: create.py
# Description: This script prepares a dataset for YOLOv8 classification 
#              by splitting it into training and validation folders.
# Author: Walid A.
# Date: 06/11/2025
# Usage: python3 ./create.py
# -------------------------------------------------------------------
import os
import shutil
import random

src = 'dataset/xxxxxxxxxxxxxxxxxx'
dst = 'dataset/xxxxxxxxxxxxxxxxxx_split'

os.makedirs(dst, exist_ok=True)
for split in ['train', 'val']:
    for cls in ['OK', 'NOK']:
        os.makedirs(os.path.join(dst, split, cls), exist_ok=True)

for cls in ['OK', 'NOK']:
    images = os.listdir(os.path.join(src, cls))
    random.shuffle(images)
    split_idx = int(0.8 * len(images))  # 80% train, 20% val

    for i, img in enumerate(images):
        if i < split_idx:
            shutil.copy(os.path.join(src, cls, img), os.path.join(dst, 'train', cls, img))
        else:
            shutil.copy(os.path.join(src, cls, img), os.path.join(dst, 'val', cls, img))