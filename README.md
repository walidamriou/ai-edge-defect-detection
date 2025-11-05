# AI Edge Defect Detection

This project uses a YOLO-based model for **edge AI defect detection** on images, classifying them as `OK` or `NOK`.

## Installation

1. **Upgrade pip** to ensure compatibility:

```bash
python3 -m pip install --upgrade pip
```

2. **Install the Ultralytics YOLO library**:

```bash
python3 -m pip install ultralytics
```

3. **Verify installation**:

```bash
python3 -c 'from ultralytics import YOLO; print("Ultralytics is installed!")'
```

Once installed, you can start using your trained YOLO classification model to predict defects in images.  

## Tools
This project includes several utility scripts for dataset preparation, training, and testing the model:
- **`create.py`** – Generate and prepare an acceptable dataset for training.  
- **`train.py`** – Train the YOLO model using the prepared dataset.  
- **`test1.py`** – Test a single image and generate a detailed report of predictions.  
- **`test2.py`** – Test a single image and quickly check if the prediction matches expectations.  
- **`test3.py`** – Test a batch or group of images and summarize results for all of them.  