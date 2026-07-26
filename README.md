# Tomato Leaf Disease Detection System

An AI-powered computer vision project for detecting and classifying tomato leaf diseases from images. This project combines object detection and image classification into a practical web-based application that can be used for plant health monitoring and precision agriculture.

## Why this project matters

Early detection of plant disease can help farmers reduce crop losses, improve yield quality, and make faster decisions. This project demonstrates how deep learning can support real-world agricultural monitoring through an end-to-end image-based solution.

## Project overview

The system works in two stages:

1. YOLOv8 detects and localizes the leaf in the input image.
2. A MobileNetV2 classifier identifies the disease among multiple tomato leaf classes.

This hybrid approach makes the solution more practical than a simple image classifier because it can focus on the relevant region of the image before making a prediction.

## Key features

- Image-based disease detection for tomato leaves
- Hybrid pipeline using object detection + classification
- Flask web dashboard for uploading images and viewing results
- Support for multiple disease classes
- Model comparison workflow for CNN vs MobileNetV2
- Reproducible training and evaluation scripts

## Dataset

The project uses a tomato leaf disease dataset with the following classes:

- Blight
- Healthy
- Leaf Mold
- Septoria
- Virus

## Models used

- YOLOv8 for leaf detection/localization
- MobileNetV2 for disease classification
- Custom CNN baseline for comparison and evaluation

## Results

The classification experiments showed strong performance for MobileNetV2 on the tomato leaf dataset.

- MobileNetV2 accuracy: 98.00%
- MobileNetV2 macro F1-score: 97.96%
- MobileNetV2 weighted F1-score: 98.00%

These results highlight the effectiveness of transfer learning for plant disease recognition tasks.

## Tech stack

- Python
- PyTorch
- Ultralytics YOLOv8
- Flask
- Pillow
- Matplotlib
- Jupyter/Notebook-based experimentation support

## Project structure

- mobile/: Flask app, model inference helpers, training scripts, and web templates
- scripts/data_prep/: dataset preparation and balancing utilities
- scripts/evaluation/: evaluation and visualization scripts
- scripts/training/: YOLO training entrypoint
- reports/: evaluation reports and results summaries
- dataset_5class/: classification dataset
- leaf-desease-detection.v1i.yolov8/: YOLO dataset configuration and labels
- runs/: training output and experiment artifacts

## How to run locally

### 1. Install dependencies

```bash
pip install -r mobile/requirements.txt
```

### 2. Start the web app

```bash
python mobile/app.py
```

Then open the app in your browser at:

```text
http://127.0.0.1:5000/
```

### 3. Train the MobileNet model

```bash
python mobile/train_mobilenet.py
```

### 4. Train YOLOv8

```bash
python scripts/training/train_yolov8.py
```

### 5. Evaluate models

```bash
python scripts/evaluation/evaluate_models.py --model mobilenet --weights mobile/mobilenet_model_best.pth
```


## License

This project is intended for educational and portfolio purposes.

## Contact

If you want to connect or discuss this project, feel free to reach out.
