# Tomato Leaf Disease Detection Project

## Project Summary

This project focuses on building an AI-powered system for detecting and classifying tomato leaf diseases from images. The goal is to support early diagnosis of plant disease using computer vision, with a practical web-based interface that can be used by farmers, researchers, or agricultural technology teams.

## Problem

Plant diseases can significantly reduce crop quality and yield if they are not detected early. Manual inspection is often time-consuming and may not be accurate at scale. This project addresses that challenge by using deep learning to identify diseases automatically from leaf images.

## Solution

The system combines two stages:

1. YOLOv8 detects and localizes the leaf in the image.
2. A MobileNetV2 classifier predicts the disease class.

This hybrid approach improves the reliability of the prediction by focusing the classifier on the relevant part of the image.

## Technologies Used

- Python
- PyTorch
- Ultralytics YOLOv8
- Flask
- Pillow
- Matplotlib

## Key Results

The MobileNetV2 model achieved strong performance on the tomato leaf disease dataset:

- Accuracy: 98.00%
- Macro F1-score: 97.96%
- Weighted F1-score: 98.00%

These results show the effectiveness of transfer learning for plant disease recognition tasks.

## What This Project Demonstrates

- End-to-end machine learning workflow
- Computer vision and deep learning skills
- Model training, evaluation, and comparison
- Application deployment through a simple web interface
- Practical use case in agriculture and precision farming

## Project Structure

- mobile/: Flask dashboard and inference pipeline
- scripts/: training, evaluation, and preprocessing utilities
- reports/: project evaluation and comparison results
- dataset/: image dataset for training and testing

## Future Improvements

- Expand the dataset with more disease classes
- Improve real-time detection performance
- Add deployment to cloud platforms
- Integrate camera-based live prediction

## Why This Is a Strong Portfolio Project

This project is a strong addition to a data science, AI, or machine learning portfolio because it shows both technical depth and real-world impact. It demonstrates the ability to move from data preparation and model training to building a usable application.
