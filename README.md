# Task 3: 3D CT Region-wise Feature Extraction & Similarity Analysis

## Overview

This task focuses on extracting deep features from tibia, femur, and background regions in a 3D knee CT volume. A pretrained 2D DenseNet121 was inflated to handle 3D inputs. Feature vectors from selected convolutional layers were extracted for each region, and cosine similarity was computed to measure inter-region representational similarity.


## Repository Structure
- src/ — Scripts and modules for segmentation, model conversion, feature extraction, and similarity computing.
- notebooks/ — Data exploration and analysis notebooks.
- results/ — Saved segmentation mask and similarity csv file
- main.py — Full pipeline script to run all tasks sequentially.

## Installation
Install required packages with:

pip install -r requirements.txt

## Usage
Run the full processing pipeline with:
python main.py

## Report
See report.pdf for a detailed explanation of the approach and results.
