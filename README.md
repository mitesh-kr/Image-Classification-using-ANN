# Neural Network from Scratch (Without PyTorch or TensorFlow) - Assignment 3 (CSL7620)

## Overview
This project implements a **feedforward neural network** from scratch in Python uisng Numpy only following the specifications given in **CSL7620: Machine Learning** (AY 2023-24, Semester I). The network is trained on a **multi-class classification dataset**.

## Dataset
- The dataset consists of **70,000 images** belonging to **10 different object categories**.
- The dataset link: [Dataset](https://drive.google.com/file/d/1yWw3SL3rMgnSlP2R3tJMueJkIiNZ6X5U/view?usp=sharing)

## Network Architecture
- **Input Layer**: Size = Number of input features (based on dataset)
- **Hidden Layer 1**: 128 neurons (Activation: Sigmoid)
- **Hidden Layer 2**: 64 neurons (Activation: Sigmoid)
- **Hidden Layer 3**: 32 neurons (Activation: Sigmoid)
- **Output Layer**: Number of classes (Activation: Softmax)

## Implementation Details
- **Bias**: Set to **1**.
- **Train-Test Splits**: 70:30, 80:20, 90:10 (randomized).
- **Activation Functions**:
  - Hidden Layers: **Sigmoid**
  - Output Layer: **Softmax**
- **Loss Function**: **Cross-entropy**
- **Optimization**: **Gradient Descent**
- **Training**: **25 epochs**, tracking loss and accuracy.
- **Evaluation**:
  - Accuracy & loss **plotted per epoch**.
  - **Confusion matrix** for all training splits.
  - **Total Parameters**:
    - **Trainable Parameters**: `111,146`
    - **Non-Trainable Parameters**: `7,840`

## Results

for 70:30 split test accuracy = 84.55 %

for 80:20 split test accuracy = 85.94 %

for 90:10 split test accuracy = 86.46 %



## Installation
To set up the environment, install the required dependencies:

```bash
git clone https://github.com/mitesh-kr/Image-Classification-using-ANN.git
cd Image-Classification-using-ANN
pip install requirement.txt
