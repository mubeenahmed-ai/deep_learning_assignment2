# AI600 Deep Learning - Assignment 2: Doodle Recognition

**Author:** Mubeen Ahmad  
**Roll No:** 25280101 (MS-AI)  
**Course:** AI600 - Deep Learning, Spring 2026

---

## Assignment Overview

This repository contains the complete implementation and analysis for Assignment 2, which focuses on:
* **Multi-Layer Perceptron (MLP) for Image Recognition:** Classifying $28 \times 28$ grayscale doodle sketches into 15 distinct categories.
* **Architectural Optimization:** Designing a "Wide-Funnel" MLP that maximizes accuracy while staying strictly under a **1,000,000 parameter** constraint.
* **Regularization Techniques:** Implementing BatchNorm, GELU activations, and strategic Dropout to bridge the generalization gap between training and validation data.
* **Error Analysis:** Utilizing Confusion Matrices to identify structural ambiguities in non-convolutional architectures.

---

## Repository Structure

Based on the local directory structure:

```text
.
├── 25280101_mubeen_dl_PA2.ipynb    # Complete Python/PyTorch implementation
├── champion_model_weights.pth      # Best model weights (80.17% Val Acc)
├── submission.txt                  # Comma-separated predictions for leaderboard
├── processed_data/                 # Dataset folder (x_train, y_train, test_images)
├── README.md                       # This file
└── figures/                        # Generated plots and analysis
    ├── champion_curves.png         # Training/Validation logs for Champion model
    ├── confusion_matrix.png        # Class-wise performance and error heatmaps
    ├── comparison_curves.png       # Comparative analysis of all architectures
    ├── pancake_curves.png          # Baseline 1: Wide/Shallow MLP analysis
    └── tower_curves.png            # Baseline 2: Deep/Narrow MLP analysis
