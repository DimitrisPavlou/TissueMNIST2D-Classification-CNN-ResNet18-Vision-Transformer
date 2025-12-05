# 🧬 TissueMNIST2D Classification — CNN, ResNet18 & Vision Transformer

This repository focuses on medical image classification using the **MedMNIST** suite of datasets.  
The selected dataset is **TissueMNIST2D**, consisting of 2D fluorescence microscopy tissue images.  
Three main architectures were implemented and compared: a **custom CNN**, **ResNet18**, and a **Vision Transformer (ViT)**.  
Additional experiments include **transfer learning** using pretrained weights. The goal of the project is to improve the results 
based on the public benchmarks for this specific dataset. 

More info for MedMnist: https://medmnist.com
---

## 📂 Dataset: TissueMNIST2D

- Part of the **MedMNIST v2** collection  
- Contains **28×28, 64x64 and 224x224 grayscale tissue images**
- In this project, 64x64 images were used.
- 8 tissue-type classes  
- Automatically downloaded using the MedMNIST Python API

---

## 🧠 Models Implemented

### **1. Custom CNN**
A lightweight baseline convolutional neural network built from scratch.

### **2. ResNet18**
A residual network architecture evaluated in two modes:
- **Training from scratch**
- **Transfer learning** (ImageNet-pretrained backbone, frozen except classifier)

### **3. Vision Transformer (ViT)**
A transformer-based model adapted for 28×28 images using:
- Patch embeddings  
- Multi-head self-attention  
- From scratch implementation of a small ViT
- Transfer Learning using a pre-trained ViT

---

## 🔁 Transfer Learning

Transfer learning experiments include:
- ResNet18 pretrained on **ImageNet**
- ViT pretrained weights (via `timm`, if available)

Only the classification head is trained, while the feature extractor remains frozen.

---

## 🏋️ Training Details

- Framework: **PyTorch**
- Optimizer: **AdamW**
- Loss: **Cross-Entropy**
- Standard preprocessing and normalization  
- Evaluation performed using accuracy, confusion matrix and roc curves

---

## 📊 Results & Comparison

The project includes comparisons of:
- CNN vs ResNet18 vs Vision Transformer
- Scratch training vs Transfer learning  
- Training curves, validation accuracy, and qualitative evaluation

Visual results and logs are stored in the `results/` directory.

