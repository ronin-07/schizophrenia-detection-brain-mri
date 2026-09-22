# Deep Transfer Learning for Schizophrenia Detection using Brain MRI

This project investigates the use of deep transfer learning for automated schizophrenia detection using structural brain MRI scans.

The proposed approach uses an ImageNet-pretrained **VGG19** convolutional neural network together with additional fully connected layers to classify MRI images into schizophrenia and control groups. The project focuses on the **axial view** of brain MRI scans and applies image preprocessing and data augmentation to address the limited availability of schizophrenia MRI data.

The proposed extended VGG19 model achieved **90.9% accuracy** on the evaluated COBRE dataset.

---

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [MRI Data](#1-mri-data)
  - [Axial Slice Extraction](#2-axial-slice-extraction)
  - [Image Preprocessing](#3-image-preprocessing)
  - [Data Splitting and Augmentation](#4-data-splitting-and-augmentation)
  - [Transfer Learning](#5-transfer-learning)
  - [Proposed Extended VGG19 Model](#6-proposed-extended-vgg19-model)
  - [Training](#7-training)
  - [Evaluation](#8-evaluation)
- [Models Evaluated](#models-evaluated)
- [Results](#results)
  - [Hyperparameter Experiments](#hyperparameter-experiments)
  - [Model Comparison](#model-comparison)
  - [Comparison with Existing Work](#comparison-with-existing-work)
- [Visualizations](#visualizations)
- [Repository Structure](#repository-structure)
- [Implementation](#implementation)
- [Technologies Used](#technologies-used)
- [Publication](#publication)
- [Project Report](#project-report)
- [Authors](#authors)
- [Future Work](#future-work)
- [Disclaimer](#disclaimer)

---

## Overview

Schizophrenia is a complex psychiatric disorder that affects cognition, behavior, and perception. Diagnosis is primarily based on clinical assessment and interviews, and there is no single objective medical index that can independently establish a diagnosis.

This project explores the application of deep learning to structural brain MRI data for schizophrenia classification.

A key challenge is the relatively small amount of publicly available MRI data for schizophrenia. Training a deep neural network from scratch on such a dataset can lead to overfitting and poor generalization.

To address this, the project uses **transfer learning**, leveraging feature representations learned by CNNs pretrained on ImageNet.

Instead of using all three anatomical views of a 3D MRI scan, the project concentrates on the **axial view**, which provides a clear representation of subcortical and ventricular regions discussed in the project as relevant to schizophrenia-related analysis.

---

## Objectives

The main objectives of the project are:

- Investigate deep learning for schizophrenia detection using brain MRI.
- Analyze 3D MRI data and extract axial brain slices.
- Apply image preprocessing techniques to improve image differentiability.
- Increase the effective training data using data augmentation.
- Investigate transfer learning for classification with limited MRI data.
- Develop an extended VGG19-based classification model.
- Compare the proposed model with other transfer-learning CNN architectures.
- Evaluate performance using accuracy, precision, recall, F1-score, and ROC analysis.
- Investigate the effect of activation functions and image-processing filters on model performance.

---

## Dataset

The project uses the publicly available **COBRE (Center for Biomedical Research Excellence)** brain MRI dataset.

According to the project report, the dataset contains **146 subjects**:

| Characteristic | Value |
|---|---:|
| Total subjects/images reported | 146 |
| Control subjects | 72 |
| Schizophrenia subjects | 74 |
| Mean age | 36.97 ± 12.78 years |
| Age range | 18–65 years |
| Female | 37/146 (25.34%) |
| Acquisition period | 2009–2013 |
| Scanner field strength | 3T |
| Subjects with excessive noise | 1 |
| Strict schizophrenia diagnosis | 74 |

The original MRI dataset is **not included in this repository**.

The notebook expects the MRI data to be organized into schizophrenia (`YES`) and control (`NO`) subject directories.

### Data privacy and repository size

MRI scans are not uploaded to GitHub. The repository contains the implementation, documentation, and project visualizations rather than the original medical imaging dataset.

---

## Methodology

The overall workflow developed in the project is:

```text
3D Brain MRI
      |
      v
Axial Slice Extraction
      |
      v
Image Preprocessing
      |
      +--> Thresholding
      |
      +--> Smoothing / Blurring
      |
      v
Data Augmentation
      |
      v
Train / Validation / Test Split
      |
      v
Transfer Learning CNN
      |
      +--> VGG16
      +--> ResNet101
      +--> EfficientNetB0
      +--> Extended VGG19
      |
      v
Binary Classification
      |
      v
Performance Evaluation
```

### 1. MRI Data

The original MRI scans are 3D volumes stored in NIfTI format.

The implementation uses **NiBabel** to load the NIfTI volumes and convert them into NumPy arrays.

### 2. Axial Slice Extraction

The project analyzes the three anatomical orientations of the brain:

- Axial
- Sagittal
- Coronal

The proposed approach focuses on the **axial view**.

The project extracts axial slices from the 3D MRI volume and uses these images as inputs to the classification pipeline.

### 3. Image Preprocessing

The implementation applies preprocessing to the extracted images.

The notebook uses:

- Binary thresholding
- Average blurring / smoothing

The preprocessing function applies OpenCV thresholding followed by a `(5, 5)` average blur.

The project report also investigates different filtering approaches, including Gaussian blur and average blur.

### 4. Data Splitting and Augmentation

The subject-level data is divided into:

- Training data
- Validation data
- Test data

The notebook uses stratified splitting to maintain the class distribution.

For training and validation subjects, multiple axial images are generated/used per subject to increase the amount of training data.

The test set uses a single selected image per subject in the implementation.

This approach was used because the available dataset is relatively small.

### 5. Transfer Learning

The project evaluates several CNN architectures using transfer learning.

The proposed VGG19 model uses an **ImageNet-pretrained VGG19** feature extractor.

The convolutional base is frozen, allowing the additional classification layers to learn task-specific features while leveraging the pretrained representation.

### 6. Proposed Extended VGG19 Model

The proposed architecture extends the pretrained VGG19 network with additional classification layers.

The architecture used in the implementation is:

```text
Image
  |
  v
VGG19 Convolutional Base
(pretrained on ImageNet)
  |
  v
Flatten
  |
  v
Dropout (0.3)
  |
  v
Dense (128) + ReLU
  |
  v
Dropout (0.3)
  |
  v
Dense (64) + ReLU
  |
  v
Dropout (0.3)
  |
  v
Dense (32) + ReLU
  |
  v
Dense (1) + Sigmoid
  |
  v
Schizophrenia / Control
```

The report describes the proposed model as an extended VGG19 architecture with fully connected layers.

The project uses:

- **ReLU** activation in hidden layers
- **Sigmoid** activation for binary classification
- **Adam** optimizer
- **Binary cross-entropy** loss

### 7. Training

The implementation trains the model with:

- Batch size: **32**
- Maximum epochs: **100**
- Optimizer: **Adam**
- Loss: **Binary cross-entropy**
- Early stopping based on validation accuracy
- Patience: **20 epochs**
- Best model weights restored after early stopping

The VGG19 convolutional base is frozen during the proposed model training.

### 8. Evaluation

The project evaluates the classification models using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC curves
- Loss curves
- Confusion matrix analysis

These metrics are used to compare the proposed model with the other CNN architectures and with selected existing approaches reported in the project literature review.

---

## Models Evaluated

The project investigates the following transfer-learning architectures:

| Model | Role |
|---|---|
| VGG16 | Baseline transfer-learning CNN |
| ResNet101 | Deep residual CNN |
| EfficientNetB0 | Efficient CNN architecture |
| Extended VGG19 | Proposed model |

---

## Results

The proposed extended VGG19 model achieved approximately **90.9% accuracy** on the evaluated dataset.

### Hyperparameter Experiments

The project report evaluates different activation functions and image-processing filters.

| Experiment | Configuration | Accuracy | Precision | Recall | F1-score |
|---|---|---:|---:|---:|---:|
| Activation function | ReLU | 0.91 | 0.90 | 0.91 | 0.90 |
| Activation function | Tanh | 0.77 | 0.81 | 0.81 | 0.78 |
| Activation function | Sigmoid | 0.71 | 0.66 | 0.63 | 0.63 |
| Image-processing filter | Gaussian Blur | 0.72 | 0.81 | 0.81 | 0.75 |
| Image-processing filter | Average Blur | 0.91 | 0.90 | 0.91 | 0.90 |

### Model Comparison

The project compares the proposed extended VGG19 model with other transfer-learning architectures.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| VGG16 | 0.71 | 0.63 | 0.63 | 0.66 |
| ResNet101 | 0.77 | 0.72 | 0.76 | 0.72 |
| EfficientNetB0 | 0.81 | 0.81 | 0.81 | 0.78 |
| **Extended VGG19** | **0.91** | **0.90** | **0.916** | **0.90** |

### Comparison with Existing Work

The project report compares the proposed model with selected approaches from the literature using the COBRE dataset.

| Method | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Zheng et al. | 0.87 | 0.87 | 0.890 | 0.85 |
| Latha et al. | 0.90 | 0.93 | 0.875 | 0.89 |
| **Proposed Extended VGG19** | **0.91** | **0.90** | **0.916** | **0.90** |

The project report states that the proposed model improved accuracy by approximately **1–3 percentage points** relative to the compared existing models.

---

## Visualizations

The project contains visual material documenting the complete methodology and experiments.

### Project and Dataset Figures

- Number of patients receiving treatment for schizophrenia by gender and year
- Hospitalizations for schizophrenia by age group
- Eight brain regions
- System workflow
- Axial, sagittal, and coronal MRI views
- Subcortical region in the axial view
- Brain images at different time intervals
- Preprocessing pipeline

### Model Figures

- VGG19 architecture
- Proposed extended VGG19 architecture
- Model summary
- Transfer-learning training process

### Experimental Figures

- VGG16 ROC and loss curves
- ResNet101 ROC and loss curves
- EfficientNetB0 ROC and loss curves
- Extended VGG19 ROC and loss curves
- Accuracy comparison
- Precision comparison
- Recall comparison
- F1-score comparison

---

## Repository Structure

The original project folder contains the following material:

```text
Final Major Project/
|
+-- Code/
|   +-- project work code.ipynb
|
+-- Images/
|   +-- Figure 1 Number of patients receiving treatments...
|   +-- Figure 2 Hospitalizations for schizophrenia...
|   +-- Figure 3 Diagram of eight brain regions...
|   +-- Figure 4 System workflow...
|   +-- Figure 5 Axial, Sagittal, Coronal views...
|   +-- Figure 6 Subcortical region...
|   +-- Figure 7 Images of brain at different time intervals...
|   +-- Figure 8 Preprocessing pipeline...
|   +-- Figure 9 VGG-19 Architecture...
|   +-- Figure 10 Proposed deep neural network...
|   +-- Figure 11 Model summary...
|   +-- Figure 12 Transfer learning training process...
|   +-- Figure 13 ROC and loss curve for VGG16...
|   +-- Figure 14 ROC and loss curve for ResNet101...
|   +-- Figure 15 ROC and loss curve for EfficientNetB0...
|   +-- Figure 16 ROC and loss curve for extended VGG19...
|   +-- Figure 17 Comparison of accuracy...
|   +-- Figure 18 Comparison of Precision...
|   +-- Figure 19 Comparison of Recall...
|   +-- Figure 20 Comparison of F1-score...
|
+-- Project Paper.docx
+-- Project Paper.pdf
+-- Project Report.docx
+-- Project Report.pdf
```

For the GitHub version, the project can be organized more cleanly as:

```text
schizophrenia-detection-brain-mri/
|
+-- README.md
|
+-- notebook/
|   +-- project_work_code.ipynb
|
+-- results/
|   +-- preprocessing_pipeline.png
|   +-- system_workflow.png
|   +-- vgg19_architecture.png
|   +-- proposed_vgg19_model.png
|   +-- model_summary.png
|   +-- transfer_learning_training.png
|   +-- vgg16_roc_loss.png
|   +-- resnet101_roc_loss.png
|   +-- efficientnetb0_roc_loss.png
|   +-- vgg19_roc_loss.png
|   +-- accuracy_comparison.png
|   +-- precision_comparison.png
|   +-- recall_comparison.png
|   +-- f1_comparison.png
|
+-- report/
|   +-- project_report.pdf
|
+-- requirements.txt
```

The original Word documents are not necessary for the GitHub repository if the PDF report is included.

The original MRI dataset should not be uploaded.

---

## Implementation

The main implementation is provided in:

```text
notebook/project_work_code.ipynb
```

The notebook currently uses the following main libraries:

```python
tensorflow
keras
numpy
opencv-python
nibabel
scikit-learn
matplotlib
imutils
```

The implementation was originally developed in a Google Colab environment and therefore contains Google Drive dataset paths.

Before running the notebook in another environment, update the dataset paths to point to the local COBRE dataset.

The original notebook expects subject data in a structure conceptually similar to:

```text
COBRE/
|
+-- YES/
|   +-- subject_1/
|   +-- subject_2/
|   +-- ...
|
+-- NO/
    +-- subject_1/
    +-- subject_2/
    +-- ...
```

The notebook reads NIfTI MRI files, extracts axial slices, applies preprocessing, and prepares the images for the CNN models.

---

## Reproducibility

The implementation includes random-seed initialization for reproducibility.

A default seed of:

```python
1530
```

is used in the notebook.

Random seeds are configured for:

- Python
- NumPy
- TensorFlow
- PyTorch

Exact results may still vary depending on the execution environment, library versions, hardware, and training conditions.

---

## Technologies Used

### Programming Language

- Python

### Deep Learning

- TensorFlow
- Keras
- VGG16
- VGG19
- ResNet101
- EfficientNetB0

### Medical Imaging

- NiBabel
- NIfTI MRI data

### Image Processing

- OpenCV

### Machine Learning

- Scikit-learn

### Numerical Computing

- NumPy

### Visualization

- Matplotlib

### Development Environment

- Jupyter Notebook
- Google Colab

---

## Publication

This project was published as a Springer book chapter:

### Deep Transfer Learning for Schizophrenia Detection using Brain MRI

**Authors:**

- Siddhant Mudholkar
- Amitesh Agrawal
- Dilip Singh Sisodia

**DOI:**

https://doi.org/10.1007/978-3-031-54547-4_6

---

## Limitations

The project has several limitations:

- The available schizophrenia MRI dataset is relatively small.
- The proposed approach focuses on the axial view rather than using the complete 3D MRI volume.
- Transfer learning is based on models originally pretrained on natural-image data.
- The model is evaluated on the available COBRE dataset and should not be assumed to generalize to other populations or datasets without further validation.
- The model is a research prototype and is not clinically validated.

---

## Future Work

Potential future directions include:

- Increasing the amount and diversity of MRI training data.
- Evaluating the approach on additional schizophrenia MRI datasets.
- Investigating 3D CNN architectures.
- Combining information from axial, sagittal, and coronal views.
- Investigating multimodal MRI information.
- Combining MRI features with phenotypic or clinical information.
- Exploring additional transfer-learning architectures.
- Improving preprocessing and region-of-interest extraction.
- Performing external validation on independent datasets.

---

## Disclaimer

This project is intended for **research and educational purposes only**.

The model is **not a clinically validated diagnostic system** and should not be used for medical diagnosis, treatment decisions, or other clinical decision-making.

---

## Citation

If you use this work, please cite the published paper:

```text
Mudholkar, S., Agrawal, A., & Sisodia, D. S.
Deep Transfer Learning for Schizophrenia Detection using Brain MRI.
Springer.
https://doi.org/10.1007/978-3-031-54547-4_6
```

---

## Authors

**Siddhant Mudholkar**  
Department of Computer Science & Engineering  
National Institute of Technology Raipur

**Amitesh Agrawal**  
Department of Computer Science & Engineering  
National Institute of Technology Raipur

**Dilip Singh Sisodia**  
Department of Computer Science & Engineering  
National Institute of Technology Raipur
