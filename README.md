# Face Classification KNN Notebook

This project implements a **face image classification pipeline** using a **K-Nearest Neighbors (KNN)** classifier developed from scratch.  
The notebook also includes **image preprocessing**, **data augmentation**, and **model evaluation**, applied to a dataset of face images.

The pipeline is designed for educational purposes and focuses on understanding each step of image classification without relying on black-box models.

---

## Disclaimer

**Important:**  
This repository contains a **functional but non-modularized notebook**.  
Due to time constraints, the code is shared in its raw notebook form, prioritizing logic demonstration over code structure.

---

## Objectives

- Practice building a **KNN Classifier manually**, without using libraries like `sklearn.neighbors`
- Apply **image preprocessing techniques** (grayscale conversion, normalization)
- Use **Data Augmentation** to expand the dataset and prevent overfitting
- Explore the **Elbow Method** to select the optimal number of neighbors (K)
- Test the pipeline end-to-end on a **face image dataset**

---

## Features

- **Image Loading**
    - Loads face images from a directory structure into NumPy arrays
    - Supports both RGB and grayscale processing

- **Data Augmentation**
    - Uses `ImageDataGenerator` for:
        - Rotation
        - Width and height shift
        - Shear
        - Zoom
    - Simulates a larger dataset for training

- **Manual KNN Implementation**
    - Computes Euclidean distances manually
    - Predicts the most common label among K nearest neighbors
    - Fully transparent implementation (no pre-built KNN library)

- **Model Evaluation**
    - Calculates accuracy manually
    - Prints per-image predictions (can be extended to confusion matrix)

- **Elbow Method for K selection**
    - Uses `KneeLocator` to determine the optimal number of neighbors (K)
    - Plots accuracy vs. K to visualize the elbow point

---

## Technologies Used

- Python 3.x
- NumPy
- TensorFlow / Keras (only for data augmentation)
- PIL (image processing)
- Matplotlib / Seaborn (optional, for plots)
- scikit-learn (only for comparison or auxiliary tasks, not for KNN itself)
- kneed (for Elbow Method)

---

## How to Run

### 1. Clone the repository

```
git clone https://github.com/your-username/face-classification-knn-notebook.git
cd face-classification-knn-notebook
```

### 2. Install dependencies

```
pip install numpy tensorflow pillow matplotlib scikit-learn kneed
```

### 3. Run the notebook

Open `faces-classification-knn.ipynb` in Jupyter Notebook or Google Colab and execute all cells.

---

## Dataset

The dataset used is a **face image dataset** with the following structure:

```
data/
    train/
        person_1/
            img1.jpg
            img2.jpg
            ...
        person_2/
            img1.jpg
            ...
    test/
        person_1/
            imgX.jpg
        person_2/
            imgY.jpg
```

You can replace this dataset with your own face images following the same folder structure.

---

## Notebook Structure

| Section | Description |
|----------|-------------|
| **Data Import** | Loads the images into NumPy arrays |
| **Grayscale Conversion** | Converts RGB to grayscale if selected |
| **Data Augmentation** | Expands the dataset using rotation, shift, zoom, etc. |
| **Manual KNN Classifier** | Implements KNN from scratch (Euclidean distance + voting) |
| **Accuracy Calculation** | Compares predictions with ground truth |
| **Elbow Method** | Determines the optimal K using `KneeLocator` |

---

## Example Outputs

- **Augmented Images:** Randomized transformations applied to faces  
- **Distance Matrix:** Manually calculated Euclidean distances between image vectors  
- **Elbow Plot:** Visual representation of the best K

---

## Next Steps (Future Improvements)

- Refactor the code into reusable functions or classes
- Add confusion matrix and precision/recall metrics
- Save and load models for reusability (optional)
- Integrate with OpenCV for real-time prediction (optional)
- Package as a Python module or API

---

