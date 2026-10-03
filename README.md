# Deep Learning-Based Multi-Label Chest X-Ray Classification for Clinical Decision Support

A deep learning system that classifies chest X-ray images into 15 thoracic disease categories. Built as a capstone project for NCPS730 at Sol Plaatje University.

**Author:** Katlego Mashala (202470697)

**Supervisor:** Mrs Nthabiseng Modiba

**Year:** 2026

---

## Overview

This project builds a multi-label deep learning model that detects 15 different chest diseases from a single X-ray image. The model uses transfer learning with the ResNet50 architecture and a custom weighted loss function to handle severe class imbalance.

The trained model is deployed as a Streamlit web application that allows users to upload a chest X-ray and receive predictions with confidence scores and Grad-CAM heatmaps.

**Best result:** Validation AUC of 0.8843 on 15,000 training images.

---

## Repository Structure
```bash
ChestXray_System/
├── data/ # Chest X-ray images (not committed — see below)
├── models/
│ ├── app.py # Streamlit application
│ └── best_chest_xray_model.keras # Trained model file (not committed — see below)
├── src/
│ ├── chest-xray-notebook.ipynb # Kaggle training notebook
│ └── train.ipynb # Local GPU training notebook
├── .gitignore
├── requirements.txt
└── README.md
```
---

## Requirements

- Python 3.10 or later
- pip

All Python dependencies are listed in `requirements.txt`. To install them:

```bash
pip install -r requirements.txt

Main libraries used:
* TensorFlow 2.19
* NumPy
* Pandas
* OpenCV
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit
```
## Dataset Setup
This project uses the NIH ChestX-ray14 dataset. The dataset is not included in this repository because of its size (approximately 45 GB).

To set up the data:
1. Download the dataset from the official NIH source:
[PASTE NIH CHESTX-RAY14 DATASET LINK HERE]

2. Extract the image folders and place them inside the data/ folder.

3. Download the metadata file Data_Entry_2017_v2020.csv and place it in the src/ folder.

The dataset should look like this after extraction:
```bash
data/
├── images_001/
├── images_002/
├── images_003/
└── ...
```
## Running the Streamlit Application
To launch the web application:
streamlit run models/app.py

Then open the URL shown in the terminal (usually http://localhost:8501) in your browser.

The application allows you to:
* Upload a chest X-ray image (JPG or PNG)
* View predictions for all 15 disease categories
* See confidence scores for each prediction
* View a Grad-CAM heatmap showing which part of the X-ray influenced the prediction

The app runs on CPU by default and does not require a GPU.

## Running the Training Notebooks

To retrain the model from scratch:

Kaggle version:
Open src/chest-xray-notebook.ipynb on Kaggle or Jupyter. This notebook was used for training on the Kaggle cloud platform with a Tesla P100 GPU.

Local GPU version:
Open src/train.ipynb on a machine with a dedicated GPU. This notebook was used for the larger 15,000-image training run that achieved the best AUC.

Both notebooks assume the dataset is already available on disk. Update the dataset_path variable at the top of the notebook if your paths differ.

## Model Summary
Item	Value
Architecture	ResNet50 (pre-trained on ImageNet) + custom classification head
Input size	224 × 224 × 3
Output classes	15 (14 diseases + Normal)
Loss function	Custom Weighted Binary Cross-Entropy
Optimiser	Adam (learning rate 2e-6)
Best validation AUC	0.8843
Training images	15,000
Train/test split	80 / 20


## Diseases Detected
The model predicts one probability for each of the following 15 categories:
1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural Thickening
14. Hernia
15. Normal

## Notes
* This project is a proof-of-concept prototype developed for academic purposes. It is not intended for clinical deployment and must not replace professional medical diagnosis.
* The model requires further validation before it can be considered for use in a real healthcare setting.
* The full implementation details, methodology, and results are documented in the capstone project report.

## License
This project is submitted as part of the NCPS730 capstone module at Sol Plaatje University. All rights reserved by the author.