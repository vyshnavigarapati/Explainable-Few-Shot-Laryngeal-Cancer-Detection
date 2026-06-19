# Explainable Few-Shot Laryngeal Cancer Detection

## Overview

This project is an AI-powered medical image classification system designed to detect laryngeal cancer from endoscopic images. The system classifies images into three categories: **Normal**, **Precancer**, and **Cancer**.

To improve transparency and trust in AI predictions, the project integrates **Explainable AI (XAI)** using **Grad-CAM**, which highlights the regions of the image that influenced the model's prediction.

---

## Problem Statement

Early detection of laryngeal cancer is critical for effective treatment. Manual diagnosis from endoscopic images requires medical expertise and can be time-consuming.

This project aims to assist healthcare professionals by automating image classification and providing visual explanations for model predictions.

---

## Objectives

- Detect laryngeal abnormalities from endoscopic images.
- Classify images into Normal, Precancer, and Cancer categories.
- Generate prediction confidence scores.
- Provide visual explanations using Grad-CAM.
- Build an interactive web interface using Streamlit.

---

## Technologies Used

- Python
- PyTorch
- OpenCV
- Streamlit
- Grad-CAM
- NumPy
- Pillow
- Matplotlib

---

## Dataset Structure

```
Dataset/
├── cancer/
├── normal/
└── precancer/
```

The dataset contains laryngeal endoscopic images grouped into three classes.

---

## Project Structure

```
LarynxProject/
│
├── Dataset/
│   ├── cancer/
│   ├── normal/
│   └── precancer/
│
├── models/
│   ├── best_model.pth
│   ├── accuracygraph.png
│   ├── confusionmatrix.png
│   └── gradcam_output.jpg
│
├── src/
│   ├── app.py
│   ├── predict.py
│   ├── gradcam.py
│   └── train.py
│
└── temp.jpg
```

---

## System Workflow

1. User uploads an endoscopic image.
2. The image is preprocessed.
3. The trained model (`best_model.pth`) is loaded.
4. The model predicts the image category.
5. Confidence score is calculated.
6. Grad-CAM generates a heatmap.
7. Results are displayed through the Streamlit interface.

---

## Explainable AI (Grad-CAM)

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize the areas of the image that contributed most to the model's prediction.

Benefits:

- Improves model interpretability.
- Increases trust in AI-assisted diagnosis.
- Helps understand prediction behavior.
- Supports medical decision-making.

---

## Features

- Real-time image upload.
- AI-based cancer detection.
- Three-class classification.
- Confidence score generation.
- Grad-CAM heatmap visualization.
- Interactive Streamlit application.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/vyshnavigarapati/Explainable-Few-Shot-Laryngeal-Cancer-Detection.git
```

Move to project directory:

```bash
cd Explainable-Few-Shot-Laryngeal-Cancer-Detection
```

Install required libraries:

```bash
pip install torch torchvision
pip install opencv-python
pip install streamlit
pip install numpy pillow matplotlib
```

---

## Run the Application

```bash
streamlit run src/app.py
```

Open the generated local URL in your browser.

---

## Sample Output

**Prediction:** Normal

**Confidence:** 70.57%

**Output:**
- Predicted Class
- Confidence Score
- Grad-CAM Heatmap

---

## Future Enhancements

- Improve model accuracy with larger datasets.
- Add cloud deployment.
- Support additional laryngeal diseases.
- Integrate SHAP and LIME explanations.
- Enable real-time video analysis.

---

## Conclusion

This project demonstrates the application of Deep Learning, Computer Vision, and Explainable AI in healthcare. By combining accurate classification with Grad-CAM visualization, the system provides an interpretable and user-friendly solution for laryngeal cancer detection.
