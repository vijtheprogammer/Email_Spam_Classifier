# 📧 Email Spam Classifier

This project implements an email spam classifier using **Logistic Regression** and **TF-IDF (Term Frequency-Inverse Document Frequency)** for feature extraction. The model is trained on a dataset of emails labeled as 'spam' or 'ham' (not spam), and its performance is evaluated using various metrics and visualizations.

---

## 📑 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results and Visualizations](#results-and-visualizations)
- [Saved Model and Feature Extractor](#saved-model-and-feature-extractor)
- [Custom Input Testing](#custom-input-testing)
- [License](#license)

---

## 🧠 Project Overview

The goal of this project is to build a machine learning model that can accurately classify incoming emails as either **"spam"** or **"not spam" (ham)**. This includes:

- Data preprocessing
- TF-IDF-based feature extraction
- Training a Logistic Regression model
- Model evaluation and visualization

---

## ⚙️ Features

- **Data Loading & Cleaning**: Handles missing values and prepares text data.
- **Label Conversion**: Converts 'spam'/'ham' labels into binary format.
- **Data Splitting**: Training and testing dataset separation.
- **Feature Extraction**: Applies TF-IDF vectorization to email content.
- **Model Training**: Uses Logistic Regression for classification.
- **Model Evaluation**: Accuracy scores and several visualizations.
- **Visualization Includes**:
  - Label distribution
  - Important positive/negative features
  - Confusion Matrix
  - ROC Curve & AUC Score
- **Model Persistence**: Saves model and vectorizer using `pickle`.
- **Custom Input Prediction**: Classifies new email messages.

---

## 🧰 Requirements

Install the following Python libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
