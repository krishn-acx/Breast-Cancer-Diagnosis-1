# Breast Cancer Diagnosis Classifier

A machine learning project for predicting breast cancer diagnosis (Malignant or Benign) using the Wisconsin Breast Cancer Diagnostic Dataset.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

## 📊 Project Overview

This project implements and compares multiple machine learning algorithms to classify breast cancer tumors as malignant or benign based on diagnostic features. The notebook demonstrates a complete ML pipeline including data exploration, preprocessing, model training, evaluation, and comparison.

### Key Features

- **Comprehensive EDA**: Detailed exploratory data analysis with visualizations
- **Multiple ML Models**: Implementation of K-Nearest Neighbors (KNN), Logistic Regression, and Naive Bayes
- **Model Evaluation**: Thorough performance analysis using accuracy, confusion matrices, classification reports, and ROC curves
- **Cross-Validation**: Robust performance estimation using k-fold cross-validation
- **Feature Analysis**: Understanding which features contribute most to diagnosis prediction

## 🎯 Dataset

**Dataset**: [Wisconsin Breast Cancer Diagnostic Dataset](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic) from UCI Machine Learning Repository

- **Total Samples**: 569
- **Features**: 30 (computed from digitized images of fine needle aspirate of breast mass)
- **Target Classes**: 
  - Malignant (M): 212 cases
  - Benign (B): 357 cases

### Features Include

Three sets of 10 features each (mean, standard error, and "worst" values):
- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/breast-cancer-diagnosis.git
cd breast-cancer-diagnosis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Open the Jupyter notebook:
```bash
jupyter notebook breast_cancer_diagnosis-1.ipynb
```

## 📝 Notebook Structure

1. **Importing Libraries**: Setup of required Python libraries
2. **Loading Dataset**: Fetching data from UCI ML Repository
3. **Data Exploration**: Initial data analysis and statistics
4. **Exploratory Data Analysis**: Comprehensive visualizations
5. **Data Preprocessing**: Feature scaling and train-test split
6. **Model Building**: Training KNN, Logistic Regression, and Naive Bayes
7. **Model Evaluation**: Performance metrics and comparison
8. **Cross-Validation**: Robust performance estimation

## 🤖 Models Implemented

### 1. K-Nearest Neighbors (KNN)
- Non-parametric instance-based learning algorithm
- Classifies based on nearest neighbors in feature space

### 2. Logistic Regression
- Linear model for binary classification
- Provides probabilistic predictions

### 3. Naive Bayes (Gaussian)
- Probabilistic classifier based on Bayes' theorem
- Assumes feature independence

## 📈 Results

The notebook includes detailed performance metrics for each model:
- **Accuracy scores** on test data
- **Confusion matrices** showing true/false positives and negatives
- **Classification reports** with precision, recall, and F1-scores
- **ROC curves** and AUC scores
- **Cross-validation scores** for model reliability

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **scikit-learn**: Machine learning models and tools
- **ucimlrepo**: UCI dataset repository access

## 📊 Visualizations

The notebook includes various visualizations:
- Target class distribution
- Feature correlation heatmaps
- Pairplot analysis of key features
- Confusion matrices
- ROC curves
- Cross-validation score distributions

## 🎓 Learning Outcomes

This project demonstrates:
- Data preprocessing and feature engineering
- Exploratory data analysis techniques
- Implementation of multiple ML algorithms
- Model evaluation and comparison strategies
- Cross-validation for robust performance estimation
- Interpretation of medical diagnostic data

## 🙏 Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/) for providing the dataset
- Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian for creating the dataset
- scikit-learn community for excellent documentation and tools
