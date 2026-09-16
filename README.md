# Supervised Learning Algorithms Comparison

Comparative analysis of multiple supervised learning algorithms applied to the Wine dataset from scikit-learn. Covers regression, binary classification, and multi-class classification problems.

## Overview

This project implements and compares different machine learning approaches on a real-world dataset:
- **Part I**: Regression (predicting alcohol content)
- **Part II**: Binary classification (wine type 0 vs 1)
- **Part III**: Multi-class classification (3 wine types)

## Dataset

**Wine Dataset** (scikit-learn)
- 178 observations
- 13 chemical attributes
- 3 wine classes

## Algorithms Implemented

**Regression:**
- Linear Regression
- Support Vector Regression (SVR)

**Classification:**
- Logistic Regression
- Linear SVM
- Decision Tree (CART)
- Random Forest
- Non-Linear SVM (RBF kernel)
- Neural Networks (MLP)

## Repository Content

- **Report.pdf**: Complete analysis with results and visualizations
- **Code/**: Source code
  - `supervised_learning_wine.py`: Full implementation and experiments

## Key Results

| Problem | Best Model | Accuracy/R² |
|---------|-----------|-------------|
| Regression | Linear Regression | R² ≈ 0.62 |
| Binary Classification | SVM Linear | 100% |
| Multi-class | Random Forest / MLP | 100% |



## Features

- Hyperparameter tuning with GridSearchCV
- Decision boundary visualization
- Confusion matrices and classification reports
- Model comparison and performance evaluation

## Authors

**Chamen Rayane** & **Dahmani Mahdi**

Master 2 Applied Mathematics (Data Science & Decision Support)  
Université Abderrahmane Mira, Béjaïa, Algeria
