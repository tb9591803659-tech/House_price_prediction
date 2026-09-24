<div align="center">

# 🏡 House Price Prediction

### Production-Grade Tabular Regression & Valuation Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Pipeline-Production--Ready-00C853?style=for-the-badge)](#)
[![Validation](https://img.shields.io/badge/CV-5--Fold%20Leakage--Free-blue?style=for-the-badge)](#)

<p align="center">
  <b>An end-to-end machine learning system designed to predict residential house sale prices from structured tabular housing data using scikit-learn pipelines and ensemble gradient boosting.</b>
</p>

---

</div>

## 📌 Overview

Predicting property valuations requires handling wide tabular schemas comprising spatial, structural, chronological, and categorical attributes. This project builds a production-grade regression workflow on the **Ames Housing dataset** (1,460 observations, 80 input features).

A core focus of the system is operational robustness: all feature engineering transformations, missing value imputations, numerical scalings, and categorical encodings are packaged directly with the estimator inside a unified `sklearn.pipeline.Pipeline`. This eliminates training-serving skew, encapsulates feature dependencies, and enables clean single-call inference on raw tabular input.

---

## ⚡ Key Highlights

| Feature | Architectural Focus | Technical Implementation |
| :--- | :--- | :--- |
| **Unified Pipeline Architecture** | Training-Serving Parity | Encapsulates domain feature engineering, preprocessing, and model estimation in a single serializable object to guarantee identical transformation states during inference. |
| **Systematic Benchmarking** | Model Comparison & Selection | Implements and evaluates baseline dummy regressors, regularized linear models (Ridge), bagging ensembles (Random Forest), and sequential tree boosting (Gradient Boosting). |
| **Leakage-Free Validation** | Generalization & Reliability | Enforces strict 5-fold cross-validation alongside an isolated holdout test split to ensure zero data leakage across out-of-fold partitions. |
| **Hyperparameter Optimization** | Objective Metric Minimization | Leverages `GridSearchCV` to systematically sweep parameter spaces (tree depth, shrinkage rates, estimator counts) evaluated directly on validation RMSE. |
| **Granular Error Auditing** | Model Explainability & Edge Cases | Deconstructs residual distributions across distinct price bands, overall quality tiers, and neighborhood zones to detect bias and variance outliers. |

---