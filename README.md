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

## 🛠️ Tech Stack & Tooling

<table>
  <thead>
    <tr>
      <th align="left">Component</th>
      <th align="left">Technology</th>
      <th align="left">Role & Application</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Core Language</b></td>
      <td>
        <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
      </td>
      <td>Base runtime environment, execution logic, and modular script orchestration.</td>
    </tr>
    <tr>
      <td><b>Data Manipulation & Computation</b></td>
      <td>
        <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas" />
        <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy" />
      </td>
      <td>Tabular schema wrangling, structured vectorization, linear transformations, and missingness audits.</td>
    </tr>
    <tr>
      <td><b>Machine Learning & Pipelines</b></td>
      <td>
        <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
      </td>
      <td>End-to-end transformation pipelines, feature encoders, ensemble regressors, and cross-validation harnesses.</td>
    </tr>
    <tr>
      <td><b>Model Serialization</b></td>
      <td>
        <img src="https://img.shields.io/badge/Joblib-4B8BBE?style=flat-square&logo=python&logoColor=white" alt="Joblib" />
      </td>
      <td>Binary artifact persistence for fitted pipelines, ensuring sub-millisecond cold-start loading for inference.</td>
    </tr>
    <tr>
      <td><b>Visualization & Diagnostics</b></td>
      <td>
        <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white" alt="Matplotlib" />
        <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square&logo=python&logoColor=white" alt="Seaborn" />
      </td>
      <td>Exploratory correlation matrices, distribution skewness checks, residual diagnostics, and error-band plots.</td>
    </tr>
  </tbody>
</table>

---

## 📊 Dataset Profile

The engine is built and benchmarked on the canonical **Ames Housing dataset**, a standard regression benchmark for evaluating residential valuation algorithms on rich, heterogeneous tabular data.

<div align="center">

| Metric | Specification | Context / Analytical Note |
| :--- | :--- | :--- |
| **Observation Volume** | `1,460` records | Complete single-family and residential property sales records |
| **Feature Dimensionality** | `80` features | Granular profile spanning structural, spatial, zoning, and chronological data |
| **Target Variable** | `SalePrice` | Right-skewed continuous valuation in USD ($) |
| **Supervision Paradigm** | Supervised Regression | Multi-variate continuous estimation on high-dimensional tabular data |

</div>

## 🔍 Data Understanding & Feature Typology

The dataset integrates physical measurements, discrete counts, spatial categorizations, and subjective quality appraisals. Deconstructing these attributes by their mathematical types dictates the precise transformation paths needed across the preprocessing pipeline.

---

### 🧩 Attribute Classifications & Processing Intent

<table>
  <thead>
    <tr>
      <th align="left">Feature Class</th>
      <th align="left">Sample Attributes</th>
      <th align="left">Characteristics</th>
      <th align="left">Pipeline Treatment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Numerical Features</b><br><i>(Continuous & Discrete)</i></td>
      <td>
        <code>GrLivArea</code>, <code>TotalBsmtSF</code>, <code>LotArea</code>, <code>YearBuilt</code>, <code>YearRemodAdd</code>, <code>GarageCars</code>, <code>TotalPorchSF</code>, <code>OverallQual</code>
      </td>
      <td>Physical dimensions, surface areas, timestamps, and integer counts directly amenable to geometric distances and scaling.</td>
      <td>
        Targeted domain imputations (zero/median), variance stabilization transformations, and standard/robust scaling.
      </td>
    </tr>
    <tr>
      <td><b>Categorical Features</b><br><i>(Nominal & Ordinal)</i></td>
      <td>
        <code>Neighborhood</code>, <code>KitchenQual</code>, <code>ExterQual</code>, <code>GarageType</code>, <code>MSZoning</code>, <code>SaleCondition</code>
      </td>
      <td>Discrete qualitative groupings, localized spatial clusters, and subjective quality tiers without intrinsic numeric representation.</td>
      <td>
        Deterministic structural encoding: One-Hot Encoding for unordered nominal features, and ordered integer/rank mapping for graded attributes.
      </td>
    </tr>
  </tbody>
</table>

> **Key Distinction:** Numerical features represent physical dimensions or counts that can directly undergo mathematical scaling. Categorical features represent discrete qualitative labels that require deterministic vector space projections before regression models can compute gradients or split criteria on them.

---

## ⚙️ Machine Learning Workflow

The end-to-end training and inference lifecycle is strictly decoupled to preserve test integrity while packaging the entire feature transformation state directly into the final serializable estimator.

```mermaid
flowchart TD
    %% Base styling
    classDef default fill:#181b20,stroke:#3b4354,stroke-width:1px,color:#e6edf3;
    classDef highlight fill:#1f2937,stroke:#38bdf8,stroke-width:2px,color:#38bdf8;
    classDef pipeline fill:#111827,stroke:#6366f1,stroke-width:2px,color:#e0e7ff;
    classDef terminal fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#a7f3d0;

    A[Raw Ames Housing Dataset<br/><code>1,460 rows × 80 columns</code>] --> B[Data Understanding & Inspection]
    B --> C[Exploratory Data Analysis<br/><code>Skewness, Correlations, Missingness</code>]
    C --> D[Strict Train / Test Split<br/><code>80% Train | 20% Isolated Holdout</code>]
    
    D --> E[ML Pipeline Construction]

    subgraph Pipeline [" Unified Scikit-Learn Pipeline Object "]
        E1[Custom Domain Feature Engineering<br/><code>Composite SF, Temporal Age, Interaction Ratios</code>] --> E2[ColumnTransformer: Numerical Sub-Branch<br/><code>Median Imputation + Standard/Robust Scaling</code>]
        E2 --> E3[ColumnTransformer: Categorical Sub-Branch<br/><code>Constant Imputation + One-Hot Encoding</code>]
        E3 --> E4[Model Estimator<br/><code>Ridge | Random Forest | Gradient Boosting</code>]
    end

    E --> E1
    E4 --> F[Systematic Model Benchmarking<br/><code>Baseline vs. Linear vs. Bagging vs. Boosting</code>]
    F --> G[5-Fold Cross-Validation<br/><code>Out-of-Fold Leakage-Free Validation</code>]
    G --> H[Hyperparameter Tuning<br/><code>GridSearchCV on Validation RMSE</code>]
    H --> I[Final Tuned Pipeline Assembly]
    I --> J[Holdout Evaluation & Error Auditing<br/><code>Residual Distributions by Quality & Neighborhood</code>]
    J --> K[Feature Importance & Sensitivity Extraction]
    K --> L[Model Serialization<br/><code>joblib.dump(pipeline, 'model.joblib')</code>]
    L --> M([Production Inference Engine<br/><code>predict.py --input raw_data.csv</code>]):::terminal

    class A,D highlight;
    class Pipeline pipeline;
    ```