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

---

### 🔄 Architectural Execution Phases

| Phase | Boundary | Key Responsibility |
| :--- | :--- | :--- |
| **I. Ingestion & Split** | Raw Data → Partition | Enforce hard separation via 80/20 train/test split prior to any aggregation, mean computation, or scaling to eliminate validation leakage. |
| **II. Pipeline Encapsulation** | `Pipeline` Object | Bundle engineered feature derivations, numerical standardizations, missing value replacements, and categorical sparse-matrix mappings into an atomic unit. |
| **III. Model Optimization** | CV Tuning Harness | Cross-validate candidates using 5-fold evaluation, minimizing validation Root Mean Squared Error (RMSE) across parameter grids. |
| **IV. Diagnostics & Serving** | Artifact Export | Audit error behavior on unseen holdout records, pull feature ranking attributions, and serialize the pipeline graph to disk for deterministic inference. |

---

## 🏗️ Project Architecture: Pipeline Encapsulation

### Why Encapsulation Matters

In conventional machine learning workflows, feature engineering (such as calculating property age or composite square footage) is frequently executed as an ad-hoc, global preprocessing script before feeding arrays into an estimator.

This anti-pattern introduces critical points of failure:

* **Training-Serving Skew:** Production inference scripts must duplicate every data transformation step in lockstep. Any drift in implementation leads to degraded or invalid predictions.
* **Schema Fragility:** Unencapsulated scripts fail silently when production payloads contain missing columns, reordered fields, or unseen categorical levels, resulting in dimension mismatches.
* **Information Leakage:** Computing summary statistics (e.g., column medians, scalers, one-hot vocabularies) across the full dataset prior to splitting leaks target and distribution properties into validation sets, inflating offline metrics.

---

### Failure Modes & Architectural Solutions

| Vulnerability | Naive Implementation | Encapsulated Pipeline Solution |
| :--- | :--- | :--- |
| **Statistical Leakage** | Scalers and imputers compute global metrics across all rows before train/test splitting. | Statistics are strictly learned inside `fit()` on training folds only; test folds are purely transformed via `transform()`. |
| **Categorical Mismatch** | New or missing category levels in inference cause matrix shape mismatches and runtime crashes. | `OneHotEncoder(handle_unknown='ignore')` keeps output tensor dimensions fixed regardless of unobserved categorical values. |
| **Operational Overhead** | Production environments require multi-stage script orchestration to prepare inputs. | Inference is atomic: passing raw input payloads directly into `pipeline.predict()` handles all derivations internally. |

---

## 🔎 Exploratory Data Analysis (EDA)

Exploratory diagnostics were performed prior to pipeline construction to audit record integrity, investigate missingness mechanisms, and assess target variance and covariate relationships.

---

### 1. Structural Health & Integrity Audit

* **Dataset Dimensions:** Confirmed initial shape of `1,460` records across `81` columns (80 input features + 1 target variable `SalePrice`).
* **Deduplication:** Audited full-row uniqueness; verified `0` duplicate rows across the index.
* **Feature Typing:** Identified `38` numerical features (continuous, discrete, temporal) and `43` categorical attributes (nominal, ordinal).

---

### 2. Missing Value Mechanics & Sparsity Analysis

Missing values were audited to separate random data-collection voids (**MCAR / MAR**) from systemic structural absences (**MNAR**):
> **Domain Insight (Structural Absences):** Nulls in categorical amenity features (e.g., `PoolQC`, `FireplaceQu`, `BsmtQual`, `GarageType`) are informative structural indicators meaning *"amenity does not exist on property"*, rather than dropped or corrupted telemetry. They must be explicitly imputed with constant tokens (e.g., `"None"`) rather than dropped or assigned mode values.

---

### 3. Target Variable Distribution (`SalePrice`)

An inspection of the target variable reveals significant positive right-tail skewness and kurtosis driven by high-value transactions:
* **Median Sale Price:** ~`$163,000`
* **Mean Sale Price:** ~`$180,921`
* **Analytical Impact:** The rightward divergence between mean and median highlights severe positive skewness. To stabilize heteroscedastic residuals, logarithmic transformation options ($\log(1 + y)$) were isolated for downstream modeling.

---

### 4. Bivariate Feature Correlations

Pearson correlation checks ($r$) against `SalePrice` identified the primary linear drivers of home market valuation:
* **Quality Dominance:** `OverallQual` ($r = 0.79$) serves as the strongest single predictive signal, indicating that physical finish quality dictates valuation ceilings.
* **Dimensional Footprint:** Sizing features (`GrLivArea` at $r = 0.71$, `TotalBsmtSF` at $r = 0.61$) display massive co-dependency with price, serving as essential anchors for domain-engineered composite area features.

---

