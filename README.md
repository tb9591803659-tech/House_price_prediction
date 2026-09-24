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
 * **Domain Insight (Structural Absences):** Nulls in categorical amenity features (e.g., `PoolQC`, `FireplaceQu`, `BsmtQual`, `GarageType`) are informative structural indicators meaning *"amenity does not exist on property"*, rather than dropped or corrupted telemetry. They must be explicitly imputed with constant tokens (e.g., `"None"`) rather than dropped or assigned mode values.

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

## 📐 Feature Engineering

To enrich the feature space with structural signals and spatial ergonomics, custom transformer logic was developed to compute domain-specific composite representations directly within the pipeline.

---

### 🧪 Engineered Formulations

<table>
  <thead>
    <tr>
      <th align="left">Derived Feature</th>
      <th align="left">Mathematical Definition</th>
      <th align="left">Domain Rationale</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Total Square Footage</b><br><code>TotalSF</code></td>
      <td>$$\text{TotalSF} = \text{TotalBsmtSF} + \text{1stFlrSF} + \text{2ndFlrSF}$$</td>
      <td>Consolidates all usable above-grade and below-grade living planes into a single unified volumetric metric, resolving multi-floor collinearity.</td>
    </tr>
    <tr>
      <td><b>Total Bathrooms</b><br><code>TotalBathrooms</code></td>
      <td>$$\text{TotalBathrooms} = \text{FullBath} + 0.5 \times \text{HalfBath} + \text{BsmtFullBath} + 0.5 \times \text{BsmtHalfBath}$$</td>
      <td>Weights half-baths (containing toilet and sink without bathing fixtures) at 0.5 to project diverse plumbing allocations onto an equivalent utility scale.</td>
    </tr>
    <tr>
      <td><b>Total Porch Area</b><br><code>TotalPorchSF</code></td>
      <td>$$\text{TotalPorchSF} = \text{OpenPorchSF} + \text{EnclosedPorch} + \text{3SsnPorch} + \text{ScreenPorch} + \text{WoodDeckSF}$$</td>
      <td>Aggregates fragmented exterior deck and patio attributes into an overall exterior living/leisure footprint.</td>
    </tr>
    <tr>
      <td><b>House Age at Sale</b><br><code>HouseAge</code></td>
      <td>$$\text{HouseAge} = \text{YrSold} - \text{YearBuilt}$$</td>
      <td>Replaces static calendar years with true physical age at time of transaction, directly capturing structural depreciation and wear.</td>
    </tr>
    <tr>
      <td><b>Remodel Age at Sale</b><br><code>RemodAge</code></td>
      <td>$$\text{RemodAge} = \text{YrSold} - \text{YearRemodAdd}$$</td>
      <td>Quantifies the elapsed time since the property's latest structural modernizations or architectural renovations.</td>
    </tr>
    <tr>
      <td><b>Average Area Per Room</b><br><code>TotalSFPerRoom</code></td>
      <td>$$\text{TotalSFPerRoom} = \frac{\text{TotalSF}}{\text{TotRmsAbvGrd}}$$</td>
      <td>Calculates spaciousness and spatial density, separating cramped floor plans from open-concept residential designs.</td>
    </tr>
  </tbody>
</table>

> **Empirical Validation Principle:** These features are constructed as candidate predictors; their incremental contribution is rigorously verified through cross-validated ablation and downstream feature importance audits rather than assumed *a priori*.

---

## 🔧 Preprocessing & Column Transformations

Preprocessing is orchestrated deterministically via `sklearn.compose.ColumnTransformer`, isolating transformations across disparate column modalities to guarantee zero cross-feature contamination.

---

### ⚙️ Transformation Sub-Pipelines

<table>
  <thead>
    <tr>
      <th align="left">Sub-Pipeline</th>
      <th align="left">Stage</th>
      <th align="left">Scikit-Learn Implementation</th>
      <th align="left">Operational Rationale</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><b>Numerical Branch</b></td>
      <td><b>Imputation</b></td>
      <td><code>SimpleImputer(strategy='median')</code></td>
      <td>Computes the median of each continuous column on training data only; protects central tendency from being skewed by extreme price or square-footage outliers.</td>
    </tr>
    <tr>
      <td><b>Feature Scaling</b></td>
      <td><code>StandardScaler()</code></td>
      <td>Normalizes continuous features to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$), preventing features with large raw scales (e.g., <code>LotArea</code>) from dominating regularized linear gradients.</td>
    </tr>
    <tr>
      <td rowspan="2"><b>Categorical Branch</b></td>
      <td><b>Imputation</b></td>
      <td><code>SimpleImputer(strategy='constant', fill_value='Missing')</code></td>
      <td>Preserves structural absence signals (e.g., no garage, no pool) as distinct categories rather than discarding rows or imputing false modes.</td>
    </tr>
    <tr>
      <td><b>Encoding</b></td>
      <td><code>OneHotEncoder(handle_unknown='ignore', sparse_output=False)</code></td>
      <td>Maps nominal categories into binary indicator vectors without imposing artificial ordinal scales; safely ignores unseen categorical levels encountered during production inference without throwing runtime shape exceptions.</td>
    </tr>
  </tbody>
</table>

---

## 🤖 Models Evaluated

To establish performance bounds and identify the optimal bias-variance tradeoff, four distinct regression architectures were systematically benchmarked against an empirical baseline:

---

### 📋 Candidate Architecture Specifications

<table>
  <thead>
    <tr>
      <th align="left">Model</th>
      <th align="left">Algorithmic Paradigm</th>
      <th align="left">Mathematical Formulation / Objective</th>
      <th align="left">Architectural Role</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Baseline Regressor</b><br><code>DummyRegressor</code></td>
      <td>Empirical Central Tendency</td>
      <td>$$\hat{y} = \bar{y}_{\text{train}}$$</td>
      <td>Naive control threshold ($R^2 \approx 0.0$); defines the minimum performance bound that any functional model must decisively beat.</td>
    </tr>
    <tr>
      <td><b>Linear Regression</b><br><code>LinearRegression</code></td>
      <td>Ordinary Least Squares (OLS)</td>
      <td>$$\min_{\boldsymbol{\beta}} \sum_{i=1}^n \left( y_i - \mathbf{x}_i^T \boldsymbol{\beta} \right)^2$$</td>
      <td>Standard unconstrained linear benchmark; exposes vulnerabilities to multicollinearity across correlated square-footage attributes.</td>
    </tr>
    <tr>
      <td><b>Ridge Regression</b><br><code>Ridge</code></td>
      <td>$L_2$-Regularized Linear Model</td>
      <td>$$\min_{\boldsymbol{\beta}} \left[ \sum_{i=1}^n \left( y_i - \mathbf{x}_i^T \boldsymbol{\beta} \right)^2 + \alpha \sum_{j=1}^p \beta_j^2 \right]$$</td>
      <td>Introduces quadratic weight shrinkage to penalize inflated coefficients, stabilizing regression slopes against high-dimensional collinear inputs.</td>
    </tr>
    <tr>
      <td><b>Random Forest</b><br><code>RandomForestRegressor</code></td>
      <td>Bootstrap Aggregation (Bagging)</td>
      <td>$$\hat{y} = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x})$$</td>
      <td>Constructs an ensemble of fully grown, de-correlated decision trees over bootstrap samples; targets variance reduction and non-linear interactions.</td>
    </tr>
    <tr>
      <td><b>Gradient Boosting</b><br><code>GradientBoostingRegressor</code></td>
      <td>Sequential Stage-Wise Boosting</td>
      <td>$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \gamma_m h_m(\mathbf{x})$$</td>
      <td>Fits consecutive shallow decision trees ($h_m$) iteratively against negative gradients (pseudo-residuals) of the loss function, aggressively minimizing bias.</td>
    </tr>
  </tbody>
</table>

---

## 📐 Evaluation Metrics

Model performance is evaluated across four complementary regression metrics, establishing both linear dollar-error expectations and penalizations for catastrophic valuation outliers:

---

### 📊 Metric Formulations & Analytical Intent

<table>
  <thead>
    <tr>
      <th align="left">Evaluation Metric</th>
      <th align="left">Mathematical Definition</th>
      <th align="left">Interpretability & Target Sensitivity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Mean Absolute Error</b><br><code>MAE</code></td>
      <td>$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n \left| y_i - \hat{y}_i \right|$$</td>
      <td>Measures the average magnitude of prediction deviations in raw dollar units ($). Treats all residuals uniformly without disproportionate weighting on luxury home valuation errors.</td>
    </tr>
    <tr>
      <td><b>Mean Squared Error</b><br><code>MSE</code></td>
      <td>$$\text{MSE} = \frac{1}{n} \sum_{i=1}^n \left( y_i - \hat{y}_i \right)^2$$</td>
      <td>Measures expected squared loss ($^2$). Quadratic exponentiation heavily penalizes extreme misses, making this the primary loss function optimized during gradient tree descent.</td>
    </tr>
    <tr>
      <td><b>Root Mean Squared Error</b><br><code>RMSE</code></td>
      <td>$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n \left( y_i - \hat{y}_i \right)^2}$$</td>
      <td>Restores squared error back into interpretable dollar terms ($). Retains high sensitivity to severe outliers while remaining directly comparable alongside MAE to audit error variance.</td>
    </tr>
    <tr>
      <td><b>Coefficient of Determination</b><br><code>R² Score</code></td>
      <td>$$R^2 = 1 - \frac{\sum_{i=1}^n \left( y_i - \hat{y}_i \right)^2}{\sum_{i=1}^n \left( y_i - \bar{y} \right)^2}$$</td>
      <td>Quantifies the fraction of total variance in property prices captured by the model relative to a mean baseline ($\hat{y} = \bar{y}$). Indicates explanatory power, not naive percentage accuracy.</td>
    </tr>
  </tbody>
</table>

> **Metric Sensitivity Note:** A wide divergence between **RMSE** and **MAE** ($\text{RMSE} \gg \text{MAE}$) signals high variance in residual magnitudes, indicating the model produces occasional severe valuation misses on luxury or idiosyncratic structural designs.

---

## 🏆 Model Comparison & Initial Holdout Results

All candidate pipelines were initially trained on an 80% partition (1,168 samples) and benchmarked against the untouched 20% holdout test partition (292 samples) using identical feature transformations.

---

### 📊 Performance Benchmark Matrix

| Model | MAE ($) | MSE | RMSE ($) | $R^2$ Score | Operational Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Mean)** | 62,575.93 | $7.677 \times 10^9$ | 87,619.03 | -0.000882 | Naive mean control; captures zero target variance. |
| **Linear Regression** | 21,294.20 | $4.729 \times 10^9$ | 68,766.23 | 0.383495 | Unstable; suffers severe coefficient inflation due to multicollinearity. |
| **Ridge Regression** | 19,116.25 | $9.475 \times 10^8$ | 30,781.93 | 0.876468 | $L_2$ regularization stabilizes matrix inversion, curbing variance. |
| **Random Forest** | 17,461.11 | $8.728 \times 10^8$ | 29,543.24 | 0.886210 | Bagging ensemble captures non-linear splits; outperforms linear models. |
| **Gradient Boosting** | **15,667.77** | $\mathbf{6.859 \times 10^8}$ | **26,189.03** | **0.910582** | **Best initial performer; stage-wise residual correction minimizes bias.** |

---

### 🔬 Empirical Findings & Diagnostic Analysis

* **Unconstrained OLS Collapse:** Ordinary Least Squares (OLS) produced an extreme RMSE of `$68,766.23` and a degraded $R^2$ of `0.383`. After one-hot encoding expanded high-cardinality nominal variables (e.g., `Neighborhood`), matrix near-singularity induced severe collinearity, inflating regression weights.
* **Regularization Recovery:** Applying an $L_2$ penalty via **Ridge Regression** shrunk inflated weights, slashing test RMSE by **55.2%** (down to `$30,781.93`) and lifting $R^2$ to `0.876`.
* **Tree-Based Superiority:** Both ensemble models captured non-linear boundary thresholds and complex feature interactions that linear combinations missed. 
* **Gradient Boosting Lead:** Without any hyperparameter tuning, **Gradient Boosting** achieved top marks across every evaluation metric, reducing holdout RMSE to `$26,189.03` with an $R^2$ of `0.911`.

---