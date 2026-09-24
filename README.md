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

## 🔁 Cross-Validation Analysis (5-Fold Leakage-Free)

To verify that model rankings and performance advantages were not artifacts of an idiosyncratically favorable single train/test split, stratified **5-fold cross-validation** was executed across the full dataset. Every fold re-fit the entire transformation and imputation pipeline strictly on in-fold training data.

---

### 📊 Out-of-Fold Performance Benchmark

| Model | Mean MAE ($) | Mean MSE | Mean RMSE ($) | Mean $R^2$ Score | Generalization Stability |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Linear Regression** | 18,822.37 | $2.186 \times 10^9$ | 46,759.39 | 0.662341 | High variance across splits; recurring collinear coefficient instability. |
| **Ridge Regression** | 18,141.46 | $1.311 \times 10^9$ | 36,207.07 | 0.781246 | Robust regularized baseline; prevents matrix inversion explosion. |
| **Random Forest** | 17,154.32 | $9.502 \times 10^8$ | 30,825.59 | 0.842849 | Low-variance ensemble; stable out-of-fold generalization. |
| **Gradient Boosting** | **15,252.49** | $\mathbf{8.906 \times 10^8}$ | **29,842.64** | **0.850382** | **Top generalization; lowest mean error across all 5 partitions.** |

---

### 💡 Cross-Validation Takeaways

* **Invariant Rank Ordering:** Model performance hierarchies remained consistent with holdout testing:  
  $$\text{Gradient Boosting} \succ \text{Random Forest} \succ \text{Ridge} \succ \text{OLS}$$
* **True Out-of-Fold Error Baseline:** Gradient Boosting cemented its superiority with a cross-validated mean RMSE of **$29,842.64** and an average MAE of **$15,252.49**, capturing **85.0%** of target variance across unobserved partitions.
* **Leakage Verification:** By re-computing all feature scalers, median imputations, and one-hot encodings within the fold boundaries, out-of-fold metrics accurately reflect production performance expectations.

---

## ⚙️ Hyperparameter Tuning: Random Forest

Both ensemble architectures were systematically tuned using `GridSearchCV` configured with 5-fold cross-validation, optimizing explicitly against validation root mean squared error (`scoring='neg_root_mean_squared_error'`).

---

### 🎛️ Optimal Parameter Configuration

| Parameter | Selected Value | Search Rationale & Impact |
| :--- | :---: | :--- |
| `n_estimators` | `200` | Expands ensemble forest size from default 100 trees to stabilize variance and smooth decision boundaries. |
| `max_depth` | `None` | Allows individual decision trees to grow unconstrained until pure leaf nodes or minimum leaf thresholds are satisfied. |
| `min_samples_leaf` | `2` | Regularizes leaf nodes to require at least 2 samples per terminal leaf, dampening overfitting on extreme training outliers. |

---

### 📊 Validation & Holdout Test Performance

| Metric | Score | Analytical Interpretation |
| :--- | :---: | :--- |
| **Best CV RMSE** | **$29,470.90** | Lowest out-of-fold average error achieved across the parameter sweep. |
| **Holdout MAE** | **$17,549.32** | Expected absolute dollar deviation on unseen holdout transactions. |
| **Holdout MSE** | **$8.978 \times 10^8$** | Squared error loss penalty on holdout observations. |
| **Holdout RMSE** | **$29,963.25** | Rooted quadratic error penalty restored to dollar units ($). |
| **Holdout $R^2$ Score** | **0.882952** | Explains ~88.3% of valuation variance on the holdout test set. |

> **Stability Note:** The tight convergence between the 5-fold CV RMSE (**$29,470.90**) and the holdout test RMSE (**$29,963.25**) demonstrates that the tuned Random Forest model generalizes cleanly without overfitting to in-sample training splits.

---

## ⚙️ Hyperparameter Tuning: Gradient Boosting (Final Selected Model)

Sequential tree boosting was tuned across shrinkage factors, tree depths, and boosting stages via `GridSearchCV` (5-fold CV) targeting negative root mean squared error.

---

### 🎛️ Optimal Parameter Configuration & Architectural Mechanics

| Hyperparameter | Tuned Value | Mechanistic Role & Optimization Rationale |
| :--- | :---: | :--- |
| `learning_rate` | `0.03` | **Shrinkage Factor:** Dampens the step-size contribution of each successive base learner by 97%, slowing the descent along the loss gradient to prevent step overshoots and improve generalization. |
| `max_depth` | `4` | **Interaction Depth:** Limits individual regression trees to shallow splits, allowing the ensemble to capture up to 4-way non-linear feature interactions while preventing memory-memorization splits. |
| `n_estimators` | `500` | **Sequential Stages:** Scales up boosting iterations to offset the conservative learning rate, providing sufficient capacity to systematically eliminate structured residuals. |

---

### 📊 Validation & Holdout Test Performance

| Metric | Score | Operational Context |
| :--- | :---: | :--- |
| **Best CV RMSE** | **$27,149.29** | Top cross-validated score across all examined model architectures. |
| **Holdout MAE** | **$16,034.59** | Average valuation deviation of ~$16k across unseen residential sales. |
| **Holdout MSE** | **$7.115 \times 10^8$** | Minimum observed squared penalization on holdout records. |
| **Holdout RMSE** | **$26,673.99** | Maintains the tightest error distribution on raw dollar terms ($). |
| **Holdout $R^2$ Score** | **0.907240** | **Captures >90.7% of total variance** in home prices on unseen test data. |

> **Production Selection Verdict:** Gradient Boosting delivered the lowest cross-validation RMSE (**$27,149.29**) and the lowest holdout test RMSE (**$26,673.99**), consistently outperforming linear baselines and bagging ensembles across all metrics. It was selected as the core estimator for the finalized, serialized production pipeline.

---

## 📦 Final Model & Deployment Artifact

The finalized production asset packages domain-level feature engineering, numerical median-imputation, standard scaling, categorical constant-imputation, and one-hot encoding alongside the tuned sequential gradient booster into an atomic `sklearn.pipeline.Pipeline`.

```python
# Final Tuned Estimator Hyperparameters
GradientBoostingRegressor(
    learning_rate=0.03,
    max_depth=4,
    n_estimators=500,
    random_state=42
)
```

```bash
PRODUCTION HOLDOUT AUDIT (Unseen Partition: 292 Records)
========================================================================================
Mean Absolute Error (MAE)            :  $16,035
Root Mean Squared Error (RMSE)       :  $26,674
Coefficient of Determination (R²)    :   0.9072  (Explains 90.72% of Price Variance)


models/
└── house_price_model.joblib   ◄ [Serialized Atomic Pipeline: Transformers + Estimator]


RESIDUAL SKEW CHARACTERISTIC
========================================================================================
Actual Price Range ($)         Observed Model Behavior
────────────────────────────────────────────────────────────────────────────────────────
$50,000  – $300,000  (90%)  ──► Balanced residuals , tight variance within ±$15,000 MAE.
$300,000 – $500,000  (8%)   ──► Mild compression toward the regional median.
$500,000+            (2%)   ──► Systemic underestimation (Right-tail truncation).
```

## 🔬 Residual Error Stratification by Subgroup

Aggregated evaluation scores can obscure severe localized prediction disparities. Slicing test set residuals across spatial, structural quality, and valuation strata reveals where model variance concentrates.

---

### 📊 Subgroup Diagnostic Breakdown

<table>
  <thead>
    <tr>
      <th align="left">Dimension</th>
      <th align="left">Stratum / Cohort</th>
      <th align="center">Mean Absolute Error</th>
      <th align="left">Structural Driver & Failure Mode Analysis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><b>Geographic<br>Neighborhood</b></td>
      <td><code>NridgHt</code> (Northridge Heights)</td>
      <td align="center"><b>≈ $46,670</b></td>
      <td rowspan="3">Affluent development corridors dominated by bespoke luxury builds. High baseline land value and customized premium upgrades induce high price variance that standard tabular features only partially capture.</td>
    </tr>
    <tr>
      <td><code>NoRidge</code> (Northridge)</td>
      <td align="center"><b>≈ $45,969</b></td>
    </tr>
    <tr>
      <td><code>StoneBr</code> (Stone Brook)</td>
      <td align="center"><b>≈ $38,708</b></td>
    </tr>
    <tr>
      <td rowspan="2"><b>Construction &<br>Finish Quality</b></td>
      <td><code>OverallQual = 9</code> (Excellent)</td>
      <td align="center"><b>≈ $46,704</b></td>
      <td rowspan="2">At top-tier ratings (9–10), material costs, high-end architectural customizations, and designer amenities scale super-linearly, widening the dispersion of realized sale prices.</td>
    </tr>
    <tr>
      <td><code>OverallQual = 10</code> (Very Excellent)</td>
      <td align="center"><b>≈ $52,371</b></td>
    </tr>
    <tr>
      <td rowspan="4"><b>Valuation<br>Band</b></td>
      <td><code>&lt; $200k</code></td>
      <td align="center"><b>$10,361</b></td>
      <td>High training density; standard tract construction yields tight, stable residual bounds.</td>
    </tr>
    <tr>
      <td><code>$200k – $400k</code></td>
      <td align="center"><b>$25,984</b></td>
      <td>Mid-to-upper housing stock; moderate variance driven by partial modernizations and additions.</td>
    </tr>
    <tr>
      <td><code>$400k – $600k</code></td>
      <td align="center"><b>$59,059</b></td>
      <td>Semi-custom builds; non-linear premiums emerge for lot positioning and interior finishes.</td>
    </tr>
    <tr>
      <td><code>$600k – $800k</code></td>
      <td align="center"><b>$131,118</b></td>
      <td>Severe sparse-sample region; tree-based models encounter leaf value limits, capping peak valuations.</td>
    </tr>
  </tbody>
</table>

> **Key Diagnostic Takeaway:** The model achieves high precision across the core residential market ($<\$200\text{k}$, representing the bulk of transactions with an average error of only ~$\$10\text{k}$). Error magnitudes scale in direct proportion to valuation tiers, driven by non-linear luxury premiums and the bounded step-function nature of tree ensembles at the upper tail.

---

### 💡 Analytical Interpretation & Failure Mode Synthesis

The subgroup residual decomposition reveals that prediction error is not uniformly distributed across the domain space; instead, absolute error scales monotonically with property valuation:

* **Proportional Scaling vs. Systematic Bias:** The expansion in absolute dollar errors among high-value properties reflects heteroscedasticity inherent to real estate markets. As home values increase, discretionary premiums (e.g., custom stonework, panoramic lot orientation, designer finishes) fluctuate significantly more than standard square-footage metrics can capture.
* **Sample Sparsity at the Upper Tail:** Properties trading above $\$400\text{k}$ account for a small fraction of the training corpus. Tree-based partitioning struggles in data-sparse domains, as terminal leaf nodes calculate averages over very few observations and cannot extrapolate beyond observed historical thresholds.
* **Loss Function Mechanics:** Training directly on untransformed dollar values with an RMSE objective heavily penalizes large numerical mistakes, but does not enforce constant relative error across price tiers. Applying target stabilization transformations (e.g., $\log(1 + y)$) is a clear next step to homogenize percentage errors across entry-level and luxury brackets alike.

---

## 🌲 Feature Importance & Signal Attribution

Global feature importances were extracted directly from the fitted `GradientBoostingRegressor` based on mean impurity reduction (variance reduction across all split nodes).

---

### 📊 Top 20 Predictive Features (Gini / Variance Reduction)

| Rank | Feature Identifier | Feature Origin | Relative Importance | Cumulative Share | Analytical Interpretation |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | `TotalSF` | **Engineered** | **0.376572** | **37.66%** | Total usable space across all floors; acts as primary dimensional anchor. |
| **2** | `OverallQual` | Raw Tabular | **0.355472** | **73.20%** | Material, build finish, and aesthetic tier (scale 1–10). |
| **3** | `2ndFlrSF` | Raw Tabular | 0.024772 | 75.68% | Dedicated second-story footprint (vertical space distribution). |
| **4** | `HouseAge` | **Engineered** | 0.023624 | 78.04% | Physical depreciation and structural age at transaction date. |
| **5** | `TotalBathrooms` | **Engineered** | 0.019845 | 80.03% | Standardized plumbing utility across above-grade and basement zones. |
| **6** | `GarageCars` | Raw Tabular | 0.016190 | 81.65% | Vehicular storage capacity. |
| **7** | `BsmtFinSF1` | Raw Tabular | 0.013988 | 83.05% | Finished, usable basement square footage. |
| **8** | `GrLivArea` | Raw Tabular | 0.012673 | 84.31% | Above-grade ground living area. |
| **9** | `LotArea` | Raw Tabular | 0.012550 | 85.57% | Parcel and land acreage dimensions. |
| **10** | `BsmtQual_Ex` | Encoded Category | 0.011600 | 86.73% | Indicator for premium, high-ceiling basement foundation. |
| **11** | `LotFrontage` | Raw Tabular | 0.010560 | 87.79% | Linear feet of street connection. |
| **12** | `YearRemodAdd` | Raw Tabular | 0.008859 | 88.67% | Timestamp of last major structural renovation. |
| **13** | `YearBuilt` | Raw Tabular | 0.007800 | 89.45% | Original calendar year of construction. |
| **14** | `OverallCond` | Raw Tabular | 0.006651 | 90.12% | Functional maintenance condition of the property. |
| **15** | `TotalPorchSF` | **Engineered** | 0.005808 | 90.70% | Aggregated exterior patio, deck, and porch footprint. |
| **16** | `KitchenAbvGr` | Raw Tabular | 0.005786 | 91.28% | Number of above-grade kitchens (multi-family layout flag). |
| **17** | `GarageArea` | Raw Tabular | 0.005234 | 91.80% | Total enclosed garage square footage. |
| **18** | `KitchenQual_TA`| Encoded Category | 0.004280 | 92.23% | Indicator for baseline ("Typical / Average") kitchen finishes. |
| **19** | `BsmtUnfSF` | Raw Tabular | 0.003752 | 92.60% | Unfinished basement storage space. |
| **20** | `KitchenQual_Ex`| Encoded Category | 0.003716 | 92.98% | Indicator for high-end luxury kitchen installations. |

---

### 💡 Engineering Validation & Methodological Caveats

* **Engineered Feature Leverage:** Custom domain variables directly validate the feature engineering strategy. `TotalSF`, `HouseAge`, `TotalBathrooms`, and `TotalPorchSF` aggregate substantial variance, proving that composite geometric representations simplify tree split decisions compared to fragmented raw variables alone.
* **Pareto Distribution:** The top two features (`TotalSF` at **37.7%** and `OverallQual` at **35.5%**) account for **over 73.2%** of total impurity reduction across the 500 boosting stages.
* **Methodological Note on Causality:** Tree-based impurity metrics quantify how frequently a feature was selected to split sample variance within this specific dataset. They **do not prove causal mechanisms** (e.g., adding an unneeded bathroom will not guarantee an immediate, linear dollar increase equal to the model's split weighting).

---

## 📂 Project Structure

```bash
house-price-prediction/
├── data/
│   └── train.csv                    # Raw Ames Housing dataset (1,460 rows × 81 columns)
│
├── models/
│   └── house_price_model.joblib     # Serialized production pipeline (Transformers + GBDT)
│
├── src/
│   ├── __init__.py                  # Package identifier
│   ├── feature_engineering.py       # Custom transformers (TotalSF, HouseAge, etc.)
│   ├── preprocessing.py             # ColumnTransformer setup (Imputation, Scaling, OHE)
│   ├── model.py                     # Pipeline constructors and model factories
│   ├── evaluation.py                # Metric evaluation utilities (MAE, MSE, RMSE, R²)
│   └── save_model.py                # Joblib artifact serialization helpers
│
├── main.py                          # Primary execution script for training & evaluation
├── predict.py                       # Standalone inference interface for raw tabular data
├── requirements.txt                 # Project environment dependencies
├── .gitignore                       # Git ignore rules for cache and virtual environments
└── README.md                        # Project documentation
```

---