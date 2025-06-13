# Data-Science
# Explainable ML for Job-Skill Matching, Sentiment & Rank Prediction

A multi-domain investigation into how well SHAP (and LIME) explanations reveal the inner workings of tree-based models, using three real-world datasets:

1. **Job–Skill Matching** (Glassdoor salary listings)  
2. **Sentiment Analysis** (text-based sentiment labels)  
3. **Rank Prediction** (numeric scores or ratings)

We train Random Forest and XGBoost classifiers/regressors on each task, explain their predictions with SHAP and LIME, and evaluate those explanations along five rigorous axes.

---

## Table of Contents

- [Research Questions](#research-questions)  
- [Datasets](#datasets)  
- [Methodology](#methodology)  
  - [Data Cleaning & Feature Engineering](#data-cleaning--feature-engineering)  
  - [Model Training](#model-training)  
  - [Explanation Methods](#explanation-methods)  
  - [Evaluation Metrics](#evaluation-metrics)  
- [Key Findings](#key-findings)  
- [Repository Structure](#repository-structure)  
- [Usage](#usage)  
- [Installation & Dependencies](#installation--dependencies)  
- [Future Directions](#future-directions)  
- [License](#license)  

---

## Research Questions

**Main RQ**  
> How effectively can SHAP and LIME explain complex tree-based models across different domains (job–skill matching, sentiment analysis, rank prediction)?

**Sub-RQ 1: Global Explanation Quality**  
> Which features do SHAP’s global importance scores identify as most predictive, and how faithfully do those global rankings reflect actual model behavior?  
> - **Metrics:**  
>   - **Global Fidelity (MEMC):** average AUC drop when zeroing out top-k SHAP features  
>   - **Top-k AUPRC:** precision of a fresh classifier trained only on the top-k SHAP features  

**Sub-RQ 2: Local Explanation Consistency & Agreement**  
> How stable and consistent are local explanations under perturbations, and how well do SHAP and LIME agree on individual predictions?  
> - **Stability:** Infidelity & Sensitivity under small input perturbations  
> - **Agreement:** Spearman’s ρ between SHAP vectors and LIME weights  

**Sub-RQ 3: Monotonicity of Key Features**  
> Do SHAP attributions vary monotonically with crucial continuous inputs (e.g. average salary, sentiment score, rank)?  
> - **Metric:** Spearman’s ρ between feature-quantile bins and mean SHAP in each bin  

**Sub-RQ 4: Cross-Model & Cross-Domain Comparison**  
> How do explanation-quality metrics compare:  
> 1. Random Forest vs. XGBoost  
> 2. Job–Skill vs. Sentiment vs. Rank datasets  

---

## Datasets

1. **Job–Skill Matching**  
   - Source: [Glassdoor job postings (2017–2018)](https://www.kaggle.com/datasets/thedevastator/jobs-dataset-from-glassdoor)  
   - Key features: job title, salary estimate, location, company info, job description length, “same_state”, candidate age & Boolean skill flags.  

2. **Sentiment Analysis**  
   - Source: _(to be specified)_  
   - Features: text embeddings or n-gram counts, sentiment labels (positive/negative).  

3. **Rank Prediction**  
   - Source: _(to be specified)_  
   - Features: job-rank or rating metadata, candidate/viewer demographics.

A cleaned CSV for the job dataset (`salary_data_cleaned_final.csv`) is included in this repo.

---

## Methodology

### Data Cleaning & Feature Engineering

- **Salary column parsing** → `min_salary`, `max_salary`, `avg_salary`  
- **Revenue text** → numeric midpoints  
- **Size** → categorical buckets (S, M, L, XL)  
- **Location fields** → one-hot encoding  
- **Drop** low-utility columns (`job_description`, `Founded`, `competitors`)  
- **Scale** `Rating` to 0–5, drop invalid rows, filter out hourly-only listings.

### Model Training

For each skill flag (and analogously for sentiment/rank):

1. **Split** 80/20 train/test (stratified on a dummy target)  
2. **Fit**  
   - Random Forest (100 trees)  
   - XGBoost (100 rounds, `eval_metric='logloss'`)  
3. **Compute** Accuracy & ROC-AUC on test set  

### Explanation Methods

- **SHAP**  
  - `TreeExplainer` for global & local feature attributions  
  - Summary plots and top-10 mean(|SHAP|) features  
- **LIME**  
  - `LimeTabularExplainer` for local explanations  
  - Spearman correlation with SHAP attributions

### Evaluation Metrics

1. **Global Fidelity (MEMC)**  
2. **Top-k Feature Selection AUPRC**  
3. **Local SHAP–LIME Agreement**  
4. **Robustness**  
   - *Infidelity* & *Sensitivity* under input perturbations  
   - *Batched implementations* for efficiency  
5. **Monotonicity**  
   - Spearman’s ρ vs. quantile-binned key continuous feature  

Detailed implementations are in  
- `notebooks/01`
- `notebooks/02`  
- `notebooks/03`  

---

## Key Findings

- **Random Forest + SHAP** yields more robust, more monotonic, and more predictive explanation sets compared to XGBoost.  
- **XGBoost SHAP** features cause larger fidelity drops (MEMC) yet are far less stable under small perturbations.  
- **Local SHAP–LIME agreement** is weak (ρ ≈ 0.2), suggesting the two methods offer complementary perspectives.  
- **Top-k AP** shows RF’s top features alone can recover high precision (AP@3 > 0.78 across tasks).  

_For full results and plots, see_ `notebooks/03`.

---

## Repository Structure

