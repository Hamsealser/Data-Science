# Explainability of Tree-Based Models: A Comparative Study of SHAP and LIME

## Research Question  
**How effectively can SHAP and LIME explain complex tree-based models across different domains?**  
This analysis investigates the reliability, consistency, and robustness of SHAP and LIME across three different tasks and datasets, using both Random Forest and XGBoost models.
We hypothesize that SHAP will provide more faithful, stable, and interpretable explanations than LIME across various datasets and models. Metrics like infidelity, monotonicity, and agreement are used to test this.



---

## Datasets and Experimental Setup

Three datasets were used to evaluate explanation quality in different contexts:

| Dataset              | Task                                                               | Models        | Explanations   |
|----------------------|--------------------------------------------------------------------|---------------|----------------|
| Salary Data (Job Description)| Use job descriptions (sentiment + keywords) to predict experience level and relate it to average salary | RF, XGBoost   | SHAP, LIME     |
| Fairjob Data| Predict whether a user will click on a job ad (FairJob dataset)    | RF, XGBoost   | SHAP, LIME     |
|Salary Data | Analyze how different skill indicators (e.g., Python, AWS) impact the predicted average salary | RF, XGBoost   | SHAP, LIME     |


In total, the study compares **12 model-explainer combinations**: 3 datasets × 2 models × 2 explanation techniques.

---

## Sub-RQ1: Global Explanation Quality
Which features do SHAP’s global importance scores identify as most predictive, and how faithfully do those global rankings reflect actual model behavior?

### MEMC (Mean Excluded Marginal Contribution)  
SHAP’s global feature rankings were most faithful in the FairJob dataset, where removing top-ranked features led to clear drops in model performance — especially in XGBoost, with MEMC of 0.137 vs. 0.114 for RF. ✅

In the Skill → Salary task, high MEMC values for features like Python (0.247) and AWS (0.286) in XGBoost showed strong model reliance on these features. However, RF models showed lower MEMC, indicating more diffuse reliance. ⚠️

In the Job Description → Experience task, MEMC scores were relatively low for both models (0.18 and ~0.25), suggesting weaker alignment between SHAP scores and actual model behavior. ⚠️❌

### AUPRC (Top-k Feature Selection)  
Across datasets, AUPRC confirmed that SHAP’s top features preserved predictive power:

In FairJob, AUPRC@3 reached 0.803 (XGBoost) and 0.752 (RF), reflecting high global fidelity. ✅

In the Skill task, AUPRC@3 exceeded 0.84 for Python and Excel, again confirming that top SHAP features carried strong signals. ✅

However, in the Job Description dataset, AUPRC for the top feature (salary) was only 0.64, with lower MEMC — suggesting only partial faithfulness. ⚠️

Conclusion: SHAP’s global importance scores were generally predictive and informative, especially in structured tasks like click prediction and skill impact.
Yet, their faithfulness varied across datasets and models — strong in XGBoost and FairJob, but weaker in Random Forest and text-heavy tasks.
Overall rating: ✅✅⚠️
---

## Sub-RQ2: Local Explanation Consistency and Agreement
How stable and consistent are local explanations under perturbations, and how well do SHAP and LIME agree on individual predictions?
### SHAP vs. LIME (Spearman Correlation)  
Local agreement between SHAP and LIME was consistently weak or negative across all datasets:

In the FairJob dataset, agreement was particularly poor, with Spearman ρ ≈ –0.72 (RF) and –0.45 (XGBoost), reflecting fundamentally different local explanations. ❌

In the Skill → Salary task, agreement scores ranged from 0.16 to 0.22, showing some convergence but not strong alignment. ⚠️

This suggests that SHAP and LIME rarely point to the same top features on an instance level — a critical gap for explainability in user-facing applications. ❌

### Infidelity  
SHAP explanations were generally faithful to the model’s output, especially in Random Forest models:

Across all datasets, RF models had lower infidelity scores than XGBoost (e.g., 0.0025 vs. 0.0056 in FairJob), showing that SHAP matched model behavior more closely. ✅

In Skill → Salary, XGBoost infidelity rose significantly (e.g., 0.103 for Python), suggesting unstable explanations despite high feature importance. ⚠️❌
### Sensitivity  
Stability under small input perturbations was highly model-dependent:

RF models were far more stable, with sensitivity values close to zero — e.g., 0.0015 in FairJob — indicating robust local explanations. ✅

XGBoost models, however, were highly sensitive, especially in the Skill task (e.g., 57.5 for Spark, 49.6 for Python), meaning small changes to input led to large changes in explanation. ❌
Conclusion:
Local SHAP explanations were stable and faithful in Random Forest models, but sensitive and inconsistent in XGBoost.
SHAP–LIME agreement was consistently poor, undermining trust in explanations across methods.
Overall rating: ✅⚠️❌
---

## Sub-RQ3: Monotonicity of Key Features
Monotonicity of SHAP values with respect to key continuous features varied significantly across datasets and models:

In the Skill → Salary task, SHAP values for features like Python (0.95) and AWS (0.92) in Random Forest and XGBoost showed strong monotonic relationships — i.e., as the feature value increased, its SHAP contribution also consistently increased. ✅

In the FairJob dataset, monotonicity was moderate (e.g., ~0.49 for XGBoost, ~0.45 for RF), indicating partial alignment between input strength (e.g., salary, click probability) and model explanation. ⚠️

In the Job Description → Experience task, monotonicity scores were very low (~0.15–0.20), suggesting little to no consistent pattern between input changes and SHAP attributions. ❌

Conclusion
SHAP attributions were strongly monotonic in well-structured, numeric tasks like skill-based salary prediction, but less consistent in more abstract or text-based settings.
Overall rating: ✅⚠️❌

---

## Sub-RQ4: Cross-Model and Cross-Domain Comparison
| Dataset                     | Target       | Model    | SHAP MEMC | AUPRC@3 | Infidelity | Sensitivity | Monotonicity | SHAP–LIME Agreement |
|-----------------------------|--------------|----------|-----------|---------|-------------|--------------|---------------|----------------------|
| Salary Data (Job Description) | Avg Salary   | RF       | ⚠️ 0.18   | ⚠️ 0.64 | ✅ 0.0067   | ⚠️ 0.78     | ❌ 0.15        | ❌ Low / unstable     |
|                             |              | XGBoost  | ✅ ~0.25   | ✅ 0.71 | ❌ 0.0090   | ❌ 0.82     | ❌ 0.20        | ❌ Low                |
| FairJob                     | Click        | RF       | ⚠️ 0.114  | ✅ 0.752| ✅ 0.0025   | ✅ 0.0015   | ⚠️ 0.455       | ❌ -0.722             |
|                             |              | XGBoost  | ✅ 0.137   | ✅ 0.803| ⚠️ 0.0056   | ❌ 0.0108   | ⚠️ 0.492       | ❌ -0.445             |
| Salary Data (Skill → Salary)| Python       | RF       | ⚠️ 0.151  | ✅ 0.846| ✅ 0.0006   | —           | ✅ 0.95        | ⚠️ 0.18               |
|                             |              | XGBoost  | ✅ 0.247   | ✅ 0.887| ❌ 0.103    | ❌ 49.64    | ⚠️ 0.39        | ⚠️ 0.18               |
|                             | Spark        | RF       | ❌ 0.017   | ✅ 0.781| ✅ 0.0005   | —           | ✅ 0.81        | ⚠️ 0.22               |
|                             |              | XGBoost  | ⚠️ 0.068   | ⚠️ 0.618| ❌ 0.076    | ❌ 57.51    | ❌ -0.09       | ⚠️ 0.22               |
|                             | AWS          | RF       | ⚠️ 0.046   | ✅ 0.798| ✅ 0.0008   | —           | ✅ 0.82        | ⚠️ 0.16               |
|                             |              | XGBoost  | ✅ 0.286   | ⚠️ 0.700| ❌ 0.062    | ❌ 42.46    | ✅ 0.92        | ⚠️ 0.16               |
|                             | Excel        | RF       | ⚠️ 0.089   | ✅ 0.799| ✅ 0.0008   | —           | ❌ -0.78       | ⚠️ 0.19               |
|                             |              | XGBoost  | ⚠️ 0.162   | ✅ 0.759| ❌ 0.122    | ❌ 57.53    | ❌ -0.59       | ⚠️ 0.19               |


These comparisons show that SHAP provides more detailed and accurate explanations than LIME, especially in XGBoost models, where top-ranked features aligned well with model behavior. However, this came at the cost of greater sensitivity and reduced local stability. Random Forest models offered more robust and consistent explanations but tended to distribute importance more evenly, leading to lower global faithfulness.
---

## Final Summary
SHAP emerged as a more effective and reliable explainability technique than LIME across nearly all evaluation criteria.
It produced high-fidelity global feature rankings and locally faithful attributions, even in complex, high-dimensional tasks. SHAP’s alignment with model behavior was particularly strong in XGBoost, which also achieved the highest predictive performance and most concentrated feature reliance.

However, SHAP explanations were also more sensitive in XGBoost, reflecting its sharper decision boundaries and susceptibility to input perturbations.
By contrast, Random Forest offered more robust and stable explanations, albeit with less feature concentration.

LIME, on the other hand, consistently showed weak to negative agreement with SHAP and less reliable local explanations — especially in low-dimensional or text-based contexts.

In conclusion, SHAP is the preferred tool for explaining tree-based models across diverse domains. It enables trustworthy explanations that support both global interpretation and instance-level reasoning — making it a strong choice for practical applications like job recommendations, skill-based salary modeling, and click prediction. ✅









