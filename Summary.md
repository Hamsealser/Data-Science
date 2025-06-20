# Explainability of Tree-Based Models: A Comparative Study of SHAP and LIME

## Research Question  
**How effectively can SHAP and LIME explain complex tree-based models across different domains?**  
This analysis investigates the reliability, consistency, and robustness of SHAP and LIME across three different tasks and datasets, using both Random Forest and XGBoost models.

---

## Datasets and Experimental Setup

Three datasets were used to evaluate explanation quality in different contexts:

| Dataset              | Task                                | Models        | Explanations   |
|----------------------|-------------------------------------|---------------|----------------|
| Job Experience       | Predict experience level from jobs  | RF, XGBoost   | SHAP, LIME     |
| Click Prediction     | Predict ad clicks (FairJob)         | RF, XGBoost   | SHAP, LIME     |
| Skill Detection      | Predict presence of key skills      | RF, XGBoost   | SHAP, LIME     |

In total, the study compares **18 model-explainer combinations**: 3 datasets × 2 models × 2 explanation techniques.

---

## Sub-RQ1: Global Explanation Quality

### MEMC (Mean Excluded Marginal Contribution)  
SHAP-based MEMC scores showed that XGBoost models generally rely more heavily on their top features compared to Random Forest models. This pattern held true across all datasets. For example, in the FairJob dataset, removing the top three SHAP-ranked features led to a significantly larger drop in AUC for XGBoost than for Random Forest.  

In skill prediction, XGBoost models showed the highest MEMC scores across most skills, indicating a sharper concentration of model reliance on specific features.

### AUPRC (Top-k Feature Selection)  
Using only the top-k SHAP features, both models retained a substantial amount of predictive power, especially in XGBoost. For instance, in the click prediction task, AUPRC for the top 3 SHAP features was above 0.78 for XGBoost, while Random Forest reached about 0.56.  

In the job experience model, even one SHAP-ranked feature (salary) yielded an AUPRC of 0.64. These results suggest that SHAP’s global feature rankings are both informative and actionable.

---

## Sub-RQ2: Local Explanation Consistency and Agreement

### SHAP vs. LIME (Spearman Correlation)  
Across all datasets, the agreement between SHAP and LIME was weak or even negative. The worst disagreement appeared in the FairJob dataset using Random Forest, where the Spearman correlation between SHAP and LIME rankings was approximately -0.72.  

Even in XGBoost models, where explanations are more concentrated, SHAP and LIME often highlighted different features. This indicates a fundamental methodological mismatch: SHAP is model-specific and grounded in the internal logic of the model, while LIME fits a local surrogate that may overlook global feature relevance.

### Infidelity  
SHAP explanations for Random Forest were consistently more faithful to the model’s behavior. In all three datasets, SHAP had lower infidelity scores in RF models compared to XGBoost. For instance, in the FairJob dataset, infidelity for RF was 0.0025, compared to 0.0040 in XGBoost.

### Sensitivity  
SHAP explanations in XGBoost were more sensitive to small input perturbations. This suggests that while SHAP captures meaningful interactions, it can also reflect instability in models like XGBoost, where the decision boundaries are sharper. Random Forest models had lower sensitivity and therefore produced more stable explanations.

---

## Sub-RQ3: Monotonicity of Key Features

Monotonicity was evaluated by checking whether SHAP values changed consistently (either increasing or decreasing) as input feature values increased.  

XGBoost consistently showed stronger monotonic relationships than Random Forest. For example, in the FairJob dataset, monotonicity between salary and SHAP value was approximately 0.49 in XGBoost compared to 0.45 in RF. In the job experience dataset, however, the correlation was weak (around 0.15), indicating no clear trend between salary and its influence on the prediction.

---

## Sub-RQ4: Cross-Model and Cross-Domain Comparison

| Dataset         | Model      | SHAP MEMC | AUPRC@3 | Infidelity | Sensitivity | Monotonicity | SHAP–LIME Agreement |
|-----------------|------------|-----------|---------|-------------|--------------|---------------|----------------------|
| Job Experience  | RF         | 0.18      | 0.64    | 0.0067      | 0.78         | 0.15          | Low / unstable       |
|                 | XGBoost    | ~0.25     | 0.71    | 0.0090      | 0.82         | 0.20          | Low                  |
| Click Prediction| RF         | 0.02      | 0.56    | 0.0025      | 0.0015       | 0.45          | -0.72                |
|                 | XGBoost    | 0.06      | 0.78    | 0.0040      | 0.0037       | 0.49          | -0.65                |
| Skill Detection | RF         | varies    | ~0.60   | ~0.0040     | ~0.0050      | 0.3–0.5       | Mixed                |
|                 | XGBoost    | highest   | >0.75   | ~0.0055     | ~0.0080      | 0.4–0.6       | Low                  |

These comparisons show that SHAP delivers more detailed and accurate explanations, especially in XGBoost models. However, these explanations come with greater sensitivity and slightly reduced stability. Random Forest is more robust and interpretable, but spreads importance more evenly across features.

---

## Final Summary

SHAP proved to be a more effective and reliable explanation technique than LIME in nearly every respect. It provided high-fidelity global and local insights, even when models had complex decision boundaries or noisy input features. SHAP also aligned more closely with actual model behavior, particularly in Random Forest models.  

XGBoost consistently achieved higher predictive accuracy and stronger feature signals, which was reflected in SHAP’s ability to identify impactful features. However, these models were also more sensitive and produced more volatile explanations. LIME was less reliable overall and showed poor agreement with SHAP, especially in small or low-dimensional models.

In conclusion, SHAP is the preferred tool for explaining tree-based models across multiple domains. It offers trustworthy explanations that support both global insight and local reasoning, making it suitable for real-world applications such as job recommendation, skill matching, and ad click prediction.
