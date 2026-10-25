# Day 2 — Machine Learning Study Notes

**Topics:** regularization, validation design, imbalanced classification, calibration, uncertainty, distribution shift, and multimodal research.

> **Purpose:** A concise but rigorous record of ten practice questions, corrected explanations, useful equations, and practical lessons for reproducible medical machine learning.
>
> **Setting:** Longitudinal clinical records and wearable sensor data, where one patient can contribute many correlated observations.

## Contents

1. [Bias, variance, and model capacity](#1-bias-variance-and-model-capacity)
2. [L1, L2, and Elastic Net](#2-l1-l2-and-elastic-net)
3. [Grouped and nested cross-validation](#3-grouped-and-nested-cross-validation)
4. [AUROC and AUPRC under class imbalance](#4-auroc-and-auprc-under-class-imbalance)
5. [Classification thresholds and clinical costs](#5-classification-thresholds-and-clinical-costs)
6. [Weighted loss and oversampling](#6-weighted-loss-and-oversampling)
7. [Probability calibration and Brier score](#7-probability-calibration-and-brier-score)
8. [Confidence intervals, bootstrap, and random seeds](#8-confidence-intervals-bootstrap-and-random-seeds)
9. [External validation and distribution shift](#9-external-validation-and-distribution-shift)
10. [Designing a credible multimodal experiment](#10-designing-a-credible-multimodal-experiment)
11. [Core distinctions to remember](#core-distinctions-to-remember)
12. [Personal learning reflection and revision plan](#personal-learning-reflection-and-revision-plan)
13. [Python practice: paired patient bootstrap](#python-practice-paired-patient-bootstrap)

---

## 1. Bias, variance, and model capacity

**Scenario.** A logistic regression model achieves **82% training / 80% test accuracy**, while a 10-layer neural network achieves **99% training / 69% test accuracy**, using 150 patients and 80 features.

**What the results suggest:**

- The neural network has a much larger generalization gap and worse test accuracy. This suggests overfitting and possibly high variance.
- A large train–test gap is **evidence**, not a mathematical proof, of high variance. Strictly, variance concerns how model predictions change across different training samples.
- More independent, representative patients may help a flexible model learn patterns that generalize, rather than memorizing the training cohort.
- If logistic regression truly underfits, possible improvements include interaction terms, nonlinear transformations, splines, or less aggressive regularization.
- **Training for more epochs** only addresses insufficient optimization, not insufficient representational capacity. A larger learning rate does not reliably reduce bias.

**Core distinction:**

| Problem | Typical pattern | Possible response |
|---|---|---|
| Underfitting / high bias | Poor training and test performance | Add relevant features, nonlinearities, or model capacity; revisit regularization |
| Overfitting / high variance | Strong training results, weak test results | Collect more independent data, regularize, simplify, and validate robustly |
| Poor optimization | Training loss has not converged | Check learning rate, optimization settings, and training duration |

## 2. L1, L2, and Elastic Net

For coefficients \(w\), penalized learning minimizes a data loss plus a regularization term:

\[
\mathcal{J}(w)=\mathcal{L}_{\mathrm{data}}(w)+\lambda R(w).
\]

| Penalty | Definition | Derivative or subgradient | Typical effect |
|---|---|---|---|
| L1 | \(\lambda\sum_j |w_j|\) | \(\lambda\operatorname{sign}(w_j)\) for nonzero \(w_j\); at zero, subgradient \([-\lambda,\lambda]\) | Shrinkage and exact zeros |
| L2 | \(\lambda\sum_j w_j^2\) | \(2\lambda w_j\) | Smooth shrinkage; usually nonzero coefficients |
| Elastic Net | \(\lambda_1\lVert w\rVert_1+\lambda_2\lVert w\rVert_2^2\) | Combines both | Sparsity plus greater stability with correlated features |

**Why L1 promotes sparsity:** the absolute-value penalty has a sharp corner at zero. The subgradient can balance the data-loss gradient at exactly zero. L2 has a smooth gradient that becomes small near zero, so it generally shrinks coefficients without setting them exactly to zero.

In a simple one-dimensional quadratic problem,

\[
\min_w\;\frac12(w-a)^2+\lambda|w|,
\]

the L1 solution is **soft thresholding**:

\[
\hat w=\operatorname{sign}(a)\max(|a|-\lambda,0).
\]

**Correlated biomarkers:** resting heart rate and average daily heart rate may share substantial information. L1 can select one and eliminate another, sometimes unstably. L2 tends to distribute weight between correlated predictors. Elastic Net balances variable selection and stability. These are tendencies, not guarantees.

**Practical reminders:** Standardize features before comparing coefficient penalties; select regularization strength using grouped validation; avoid interpreting selected coefficients as causal effects. The intercept is typically left unpenalized.

## 3. Grouped and nested cross-validation

**Scenario.** Each of 500 patients contributes many wearable windows, and the goal is to predict outcomes in previously unseen patients.

**Main error to avoid:** splitting windows at random can place correlated observations from one patient in both training and validation/test partitions. That can produce optimistic performance estimates.

### Valid approach A: independent test cohort

1. Split **patients**, not windows, into development and test cohorts.
2. Within the development cohort, perform **grouped cross-validation** to choose hyperparameters.
3. In every training fold, fit imputation, scaling, feature selection, and any resampling procedures using that fold alone.
4. Refit the selected pipeline on the full development cohort.
5. Evaluate once on the independent test patients.

### Valid approach B: nested grouped cross-validation

- **Inner CV:** selects hyperparameters using the outer-training patients.
- **Outer CV:** evaluates the entire selection procedure on patients excluded from inner tuning.
- Report the distribution of outer-fold results, noting dependence across folds and sample-size limitations.

**Why not report the maximum tuning AUROC?** The chosen hyperparameter benefits partly from favorable validation noise. Selecting the maximum among noisy scores produces **model-selection optimism**; an independent evaluation is needed.

**Preprocessing rule:**

\[
z_{\mathrm{test}}=\frac{x_{\mathrm{test}}-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}.
\]

The test set is **transformed** using training parameters, not used to **estimate** those parameters. Leaving test features unscaled while training on scaled features is also incorrect.

*Implementation note:* In scikit-learn logistic regression, smaller `C` means stronger regularization; `C` is an inverse-strength parameter, with exact scaling conventions depending on the objective.

## 4. AUROC and AUPRC under class imbalance

**Scenario.** Disease prevalence is 2%.

| Metric | Model A | Model B |
|---|---:|---:|
| AUROC | 0.94 | 0.88 |
| AUPRC | 0.16 | 0.42 |
| Recall at a selected threshold | 90% | 85% |
| Precision at a selected threshold | 8% | 30% |

**AUROC** summarizes the ranking of positive cases above negative cases across thresholds. It uses:

\[
\mathrm{TPR}=\frac{TP}{TP+FN},\qquad
\mathrm{FPR}=\frac{FP}{FP+TN}.
\]

**AUPRC** summarizes precision versus recall across thresholds:

\[
\mathrm{Precision}=\frac{TP}{TP+FP},\qquad
\mathrm{Recall}=\frac{TP}{TP+FN}.
\]

A model can have higher AUROC but lower AUPRC because even a small false-positive rate may yield many false positives in a large healthy population. Neither area metric describes predictions at just one fixed threshold.

**Prevalence matters.** For prevalence \(\pi\), sensitivity \(\mathrm{TPR}\), and false-positive rate \(\mathrm{FPR}\):

\[
\mathrm{Precision}=
\frac{\mathrm{TPR}\,\pi}
{\mathrm{TPR}\,\pi+\mathrm{FPR}(1-\pi)}.
\]

For non-informative random ranking, the reference precision is the prevalence (2% here). Model B has much stronger precision–recall performance despite lower AUROC.

**Clinical interpretation:** A precision of 30% means that about 30 of every 100 patients flagged positive have the disease. It does **not** mean 30% of all diseased patients were detected; that is recall.

**Selection principle:** Choose evaluation metrics and operating thresholds based on clinical consequences, not on which metric looks largest. AUPRC depends on prevalence; AUROC is not changed by prevalence alone if class-conditional score distributions remain unchanged.

## 5. Classification thresholds and clinical costs

**Illustrative screening comparison:**

| Risk threshold | Recall | Precision | Specificity |
|---|---:|---:|---:|
| 0.2 | 95% | 20% | 70% |
| 0.5 | 80% | 50% | 90% |
| 0.8 | 55% | 75% | 98% |

Lowering a threshold generally produces **more positive predictions**: false negatives cannot increase and false positives cannot decrease, holding the prediction scores and outcomes fixed.

A screening application may favor threshold 0.2 because the consequences of missed cases are serious. But that choice also imposes a burden of additional false alarms.

A simple cost function is:

\[
C(t)=C_{FN}\,FN(t)+C_{FP}\,FP(t).
\]

Choose a threshold using prespecified clinical costs and appropriate validation data, rather than maximizing sensitivity without regard for false positives.

If the model probabilities are **well calibrated**, decisions are binary, and costs are limited to false positives and false negatives, then:

\[
\underbrace{C_{FP}(1-p)}_{\text{cost if predicting positive}}
<\underbrace{C_{FN}p}_{\text{cost if predicting negative}}
\quad\Longrightarrow\quad
p>\frac{C_{FP}}{C_{FP}+C_{FN}}.
\]

Thus the simplified optimal threshold is:

\[
t^*=\frac{C_{FP}}{C_{FP}+C_{FN}}.
\]

With \(C_{FP}=1\) and \(C_{FN}=9\), the threshold is 0.1 rather than 0.5. Real applications additionally require assessment of intervention harms, capacity, calibration, and clinical benefit.

**Common confusion:** 80% **recall** means 20% of actual cases are missed. It is not the same as 80% precision.

## 6. Weighted loss and oversampling

**Scenario.** Among 10,000 patients, 100 are positive (1%) and 9,900 are negative.

Unweighted binary cross-entropy (BCE) is:

\[
\ell(y,p)=-y\log p-(1-y)\log(1-p).
\]

A positive patient assigned \(p=0.01\) incurs a **large** loss of \(-\log(0.01)\approx4.61\). The imbalance problem is not that positive mistakes are individually cheap; it is that the **numerous negative examples** can dominate the aggregate objective.

**Class-weighted BCE:**

\[
\ell_{w}(y,p)=-w_+y\log p-w_-(1-y)\log(1-p).
\]

Increasing \(w_+\) strengthens the optimization signal from positive cases.

**Oversampling:** repeats or samples minority-class observations more frequently during training. It changes the training distribution but does **not** create new independent patient information. Repeated exposure can contribute to overfitting.

For longitudinal wearable data:

1. Split by **patient ID first**.
2. Oversample only within the training portion.
3. Account for differing numbers of windows per patient; naïve window-level sampling can overrepresent patients with long recordings.
4. Preserve representative validation/test distributions for evaluation of deployment performance.

**Important distinction:** class weighting and oversampling change what the model learns; threshold adjustment changes how a trained model's scores become decisions. Class weighting or altered training prevalence may also distort raw probability calibration, so evaluate calibration on representative data.

## 7. Probability calibration and Brier score

**Scenario.** Each group contains 200 patients.

| Predicted risk | Observed complications | Observed frequency | Result |
|---:|---:|---:|---|
| 10% | 20 | 10% | Matches prediction |
| 30% | 80 | 40% | Underestimates risk |
| 50% | 100 | 50% | Matches prediction |
| 70% | 140 | 70% | Matches prediction |
| 90% | 180 | 90% | Matches prediction |

**Calibration:** among patients assigned a risk near \(p\), approximately a fraction \(p\) should experience the event. Group-level agreement is evidence of calibration, but finite groups do not prove perfect population calibration.

**Discrimination** asks whether positive cases receive higher scores than negative cases. A strictly increasing transformation preserves ranking and AUROC while potentially changing calibration.

**Brier score:**

\[
BS=\frac1N\sum_{i=1}^N(p_i-y_i)^2.
\]

Lower is better. It measures overall squared probabilistic prediction error, **not calibration alone**. For a group whose true event probability is \(q\), and whose constant prediction is \(p\):

\[
\mathbb{E}[(p-Y)^2]=(p-q)^2+q(1-q).
\]

The first term measures probability mismatch; the second reflects irreducible outcome variability. For the five groups above, the pooled Brier score calculated from their stated predictions and counts is **0.178**.

**Recalibration methods:**

- **Platt scaling:** logistic mapping from model scores to event probabilities.
- **Isotonic regression:** flexible monotonic mapping.
- **Temperature scaling:** for a binary model logit \(z\), \(p_T=\sigma(z/T)\), where \(T>0\) is learned on held-out calibration data.

Oversampling/undersampling are **not recalibration methods**. Fit the recalibration procedure on independent calibration patients and evaluate the frozen pipeline on untouched test patients.

## 8. Confidence intervals, bootstrap, and random seeds

**Scenario.** On 200 patients, including 30 positive cases:

\[
\mathrm{AUROC}_{LR}=0.83,\qquad
\mathrm{AUROC}_{Transformer}=0.86,
\]

\[
\Delta\mathrm{AUROC}=0.03,\qquad
95\%\ \mathrm{CI}=[-0.02,0.08].
\]

**Conclusion:** the observed estimate favors the transformer, but this interval includes zero. Under a corresponding valid two-sided procedure, the evidence is insufficient to establish a nonzero improvement at the conventional 5% significance level. It **does not** prove the models equivalent. A 0.03 difference is not automatically clinically trivial.

**Frequentist confidence interval:** if the study and interval construction were repeated many times under the method's assumptions, about 95% of the resulting 95% intervals would cover the fixed true difference. Do not say there is a 95% probability that the fixed parameter lies inside this one observed interval.

### Paired patient bootstrap

1. Resample **patient IDs with replacement**, retaining whole patient clusters.
2. Apply exactly the same sampled patients to both models.
3. Recompute AUROC for each model and their difference.
4. Repeat many times and estimate an appropriate confidence interval.
5. Handle resamples with only one outcome class, where AUROC is undefined.

Resampling individual windows as though they were independent patients can underestimate uncertainty. Cluster bootstrapping addresses dependence; it is **not** the same concept as preventing data leakage.

### Random seeds versus bootstrap

| Multiple training seeds | Bootstrap of evaluation patients |
|---|---|
| Varies initialization, batch order, dropout, etc. | Varies sampled patient clusters |
| Studies optimization/training stability | Approximates sampling uncertainty of the evaluation statistic |
| Usually retrains the model | For fixed predictions, does not require retraining |
| Should use fixed patient partitions when isolating training randomness | Must preserve paired predictions and patient clustering |

Neither approach on its own proves external generalization.

## 9. External validation and distribution shift

**Scenario.** A frozen model trained at Hospital A is evaluated at Hospital B.

| Metric | Hospital A | Hospital B |
|---|---:|---:|
| Patients | 5,000 | 2,000 |
| Prevalence | 10% | 2% |
| AUROC | 0.88 | 0.72 |
| AUPRC | 0.45 | 0.12 |
| Brier score | 0.08 | 0.14 |

Hospital B has different demographics and wearable sensors, and more missing physiological measurements.

**Potential explanations:**

| Shift | Key change | Example |
|---|---|---|
| Covariate shift | \(P_A(X)\ne P_B(X)\), with \(P(Y\mid X)\) unchanged in the idealized definition | Different demographic/physiological distributions |
| Label/prior shift | \(P_A(Y)\ne P_B(Y)\), with \(P(X\mid Y)\) unchanged in the idealized definition | Prevalence decreases |
| Concept shift | \(P_A(Y\mid X)\ne P_B(Y\mid X)\) | The same features relate differently to outcomes |
| Measurement shift | Measurement process or encoding changes | Different devices, units, or sampling rates |
| Missingness shift | Availability or missingness mechanism changes | More absent wearable streams or laboratory tests |

Real deployments may combine several kinds of shift.

**Prevalence alone can reduce precision/AUPRC.** For fixed TPR = 0.8 and FPR = 0.1, precision is about **47.1%** at 10% prevalence, but only **14.0%** at 2% prevalence. Under pure prior shift with unchanged class-conditional score distributions, AUROC remains unchanged; the AUROC decline from 0.88 to 0.72 therefore needs additional investigation, accounting for uncertainty.

**Diagnostic workflow:**

1. Confirm identical endpoint definitions, prediction horizons, units, and preprocessing semantics.
2. Compare feature distributions, missingness, demographics, device groups, and outcome frequencies.
3. Examine performance and calibration by relevant subgroups and acquisition conditions.
4. Inspect error patterns and assess potential causes of shift before changing the model.
5. Test targeted remedies: device harmonization, multi-hospital training, missingness-aware modeling, or domain adaptation.

**Do not confuse remedies:**

- Recalibration improves agreement between risk scores and observed outcome frequencies; monotonic recalibration preserves AUROC.
- Threshold changes affect classification decisions at a chosen operating point, not the underlying AUROC.
- Retraining or domain adaptation can change learned representations and potentially improve ranking, but require independently held-out evaluation.

## 10. Designing a credible multimodal experiment

**Scenario.** There are 1,000 patients, multiple wearable windows per patient, and 5% complication prevalence. Initial scores come from a single observation-level split:

| Model | AUROC | AUPRC |
|---|---:|---:|
| Logistic regression | 0.80 | 0.25 |
| Wearable-only LSTM | 0.83 | 0.31 |
| EHR-only MLP | 0.82 | 0.29 |
| Multimodal transformer | 0.87 | 0.38 |

**Why the initial conclusion is weak:** observation-level splitting may leak patient-specific information, and the largest observed metric does not establish a reproducible architectural advantage.

### Experimental protocol

1. Split at patient level, respecting temporal prediction boundaries.
2. Use grouped validation (or nested grouped CV) for tuning and an independent test cohort for final evaluation.
3. Fit all preprocessing, missingness handling, feature selection, and resampling procedures inside the appropriate training folds.
4. Compare candidate models using the same patients, endpoints, and fair tuning budgets.
5. Track training variability using multiple random seeds without silently changing the test population.
6. Report discrimination, calibration, relevant decision thresholds, paired confidence intervals, and subgroup performance.

### Baselines that isolate scientific claims

| Comparison | What it tests |
|---|---|
| Wearable only vs. EHR only | Predictive contribution of each modality |
| Simple concatenation + MLP vs. multimodal transformer | Value beyond basic fusion |
| Late fusion vs. learned multimodal fusion | Benefit of representation-level interactions |
| Transformer with vs. without cross-modal attention | Specific contribution of cross-modal attention |
| With vs. without modality dropout | Robustness to missing modalities |
| Matched or comparable capacity/tuning budgets | Whether gains are simply due to a larger model or more tuning |

### Example testable research hypothesis

**Problem:** multimodal models may learn brittle dependencies on wearable sensors that differ across hospitals.

**Proposed improvement:** train a missingness-aware multimodal model with explicit missingness indicators and modality dropout.

**Hypothesis:** this model will improve discrimination on an independently held-out hospital with different sensor availability, without materially degrading calibration.

**Experiment:** compare a standard transformer and the modified model under matched training, tuning, and evaluation conditions. Predefine an external-hospital primary outcome, estimate the **paired performance difference and uncertainty**, and examine performance across realistic missingness strata.

**What could refute it:** no reliable external improvement, worse calibration, or gains restricted to unrealistic artificial masking conditions.

**Research principle:** a useful contribution follows the sequence **problem → proposed mechanism → testable prediction → controlled comparison → uncertainty-aware conclusion**.

---

## Core distinctions to remember

| Frequently confused ideas | Distinction |
|---|---|
| High variance vs. an observed train–test gap | The gap suggests overfitting; variance concerns variability across training samples. |
| Model capacity vs. training longer | More epochs cannot create nonlinear features absent from a model class. |
| L1 vs. L2 | L1 can yield exact zeros; L2 usually shrinks smoothly. |
| Validation vs. independent test | Validation helps choose a procedure; an independent test evaluates the chosen procedure. |
| Precision vs. recall | Precision conditions on predicted positives; recall conditions on actual positives. |
| AUROC vs. AUPRC | Both span thresholds, but PR performance is sensitive to prevalence. |
| Class weighting vs. threshold adjustment | The former changes training; the latter changes decisions. |
| Oversampling vs. recalibration | The former alters training exposure; the latter maps scores to probabilities. |
| Calibration vs. discrimination | Calibration measures probability reliability; discrimination measures ranking. |
| Bootstrap vs. multiple seeds | Bootstrap estimates sampling uncertainty; seeds examine training randomness. |
| CI containing zero vs. equivalence | Failure to establish a difference is not proof of equivalent performance. |
| Recalibration vs. domain adaptation | Recalibration adjusts probabilities; adaptation can change predictive representations. |
| Hyperparameter sensitivity vs. ablation | Sensitivity changes configurations; ablation isolates a proposed component. |

## Personal learning reflection and revision plan

### Progress since Day 1

- **Improved:** patient-level leakage prevention, basic L1/L2 distinctions, clinical metric interpretation, imbalanced learning, and the difference between discrimination and calibration.
- **Developing:** grouped versus nested validation, the mathematics of regularization and prevalence, external-validation diagnostics, and rigorous ablation design.
- **Priority gaps:** frequentist confidence intervals, paired clustered uncertainty estimation, and constructing falsifiable methodological hypotheses.

### Focused revision (about 2 hours)

- [ ] **30 min — Statistical inference:** explain a 95% confidence interval; distinguish evidence of a difference from evidence of equivalence.
- [ ] **20 min — Validation design:** draw grouped CV, nested CV, independent testing, and fold-local preprocessing.
- [ ] **20 min — Bootstrap practice:** resample patient IDs, calculate paired AUROC differences, and handle invalid samples.
- [ ] **30 min — Research design:** write a problem, proposed mechanism, falsifiable hypothesis, controlled comparison, and potential refutation.
- [ ] **20 min — Calibration and imbalance:** compare weighting, oversampling, Platt/temperature scaling, threshold choice, and prevalence effects.

### Self-check questions

1. Why can a 95% confidence interval include zero even when one model's observed AUROC is higher?
2. Why can AUPRC decline after deployment if AUROC stays constant?
3. Why must imputation and standardization be refitted inside each cross-validation training fold?
4. Why does patient-level bootstrapping differ from randomly resampling wearable windows?
5. Which ablation would isolate the value of cross-modal attention?
6. What observation would contradict the proposed missingness-robustness hypothesis?

---

## Python practice: paired patient bootstrap

This implementation assumes **one binary outcome and one prediction per patient**. If the task instead predicts outcomes at many times per patient, define the appropriate evaluation unit and clustered resampling scheme before applying it.

```python
import numpy as np
from sklearn.metrics import roc_auc_score


def paired_patient_auc_bootstrap(
    patient_ids, y_true, pred_a, pred_b,
    n_bootstrap=2000, random_state=42
):
    """Paired patient-level bootstrap of AUROC(B) - AUROC(A)."""
    patient_ids = np.asarray(patient_ids)
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    n = len(patient_ids)
    if not all(len(v) == n for v in (y_true, pred_a, pred_b)):
        raise ValueError("All arrays must have the same length")
    if len(np.unique(patient_ids)) != n:
        raise ValueError("Expected one row per unique patient")
    if len(np.unique(y_true)) != 2:
        raise ValueError("Both outcome classes must be present")

    rng = np.random.default_rng(random_state)
    diffs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if np.unique(y_true[idx]).size < 2:
            continue  # AUROC is undefined for single-class samples

        auc_a = roc_auc_score(y_true[idx], pred_a[idx])
        auc_b = roc_auc_score(y_true[idx], pred_b[idx])
        diffs.append(auc_b - auc_a)

    if not diffs:
        raise RuntimeError("No valid bootstrap samples")

    diffs = np.asarray(diffs)
    point_difference = (
        roc_auc_score(y_true, pred_b)
        - roc_auc_score(y_true, pred_a)
    )
    percentile_ci = np.percentile(diffs, [2.5, 97.5])
    return point_difference, percentile_ci, diffs
```

**Why this is paired:** the *same bootstrap indices* are applied to both models. This retains their within-patient performance relationship when estimating the difference.

**Limitation:** the percentile interval is one practical option, not universally the best uncertainty procedure. With few positive patients, evaluate interval stability and consider more appropriate cluster-aware or bias-corrected approaches where justified.

---

**Day 2 takeaway:** Rigorous medical machine learning is not only about improving model scores. It requires leakage-free evaluation, interpretable metrics, uncertainty-aware comparisons, and scientific experiments designed to test specific claims.
