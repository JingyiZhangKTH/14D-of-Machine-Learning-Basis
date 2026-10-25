# Day 2 — Complete Machine Learning Learning Journal

**GitHub-ready annotated study report**  
**Subject:** Patient-centered machine learning, clinical evaluation, uncertainty, generalization, and multimodal study design  
**Coverage:** All 10 questions · my submitted answers · corrected answers · detailed feedback · concepts and equations · personal weaknesses · revision plan · runnable Python practice

> **Purpose:** This document records what I initially answered, what the correct reasoning is, why my mistakes matter, and what to revise next. It is intended to be an honest learning log, not a polished claim that everything is already mastered.
>
> **Scope:** Longitudinal EHR and wearable data with repeated observations per patient, rare outcomes, multimodal models, and external validation.

## Navigation

- [Part I — Complete Question-and-Answer Record](#part-i--complete-question-and-answer-record)
- [Part II — Technical Reference Notes and Worked Examples](#part-ii--technical-reference-notes-and-worked-examples)
- [Part III — Personal Weakness Analysis and Improvement Plan](#part-iii--personal-weakness-analysis-and-improvement-plan)

### Topics covered

| Question | Topic | Main correction |
|---|---|---|
| 1 | Bias–variance | Representation vs. optimization |
| 2 | Regularization | Why L1 is sparse and how correlated features behave |
| 3 | Validation design | Selection bias, nested CV, preprocessing leakage |
| 4 | AUROC and AUPRC | Curves across thresholds and prevalence effects |
| 5 | Decision thresholds | Precision/recall definitions and clinical cost functions |
| 6 | Class imbalance | Weighted BCE, oversampling, patient-level dependence |
| 7 | Calibration | Genuine recalibration versus rebalancing |
| 8 | Statistical uncertainty | CI interpretation, cluster bootstrap, random seeds |
| 9 | Distribution shift | Diagnosing shift before choosing a remedy |
| 10 | Multimodal study design | Ablation logic and falsifiable methodological hypotheses |

---


# Part I — Complete Question-and-Answer Record

This section documents all ten practice questions, the learner's submitted answers, a corrected model answer, and individualized feedback. Spelling and wording in quoted learner answers have been preserved where available; the purpose is to show what changed during learning, not to silently rewrite history.

**Source fidelity note:** The original wording of Questions 1 and 2 and the full submitted answer to Question 1 are not available verbatim in the retained discussion record. Their prompts and Question 1 answer are explicitly reconstructed from the session summary. Questions 2–10 contain the submitted answers as written and faithfully restate their question prompts.

## Question 1 — Bias, variance, model complexity, and optimization

### The question

*Reconstructed from the session record.* You have 150 patients and 80 physiological features. Logistic regression obtains 82% training accuracy and 80% test accuracy. A deep neural network with ten hidden layers obtains 99% training accuracy and 69% test accuracy.

1. Which model exhibits stronger evidence of high variance? What can and cannot be inferred from the train–test gap?
2. Why might adding more independent patients improve the neural network?
3. If logistic regression underfits, how could you reduce bias without immediately replacing it with a deep model?

### My answer

*Paraphrase recorded during the session; not a verbatim quotation.*

> **1.** The neural network (Model B) has higher variance because its training and test accuracy differ substantially and its test generalization is poor.
>
> **2.** Additional patients provide more diversity and information, which may improve generalization and reduce variance.
>
> **3.** Train for more epochs or use a larger learning rate.

### Feedback on my answer

| Part | Coaching score | Assessment |
|---|---:|---|
| 1 | 8/10 | Correct identification of the gap; define variance more precisely. |
| 2 | 9/10 | Strong generalization intuition; representative, independent data are key. |
| 3 | 3/10 | Confused optimization settings with representational bias. |

### Correct answer and explanation

**1 — Evidence, not a proof of variance.** The deep model has a 30-percentage-point train–test accuracy gap, versus 2 points for logistic regression. That is evidence of overfitting. Strictly, the variance of a learning algorithm concerns how much predictions or estimated models vary across independently sampled training sets. A single train–test gap does not mathematically establish the variance component; differences in data distributions, noise, and evaluation uncertainty are also relevant.

**2 — More independent patients.** Flexible networks can fit complex patient-specific patterns. Adding patients can expand coverage of clinically relevant variation and constrain the model toward relationships shared across individuals. More windows from the *same* patient do not offer the same independent information as additional patients. More data also cannot automatically repair leakage, dataset shift, or incorrect labels.

**3 — Reduce bias by changing representation or model capacity.** Possible options include interactions (heart rate × age), polynomial features, splines, a generalized additive model, a nonlinear kernel, adding informative predictors, or reducing excessively strong regularization. More epochs only help if optimization has not converged; a higher learning rate changes step size and may destabilize training. It does not expand the hypothesis class.

**Key distinction:** underfitting/representational bias is not the same as incomplete optimization.

**My recurring weakness:** translating the observation “training does not perform well enough” into the right intervention. Diagnose representational capacity and optimization separately.

---

## Question 2 — L1, L2, and Elastic Net regularization

### The question

You train logistic regression with 100 standardized physiological features. Compare the objectives

\[
\mathcal L_{\mathrm{BCE}}(w)+\lambda\sum_j |w_j|
\quad\text{and}\quad
\mathcal L_{\mathrm{BCE}}(w)+\lambda\sum_j w_j^2.
\]

1. What happens to coefficients as regularization strength increases, including extremely large strength?
2. Why can L1 yield exact-zero coefficients while L2 generally does not? Explain using the derivatives or penalty geometry.
3. What happens with highly correlated biomarkers, such as resting and average daily heart rate?
4. What is Elastic Net, and when is it useful?

### My original answers

> **1.** as lamda increases, the learned coeffecients decrease, if lamda becomes extremely large, the weight can be zero for L1, and they can be very close to 0 but never reach it for L2.
>
> **2.** becaue L1 contains absolute weight, but L2 contains square weight.
>
> **3.** the L1 will set one of the weight as zero, L2 will set the weight of one of them to nearly zero.
>
> **4.** elastic net regularization is a combination of L1 and L2, it can set the coeffecients to zero, and also contains square weights. it can help with reducing the feature dimentions

### Feedback on my answers

| Part | Coaching score | Assessment |
|---|---:|---|
| 1 | 9/10 | Correct broad shrinkage intuition; individual coefficient paths need not be monotonic. |
| 2 | 5/10 | Stated penalty definitions, but not the mathematical mechanism. |
| 3 | 4/10 | Correct L1 tendency; L2 often distributes rather than eliminates correlated coefficients. |
| 4 | 7/10 | Correct combination; did not mention grouped behavior/stability. |

### Correct answer and explanation

**1 — Regularization strength.** Stronger penalization generally reduces the magnitude of the penalized coefficient vector. Very strong L1 can set coefficients exactly to zero. L2 typically approaches zero smoothly for finite penalty values, except in special degenerate cases. Individual coefficients can move non-monotonically under correlated predictors. The intercept is often not penalized.

**2 — Derivatives and exact zeros.** For one coefficient,

\[
\frac{d}{dw}\lambda w^2=2\lambda w.
\]

This derivative shrinks toward zero as the weight approaches zero. For L1, the derivative away from zero is \(\lambda\operatorname{sign}(w)\); at zero, there is a *subgradient interval* \([-\lambda,\lambda]\). The non-smooth corner makes exact-zero solutions possible. For the toy problem \(\min_w \frac12(w-a)^2+\lambda|w|\), the solution is

\[
\hat w=\operatorname{sign}(a)\max(|a|-\lambda,0).
\]

This is **soft thresholding**: if \(|a|\le\lambda\), the estimate is exactly zero. The corresponding L2 solution is \(\hat w=a/(1+2\lambda)\).

**3 — Correlated biomarkers.** L1 may select one from a group of correlated features and suppress others, sometimes inconsistently across folds. L2 tends to spread the predictive weight across correlated variables. With *identical* standardized features and fixed total coefficient 1, the pair (1, 0) and (0.5, 0.5) have the same L1 penalty (1), but their L2 penalties differ (1 versus 0.5). Therefore L2 favors sharing weight in this example. These are tendencies, not guarantees.

**4 — Elastic Net.**

\[
\mathcal L=\mathcal L_{\mathrm{data}}+\lambda_1\|w\|_1+\lambda_2\|w\|_2^2.
\]

L1 encourages sparse predictors, and L2 stabilizes estimates when biomarkers are correlated. Choose strengths using patient-grouped validation. Standardization is necessary for a fair penalty comparison. A selected feature need not be causal.

**My recurring weakness:** I knew *what* L1/L2 do but not *why*; I also assumed L2 would suppress one correlated variable, which generally misses its grouping/shrinkage behavior.

---

## Question 3 — Patient-grouped validation, model-selection bias, and leakage

### The question

You have wearable windows from 500 patients and want to select logistic-regression regularization from \(\lambda\in\{0.001,0.01,0.1,1,10\}\). A colleague proposes randomly splitting individual windows into five folds, tuning \(\lambda\), and reporting the highest validation AUROC as final model performance.

1. What is wrong with splitting windows rather than patients?
2. Why is reporting the best tuning AUROC potentially optimistic?
3. Design a fair hyperparameter selection and generalization evaluation procedure.
4. When should the feature scaler be fitted, and how should it be applied to test data?

### My original answers

> **1.** the window split may cause data leakage, since data from the same patient may be split to both training and testing set, and they may have samilar patterns such as resting heart rate. That will cause false optimistic on performance. At the same time, patient level spliting can make sure the testing performance is from unseen data.
>
> **2.** First, we haven't check the distribution of classes, if there is class imbalance, then auroc wouldn't be a best choice since the model may predict all sample as the major class. Second, using the best cross-validation auroc cannot represent the best performance, it should train on the whole training set again and tested on the testing set. Also, patients have multiple observations, and the cross validation the specific split may generated from a good split, but it may not work well in other sets.
>
> **3.** i will use a nested cross validation, held out a testing set, and for the outer training set, i split them into training set and val set to choose the best parameter, and set different lamda values to repeat the experiments. And finally test on the testing set to get the results.
>
> **4.** for the held out testing set, i won't touch it. for the training set, i would do the standardization for each fold. Because the testing set should be unseen and untouched, remaining the original form.

### Feedback on my answers

| Part | Coaching score | Assessment |
|---|---:|---|
| 1 | 9/10 | Strong patient-identity leakage detection. |
| 2 | 6/10 | Recognized tuning vs. testing but missed selection bias; AUROC comment imprecise. |
| 3 | 7/10 | Sound separation of datasets but confused holdout testing with nested CV. |
| 4 | 7/10 | Fit scaler within folds, but test data must still be transformed. |

### Correct answer and explanation

**1 — Patient grouping.** Observations from the same patient share physiology, behavior, and measurement conditions. Group all windows for each patient into one partition. Random window splitting can overestimate generalization to unseen patients.

**2 — Model-selection optimism.** Each CV score is noisy: \(\widehat M(\lambda)=M(\lambda)+\epsilon_\lambda\). Selecting the largest observed value partly selects favorable noise. Therefore the maximum tuning AUROC is not an independent estimate of the selected procedure's generalization. Class imbalance deserves attention, but AUROC measures rankings across thresholds; an all-tied score has AUROC 0.5, not good performance.

**3 — Two valid designs (do not confuse them).**

- **Holdout design:** split patients into development and untouched test cohorts; use grouped CV inside development to select \(\lambda\); refit the full selected pipeline on development; evaluate once on test.
- **Nested grouped CV:** outer folds evaluate the *complete model-selection procedure*; each outer-training partition contains an independent inner CV for tuning; aggregate outer evaluations. A separate final test cohort is optional if available and must be reserved consistently.

**4 — Preprocessing.** Fit imputation, standardization, feature selection, and any sampling strategy inside training folds only. Transform held-out observations using **training parameters**, for example

\[
z_{\mathrm{test}}=\frac{x_{\mathrm{test}}-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}.
\]

Do *not* leave test inputs unscaled when the trained model expects standardized features. In scikit-learn, logistic-regression parameter `C` controls inverse penalty strength: smaller `C` means more regularization.

**My recurring weakness:** I often know to keep test data independent but need to say precisely what information is *fitted* on training data versus *applied* to validation/test data.

---

## Question 4 — ROC, precision–recall, and rare outcomes

### The question

Disease prevalence is 2%. Evaluate two models:

| Metric | Model A | Model B |
|---|---:|---:|
| AUROC | 0.94 | 0.88 |
| AUPRC | 0.16 | 0.42 |
| Sensitivity at a chosen threshold | 90% | 85% |
| Precision at the same model-specific operating point | 8% | 30% |

1. Why can A have higher AUROC but lower AUPRC?
2. Why is AUPRC especially informative at 2% prevalence?
3. Which model would you choose for screening, and how do clinical error costs affect the decision?
4. Clinically interpret 30% precision.

### My original answers

> **1.** because model a may predicted most samples as negative, most of the positive samples are not detected, since there are more false negative samples.
>
> **2.** because auprc can show there are less negative samples are predicted as positive, that contribute to clinical decisions
>
> **3.** i prefer model B because it has acceptable AUROC and high AUPRC, and higher precision.
>
> **4.** it means in all samples predicted as positive, how many of them are truly positive (having the disease).

### Feedback on my answers

| Part | Coaching score | Assessment |
|---|---:|---|
| 1 | 3/10 | Confused an individual threshold with an area-under-curve metric. |
| 2 | 5/10 | Some FP intuition; missing prevalence effect and formal definitions. |
| 3 | 8/10 | Model B is defensible; clinical costs must be specified. |
| 4 | 9/10 | Correct meaning, but include the actual 30% interpretation. |

### Correct answer and explanation

**1 — Different aspects of performance.** ROC is TPR versus FPR across all score thresholds; the area reflects ranking positive versus negative patients. PR is precision versus recall across thresholds. When negatives are abundant, a small FPR can create many false alarms, sharply lowering precision even when the ROC curve is strong. AUC metrics are **not** the count of predictions made at one threshold.

\[
\mathrm{TPR}=\frac{TP}{TP+FN},\quad
\mathrm{FPR}=\frac{FP}{FP+TN},\quad
\mathrm{Precision}=\frac{TP}{TP+FP}.
\]

**2 — Prevalence and PR.** If \(\pi=P(Y=1)\), then

\[
\mathrm{Precision}=\frac{\mathrm{TPR}\,\pi}{\mathrm{TPR}\,\pi+\mathrm{FPR}(1-\pi)}.
\]

A noninformative ranking has a PR reference around prevalence (0.02). With many negatives, small FPRs yield sizable FP counts, so precision is an important practical measure. AUPRC varies with prevalence even if class-conditional score distributions remain the same.

**3 — Clinical choice.** Model B offers far better AUPRC and 30% versus 8% precision, with 85% versus 90% sensitivity at the stated operating points. Prefer B if the follow-up burden of false positives is substantial; consider A if missing cases is vastly more harmful and follow-up testing is safe/cheap. Compare tuned operating points and clinical utility, not just AUROC or AUPRC.

**4 — Precision 30%.** Among 100 patients flagged positive, approximately **30 have the condition and 70 do not** in the evaluated population at that threshold. It does not describe the proportion of diseased patients detected—that is recall.

**My recurring weakness:** I interpreted curve-level metrics as if they referred to one classification threshold. Always specify the conditioning set of a metric.

---

## Question 5 — Choosing a clinical decision threshold

### The question

A cardiovascular-complication model has the following operating points:

| Threshold | Recall | Precision | Specificity |
|---|---:|---:|---:|
| 0.2 | 95% | 20% | 70% |
| 0.5 | 80% | 50% | 90% |
| 0.8 | 55% | 75% | 98% |

1. Which threshold would you initially consider for early screening of dangerous complications?
2. What happens to FP and FN counts if you lower the threshold?
3. Why might 0.5 not be optimal even for probability predictions?
4. How would you explicitly account for different false-negative and false-positive costs?

### My original answers

> **1.** 0.2 maybe, since missing one true positive can cause serious outcomes, so i want to make sure as many patients with the disease are identified.
>
> **2.** when i lower the classification threshold, false positive increase, false negative decreases
>
> **3.** because it has precision of 80%, means 20% of patients may be missed, which may cause serious problems.
>
> **4.** if the threshold is lower, false negative samples will be less, so i would lower the threshold to avoid immense cost.

### Feedback on my answers

| Part | Coaching score | Assessment |
|---|---:|---|
| 1 | 9/10 | Sensible screening intuition; account for FP burden. |
| 2 | 10/10 | Correct monotone count direction, subject to ties/unchanged counts. |
| 3 | 3/10 | Confused 80% *recall* with 80% *precision*. |
| 4 | 6/10 | Intuitive but missing explicit cost minimization. |

### Correct answer and explanation

**1 — Screening threshold.** Threshold 0.2 is a reasonable starting point when missed complications carry high harm, because sensitivity is 95%. But its precision is 20%, so 80% of predicted positives are false alarms. Its suitability depends on resources, interventions, and follow-up burden.

**2 — Threshold movement.** Lowering the threshold makes more patients positive: TP and FP can increase or stay the same; TN and FN can decrease or stay the same.

**3 — 0.5 is not automatically clinically optimal.** At threshold 0.5, **recall** is 80%, so 20% of actual cases are missed; precision is 50%. The default threshold 0.5 corresponds to particular simplified cost assumptions, not universal clinical preferences.

**4 — Cost-sensitive choice.** On representative validation patients, consider

\[
C(t)=C_{FN}\,FN(t)+C_{FP}\,FP(t).
\]

Select the threshold minimizing expected harm (or maximizing clinical utility), with costs determined in the clinical context. If probabilities are calibrated and the only two costs are false-positive and false-negative loss, predict positive when

\[
C_{FP}(1-p)<C_{FN}p
\quad\Longleftrightarrow\quad
p>\frac{C_{FP}}{C_{FP}+C_{FN}}.
\]

For \(C_{FN}=9\) and \(C_{FP}=1\), the simplified threshold is 0.1. These assumptions may fail when actions have other harms or benefits.

**My recurring weakness:** I understand that missing cases matters but need a quantitative decision rule and stricter precision/recall vocabulary.

---

## Question 6 — Weighted BCE, oversampling, and threshold adjustment

### The question

Among 10,000 patients, 100 have the complication and 9,900 do not. A neural network trained with ordinary BCE poorly detects positives. Compare class-weighted BCE, minority-class oversampling, and threshold adjustment.

1. Why can standard BCE struggle with 1% positives, and does imbalance force all-negative predictions?
2. What does class-weighted BCE change?
3. What are the advantages and risks of oversampling, especially with multiple wearable windows per patient?
4. How does class weighting differ from threshold adjustment, and what can happen to probability calibration?

### My original answers

> **1.** ordinary bce loss won't be too large for positive cases if they were predicted mistakenly. And the loss will be big if the negative samples are predicted wrongly because there are more of them.  No, class imbalance doesn't necessarily make the model predict everyone as negative. That's my intuition, but Idon't know why.
>
> **2.** the class weighted binary cross entropy will assign more loss for fewer wrong predicted samples than majority. So the model will assign more weight to the minor class.
>
> **3.** advantage: increase the number of minority class, and ease the imbalance. risk: no new information added, and could cause overfitting. if each patient have multiple wearable data windows, i will make sure oversample ramdomly the same mount for each wearable modality.
>
> **4.** using class weights will increase the loss of minority class samples' wrong prediction, changing the decision threshold  won't affect the loss, only change the output threshold. class weight may affect probability calibration

### Feedback on my answers

The original feedback did not assign numeric per-part scores. The qualitative assessment was: **Part 1 partially correct with an important loss misunderstanding; Part 2 correct intuition; Part 3 strong oversampling intuition but wrong focus on modalities; Part 4 strong distinction between changing training and changing decisions.**

### Correct answer and explanation

**1 — BCE loss.**

\[
\ell(y,p)=-y\log p-(1-y)\log(1-p).
\]

A positive case assigned \(p=0.01\) incurs a large loss \(-\log(0.01)\approx4.61\), not a small one. The issue is the *aggregate prevalence of many negative loss terms*, potentially skewing the fitted solution when positive signal is weak. Class imbalance does not force an all-negative classifier; predictive features can still distinguish minority cases.

**2 — Weighted BCE.**

\[
\ell_w(y,p)=-w_+y\log p-w_-(1-y)\log(1-p).
\]

A larger positive weight increases the importance of positive examples during training and can change learned parameters.

**3 — Oversampling.** Sample minority examples more frequently *inside training folds only*. It improves the frequency of minority-class learning signals but does not create independent biological information. Oversampling windows naively can overrepresent patients with many recordings. **Split by patient first**; consider patient-level or hierarchical sampling when appropriate. Leave evaluation cohorts representative of the deployment population.

**4 — Training versus decision.** Class weighting changes the objective and can change score distributions; lowering a threshold leaves learned scores unchanged but makes more predictions positive. Altering training class proportions or weights may impair calibration to the original disease prevalence. Check and, if needed, recalibrate on representative held-out calibration patients.

**My recurring weakness:** I conflated the size of an individual loss with the aggregate influence of many examples, and I initially treated modality balance as the key oversampling issue rather than patient independence.

---

## Question 7 — Calibration, the Brier score, and post-hoc recalibration

### The question

Five groups each contain 200 patients:

| Predicted complication risk | Observed complications |
|---:|---:|
| 10% | 20 |
| 30% | 80 |
| 50% | 100 |
| 70% | 140 |
| 90% | 180 |

1. Which groups are well calibrated? Show the numerical reasoning.
2. Can a model have the same AUROC but systematically overestimated probabilities?
3. Interpret the Brier score and what happens when squared prediction errors increase.
4. Name a genuine recalibration method and explain why the final test set must not be used to fit it.

### My original answers

> **1.** group 1 3 4 5 are well calibrated, and group 2 is less well calibrated. for good calibrated groups, the observed complications equal to risk times number of patients
>
> **2.** yes, AUROC can show the model have ability to rank positive above negative samples, so if the model has good AUROC, it can still provide useful discrimination.
>
> **3.** brier score measures average square distance between probability and groundtruth. if the probabilities are systematically further from true label, the brier score will be bigger.
>
> **4.** oversampling or downsampling some classes. if i apply this to the final test set, it will change the original distribution of classes

### Feedback on my answers

| Part | Coaching score | Assessment |
|---|---:|---|
| 1 | 10/10 | Correct group calibration. |
| 2 | 9/10 | Strong discrimination-versus-calibration understanding. |
| 3 | 9/10 | Correct Brier-score definition and direction. |
| 4 | 3/10 | Oversampling/undersampling are not recalibration techniques. |

### Correct answer and explanation

**1 — Group calibration.** The observed proportions are 10%, **40%**, 50%, 70%, and 90%. Groups 1, 3, 4, and 5 match predictions. Group 2 predicts 30% when the actual observed frequency is 40%, so it *underestimates* risk by 10 percentage points. Finite-sample matching is not proof of perfect population calibration.

**2 — Same discrimination, different calibration.** AUROC depends on score rankings. A strictly monotone probability transformation preserves ranking and therefore preserves AUROC, although it may systematically overstate risk.

**3 — Brier score.**

\[
BS=\frac1N\sum_i (p_i-y_i)^2.
\]

A lower value indicates better squared probabilistic predictions. It is not only a calibration statistic: \(\mathbb E[(p-Y)^2]=(p-q)^2+q(1-q)\) for a Bernoulli true probability \(q\). The pooled Brier score implied by the stated five groups is **0.178**, based on their fixed probabilities and observed binary counts.

**4 — Genuine recalibration.** Examples: Platt scaling, isotonic regression, or temperature scaling. Binary temperature scaling uses \(\sigma(z/T)\), where \(z\) is a logit and \(T\) is estimated using independent calibration data. Apply the learned mapping to test predictions *without using test labels to fit it*. Oversampling adjusts training sample exposure and is not itself a probability-calibration procedure.

**My recurring weakness:** I sometimes mix up a data-balancing intervention with a model-output correction. State what is changed: data frequency, learned parameters, probability mapping, or decision threshold.

---

## Question 8 — Confidence intervals, clustered bootstrap, and random seeds

### The question

On 200 independent test patients, 30 have the target complication. Logistic regression yields AUROC 0.83; a multimodal transformer yields AUROC 0.86. A paired analysis estimates

\[
\Delta\mathrm{AUROC}=0.03,\qquad95\%\ \mathrm{CI}=[-0.02,0.08].
\]

1. Can we conclude that the transformer is better?
2. What does the 95% interval mean, and why does including zero matter?
3. How should bootstrap resampling respect repeated wearable windows from each patient?
4. How do multiple training random seeds differ from bootstrapping evaluation patients?

### My original answers

> **1.** no, because first 0.03 of improvement is small, second the 95% CI include 0, which may have the possibility of coincidently improve.
>
> **2.** I am not sure...
>
> **3.** I will resample the data in the patient level to prevent leakage
>
> **4.** different random seeds investigate in different random state of choosing initial parameters and training and testing set split. bootstraping investigate different combinations of data can influence model's performance.

### Feedback on my answers

| Part | Coaching score | Assessment |
|---|---:|---|
| 1 | 8/10 | Correct cautious conclusion; 0.03 is not necessarily clinically trivial. |
| 2 | Not answered | Priority statistics gap. |
| 3 | 7/10 | Correct resampling unit; primary concern is statistical dependence, not leakage. |
| 4 | 8/10 | Strong intuition; separate training randomness from changes in test partitions. |

### Correct answer and explanation

**1 — Evidence for a difference.** The observed point estimate is +0.03, but the interval permits differences ranging from −0.02 to +0.08. It therefore does not establish a positive improvement at the corresponding conventional two-sided 5% significance level. It does *not* prove that the models are equal.

**2 — Frequentist meaning.** Imagine repeating the whole study and CI construction under the same assumptions. Across repetitions, approximately 95% of the resulting intervals would cover the fixed true population difference. The frequentist interpretation is **not** “there is a 95% probability that the parameter lies in this observed interval.” The interval including zero means the observed data are statistically compatible with a zero difference under the interval method.

**3 — Cluster bootstrap.** Resample patient IDs **with replacement**; if a patient is selected, carry the whole corresponding cluster of observations. Recompute both models' metrics using **the same resampled patients** to preserve pairing. If the outcome is per patient, derive a principled patient-level prediction first. This preserves dependence; leakage prevention is instead about separation of training and held-out patients.

**4 — Sources of variation.** Holding partitions fixed while changing training seeds investigates random initialization, batches, dropout, and optimization stochasticity. Bootstrapping independent evaluation patients investigates finite-sample uncertainty in estimated performance. Deliberately changing splits introduces a third source of variation; don't silently combine it with the seed effect.

**My recurring weakness:** precise confidence-interval interpretation and how to match the bootstrap unit to the sampling/clinical unit. Avoid claiming “the models are the same” merely because a CI includes zero.

---

## Question 9 — External-hospital validation and distribution shift

### The question

A multimodal model trained at Hospital A is evaluated without retraining at Hospital B:

| Metric | Hospital A | Hospital B |
|---|---:|---:|
| Patients | 5,000 | 2,000 |
| Disease prevalence | 10% | 2% |
| AUROC | 0.88 | 0.72 |
| AUPRC | 0.45 | 0.12 |
| Brier score | 0.08 | 0.14 |

Hospital B uses different wearable devices, serves a different demographic mix, and has more missing physiological measurements.

1. Give three mechanisms that could explain reduced performance.
2. Can lower prevalence explain reduced AUPRC? Can prevalence *alone* explain reduced AUROC?
3. What would you investigate before modifying the model?
4. What adaptation or robustness approaches would you evaluate, and how would you test them independently?

### My original answers

> **1.** 1. data format misalignment / more missing data /  data type difference, difference patient demographics make model confused because it hasn't seen this type of number, wearable sensor data format differences cause the model parameter and metrix not well aligned, missing data cause information defect, all of them may cause performance drop.
>
> **2.** when recall increases, precision drops a lot, means there are more false positive than false negative shown, which means more negative examples are recognized as positive. no, not necessarily explain AUROC, because it may still have the same ability of ranking positive above negative samples.
>
> **3.** first i would examine the data and class distribution to see if the two hospital's data are the same. also, i will check if the hypermeters are the same to make sure the training and evaluation process are exactly the same.
>
> **4.** I will use dataset like other hospital to retrain the model, to make it more robust. also I will do the implement for the data of hospital b, recalibrate the model, set different decision threshold to adapt.

### Feedback on my answers

| Part | Coaching score | Assessment |
|---|---:|---|
| 1 | 8/10 | Realistic causes; use sharper statistical language. |
| 2 | 6/10 | Correct AUROC conclusion; prevalence–precision relationship was not clearly explained. |
| 3 | 6/10 | Good initial checks; missing subgroup, missingness, outcome definition, and calibration analyses. |
| 4 | 6/10 | Plausible approaches but did not distinguish adaptation, calibration, and thresholds. |

### Correct answer and explanation

**1 — What might shift?** Covariate shift \(P_A(X)\ne P_B(X)\); prior/label shift \(P_A(Y)\ne P_B(Y)\); concept shift \(P_A(Y\mid X)\ne P_B(Y\mid X)\); measurement shift across devices; and missingness-pattern shift. These theoretical categories can overlap in practice.

**2 — Prevalence.**

\[
\mathrm{Precision}=\frac{\mathrm{TPR}\pi}{\mathrm{TPR}\pi+\mathrm{FPR}(1-\pi)}.
\]

At fixed TPR and FPR, reducing prevalence reduces precision and often AUPRC. Under a *pure prior shift* with unchanged class-conditional score distributions, AUROC is unchanged because ranking comparisons between positives and negatives have the same distribution. An AUROC drop from 0.88 to 0.72 requires additional investigation, including sampling uncertainty and possible measurement or relationship changes.

**3 — Diagnosis before treatment.** Verify input units, sensor sampling rates, feature coding, outcome definitions, prediction horizon, demographics, class balance, missingness rates/mechanisms, score distributions, calibration, and subgroup performance. Check whether a model trained to distinguish hospitals can identify domain differences. Do not retune a frozen external validation model without explicitly redefining the evaluation protocol.

**4 — Matched methodological experiments.** Consider harmonization, multiple-hospital training, missingness-aware representations, and domain adaptation. Recalibration may help probabilistic reliability; threshold changes may help clinical decisions; neither necessarily restores AUROC. For a generalization claim, keep Hospital B untouched. If explicitly adapting to B, separate B adaptation/validation from independent B test patients. Compare matched models with uncertainty and clinically meaningful outcomes.

**My recurring weakness:** diagnosing a failing model is not the same as listing fixes. Name the shift, its measurable implication, a hypothesis, and a controlled experiment.

---

## Question 10 — Full multimodal research design

### The question

You have wearable time series plus EHR data from 1,000 patients; multiple windows are available per patient and 5% have the complication. Preliminary results from **one random observation-level split** are:

| Model | AUROC | AUPRC |
|---|---:|---:|
| Logistic regression | 0.80 | 0.25 |
| Wearable-only LSTM | 0.83 | 0.31 |
| EHR-only MLP | 0.82 | 0.29 |
| Multimodal transformer | 0.87 | 0.38 |

How can you show that the transformer's apparent improvement is real, reproducible, and clinically useful rather than leakage or a favorable experiment?

1. Propose a proper experimental design: splits, preprocessing, tuning, testing.
2. Propose fair baselines and component ablations.
3. Specify metrics, class-imbalance handling, patient dependencies, calibration, and uncertainty analysis.
4. State one falsifiable methodological hypothesis for external-hospital robustness and an experiment that could refute it.

### My original answers

> **1. Experimental design:**
> 1. use 5 different random seeds, each experiment split the patients to five folds, one as testing, one as val set, others as training set. Standardize and normalize the training set, implement the missing values. for hyperparameter fine-tuning, i will conduct 5-fold cross validation, choose the hyperparameter set of the best validation set, and retrain the whole training+val set, then test on the testing set.
>
> **2. Baselines and ablations:**
> I will use the same multimodal data to train some baseline models such as LR, SVM, RF to see the performance, also i will use single modal to train and test the transformer to do the ablation experiments.  also, I will set different hyperparameters of transformer such as layers.
>
> **3. Evaluation and uncertainty:**
> AUROC, AUPRC, Accuracy, precision, recall and F1 score. I will use class weight to handle imbalance, use ? to solve patient level correlations, and plot the calibration figure, scale the prediction probablity, and for uncertainty, i am not sure.
>
> **4. Methodological hypothesis and experiment:**
> i would check the distribution and definition of both hospitals' data, my hypothese is the true performance for hospital B won't be statistically worse than hospital A. I will bootstrap the data and repeat experiments several times and check the confidence interval (CI), to see if it contains 0. if so, there won't be a statistical outperformance of hospital A.

### Feedback on my answers

| Part | Coaching score | Assessment |
|---|---:|---|
| 1 | 7/10 | Good elements; split/seeds/CV roles not unambiguously defined. |
| 2 | 7/10 | Strong baseline instincts; ablations need to isolate a mechanism. |
| 3 | 5/10 | Broad metric list; missing clustered uncertainty procedure. |
| 4 | 4/10 | Proposed a performance-comparison null rather than a new falsifiable method. |

Approximate average on the four parts: **5.8/10** (diagnostic, not a standardized proficiency measure).

### Correct research-level answer

**1 — Experimental design.** Define the clinical prediction unit and horizon. Group all data belonging to a patient. Choose either a held-out patient-level test cohort plus grouped development CV, or nested grouped CV. Fit scaling, imputation, feature engineering, class rebalancing, and calibration in appropriate development partitions only. Use the same patient partitions for all candidate models. Refit the selected pipeline on available development data and evaluate once on untouched test patients. Keep partitions fixed when varying random training seeds so training randomness is distinguishable from sample variation. Given only about 50 positive cases overall, report appropriately broad uncertainty and be wary of overly small positive counts in folds.

**2 — Fair comparisons.** Include wearable-only and EHR-only models, logistic/forest baselines with correctly engineered comparable features, concatenation + MLP, late fusion, and the full transformer. To isolate the proposed mechanism, ablate cross-modal attention, missingness indicators, and modality dropout one by one. Keep tuning budgets, outcome definitions, test patients, and representation access as comparable as possible. Changing layer count is an architecture sensitivity test, not necessarily a targeted fusion ablation.

**3 — Evaluation.** Emphasize AUROC and AUPRC (with prevalence noted), calibration curves, Brier score, and preselected clinically useful sensitivity/specificity/precision operating points. Consider class weighting **within training** and examine its effect on calibration. Use patient-level predictions for patient-level outcomes, or a repeated-measures evaluation protocol for time-varying outcomes. Estimate *paired* differences via patient-cluster bootstrap or another justified paired procedure. Check training variability with multiple fixed-partition seeds, and quantify subgroup and external performance. Avoid overinterpreting accuracy in a dataset with 95% negative cases.

**4 — Falsifiable methodological hypothesis.**

- **Problem:** sensor availability and measurement properties differ across hospitals.
- **Method:** introduce missingness-aware multimodal representations with explicit missingness masks and modality dropout.
- **Hypothesis:** this training method improves discrimination at a held-out hospital whose missingness pattern differs from development hospitals, relative to a matched standard transformer.
- **Experiment:** train both models on the same development hospitals, reserve a distinct hospital for external testing, control tuning budgets and training seeds, and prespecify an external AUROC/AUPRC comparison plus calibration/subgroup checks. Construct paired uncertainty intervals using appropriate patient units.
- **Possible refutation:** no meaningful or reliable improvement, degraded calibration, or gains only under unrealistic artificial missingness.

**Statistics correction:** A CI containing zero is *not* a demonstration of equivalent performance. To establish non-inferiority/equivalence, prespecify a scientifically justified margin and apply the appropriate statistical analysis.

**My recurring weakness:** the transition from practical model engineering to a specific scientific contribution: **problem → proposed mechanism → testable hypothesis → controlled evaluation → evidence and possible refutation**.

---

# Part II — Technical Reference Notes and Worked Examples

This retains the concise technical explanations, equations, practical examples, comparison tables, and Python exercise from the earlier Day 2 study summary. Use Part I for the exact question-and-answer progression; use Part II for quick topic-based revision.

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

# Part III — Personal Weakness Analysis and Improvement Plan

## Overall evaluation

The day showed a clear pattern: **strong practical intuition about medical-model evaluation**, but **less consistent mathematical reasoning and less precise formulation of testable scientific contributions**. This is not a lack of relevant experience; it is a gap between naming an appropriate technique and explaining exactly what quantity it changes, why it helps, and what evidence would demonstrate success.

These judgments use only the ten submitted responses and the feedback recorded above. The diagnostic scores were assigned during the teaching session rather than by a standardized assessment rubric; they should not be interpreted as an objective measurement of overall ability.

### Strengths to retain

1. **Patient-aware evaluation:** quickly recognizes why wearable windows from the same patient cannot be freely mixed across splits (Questions 3, 6, 8, 10).
2. **Clinical decision intuition:** understands why missed complications and unnecessary testing matter and often chooses a sensible screening operating point (Questions 4–5).
3. **Core classification concepts:** increasingly accurate definitions of precision, recall, AUROC, calibration, and the Brier score (especially Question 7).
4. **Class-imbalance techniques:** correctly distinguishes loss weighting from post-training threshold changes and recognizes oversampling does not produce new independent cases (Question 6).
5. **Breadth of practical ideas:** suggests classical baselines, unimodal comparisons, recalibration, and multi-hospital training (Questions 9–10).
6. **Awareness of limits:** explicitly identifies uncertainty instead of constructing a confident but unsupported statistical explanation (Questions 8 and 10).

### Weaknesses ranked by urgency

| Priority | Weakness | Evidence in my responses | Correction to internalize |
|---|---|---|---|
| **1 — Critical** | **Frequentist uncertainty and statistical claims** | Could not interpret a 95% CI (Q8); treated a CI containing zero as suggesting no hospital-performance difference (Q10). | A CI reflects repeated-sampling coverage; including zero does not prove equality or non-inferiority. Use paired, patient-aware inference. |
| **2 — Critical** | **Falsifiable methodological hypothesis** | Q10 proposed “Hospital B is not statistically worse” rather than a methodological change and predicted mechanism. | State a concrete failure mode, propose a method, predict what it changes, and define an independent test plus evidence that would refute it. |
| **3 — High** | **Nested validation and information flow** | Q3 and Q10 mixed random seeds, inner tuning, held-out tests, and outer folds; thought test features should remain unscaled (Q3). | Explicitly assign roles to each partition and fit every learned preprocessing step only on its training partition. |
| **4 — High** | **Probability and metric mathematics** | Confused AUPRC with a single threshold (Q4), confused precision with recall (Q5), and did not derive prevalence effects (Q9). | Write TP/FP/FN/TN definitions, the PPV–prevalence relation, and the cost-sensitive decision rule from memory. |
| **5 — High** | **Matched patient-level uncertainty estimation** | Recognized patient resampling but described it mainly as leakage prevention (Q8); did not know the technique in Q10. | Resample independent *patient clusters*, retain same sampled patients for every candidate model, and report paired CI for performance difference. |
| **6 — Medium** | **Choosing fixes that match failure mechanisms** | Treated oversampling as recalibration (Q7) and offered recalibration/threshold adjustment for the AUROC decline at Hospital B (Q9). | Distinguish changing data exposure, model parameters, probability mapping, and binary decisions. |
| **7 — Medium** | **Mathematical mechanism of regularization** | Knew L1/L2 definitions but not derivatives; incorrectly expected L2 to eliminate one of two correlated predictors (Q2). | Derive L1 subgradient, soft-thresholding, L2 gradient, and shared-weight penalty example. |
| **8 — Medium** | **Capacity versus optimization** | Proposed more epochs or larger LR to reduce logistic-regression representational bias (Q1). | Use interaction/nonlinear features or relax excess regularization for representational bias; use optimizer adjustments for non-convergence. |

## What changed compared with Day 1

| Topic | Earlier understanding | Understanding after Day 2 | Remaining action |
|---|---|---|---|
| Bias–variance | Bias/variance relationship initially reversed. | Correctly spots evidence of deep-model overfitting. | Explain formal variance and separate capacity from optimization. |
| L1/L2 | Confusion about which penalty yields sparsity. | Correctly identifies L1 zeros, L2 shrinkage, and Elastic Net combination. | Derive the geometry and correlated-feature behavior. |
| Leakage | Strong instinct for patient-level splitting. | Applies grouping to tuning, resampling, and external tests. | Fully specify nested-CV and preprocessing data flow. |
| Imbalanced metrics | Recognized misleading accuracy and importance of sensitivity. | Can interpret precision, prevalence, AUPRC, and operating-point tradeoffs with feedback. | Stop swapping precision/recall; derive Bayes-based precision. |
| Calibration | Strong conceptual understanding of frequency matching. | Correctly computes group calibration and interprets Brier score. | Learn a genuine recalibration method and its calibration-set requirement. |
| Statistical inference | Limited prior exposure in practice. | Can recognize that a CI containing zero weakens claims. | Learn exact frequentist interpretation and patient-level paired uncertainty. |
| Model comparison | Good intuitive baseline selection. | Suggests multimodal and unimodal controls. | Separate **architecture ablations** from mere hyperparameter sweeps. |
| Research reasoning | Identifies clinically important evaluation problems. | Beginning to propose robustness strategies. | Produce falsifiable problem–mechanism–prediction–test hypotheses. |

## The six most important mathematical statements to know without notes

**1. Group calibration**

\[
\text{Observed frequency in group }g=\frac{\#\text{ events in }g}{\#\text{ patients in }g};
\qquad\text{compare with average predicted risk in }g.
\]

**2. Precision and prevalence**

\[
\mathrm{PPV}=\frac{\mathrm{TPR}\pi}{\mathrm{TPR}\pi+\mathrm{FPR}(1-\pi)}.
\]

**3. Brier score**

\[
BS=\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2.
\]

**4. L1 and L2 derivatives**

\[
\partial_w(\lambda|w|)=
\begin{cases}
\{\lambda\}, & w>0,\\
[-\lambda,\lambda], & w=0,\\
\{-\lambda\}, & w<0,
\end{cases}
\qquad \frac{d}{dw}(\lambda w^2)=2\lambda w.
\]

**5. Simplified cost-based threshold for calibrated risk**

\[
t^*=\frac{C_{FP}}{C_{FP}+C_{FN}}.
\]

**6. Paired metric difference**

\[
\Delta_b=\mathrm{AUROC}_{B,b}-\mathrm{AUROC}_{A,b}
\]

where \(b\) indexes a bootstrap sample of **patients**, applied identically to both models.

## A focused two-hour revision schedule

| Time | Exercise | Completion criterion |
|---|---|---|
| 0–30 min | Confidence intervals, two-sided testing, equivalence vs. non-inferiority | Explain \([-0.02,0.08]\) without using the incorrect “95% probability inside” wording. |
| 30–50 min | Draw the patient-grouped holdout/CV protocol and nested CV | Every data-dependent step has exactly one stated fit population. |
| 50–70 min | Implement paired patient bootstrap using the code in Part II | Function returns a point difference and CI, skips invalid one-class resamples. |
| 70–100 min | Write a one-paragraph missingness-aware fusion research proposal | Includes mechanism, comparison, primary outcome, possible refutation. |
| 100–120 min | Work metric and calibration exercises from Q4–Q7 | Can derive precision, cost threshold, and distinguish recalibration from reweighting. |

## Self-test and answer key

Use this as a fast recall check after revision.

<details>
<summary><strong>Check 1 — CI includes zero: what conclusion?</strong></summary>

With a valid two-sided 95% CI including zero, the data do not establish a nonzero difference at the corresponding 5% significance level. They do **not** establish equality; a useful estimate of practical relevance also requires an appropriate effect-size context and uncertainty margin.

</details>

<details>
<summary><strong>Check 2 — Why fit the scaler inside each fold?</strong></summary>

Because mean and standard deviation are fitted parameters. Letting validation/test observations influence these estimates leaks information across partitions. Transform held-out observations using the training-fold parameters.

</details>

<details>
<summary><strong>Check 3 — Why can AUPRC fall when prevalence falls?</strong></summary>

At fixed true-positive and false-positive rates, a smaller prevalence gives a smaller positive predictive value via Bayes' theorem. Thus PR performance depends on base rate. AUROC is unchanged under a pure prevalence shift with unchanged class-conditional score distributions.

</details>

<details>
<summary><strong>Check 4 — What does patient-level bootstrapping preserve?</strong></summary>

Each sampled unit is an independent patient cluster, not an independently sampled wearable window. The same set of sampled patient IDs is applied to both models for paired metric comparisons.

</details>

<details>
<summary><strong>Check 5 — What distinguishes an ablation from a parameter sweep?</strong></summary>

An ablation removes or replaces a proposed mechanism while controlling the remaining setup, such as removing cross-modal attention. Changing transformer depth is a sensitivity/capacity investigation unless depth itself is the proposed mechanism.

</details>

<details>
<summary><strong>Check 6 — What is a falsifiable robustness hypothesis?</strong></summary>

Example: “A missingness-aware multimodal model with modality dropout improves patient-level AUPRC on an untouched external hospital compared with a matched baseline.” A carefully measured lack of improvement, or only unrealistic artificial-masking gains, would fail to support the hypothesis.

</details>

## Reference-quality response template for future projects

When describing an experimental study, use the following sequence:

> **Goal:** Define the target population, prediction horizon, patient-level endpoint, and use case.
>
> **Problem:** Identify a specific generalization or modeling failure.
>
> **Proposed mechanism:** Describe exactly what changes in model design or training, and why it should matter.
>
> **Hypothesis:** Specify the direction of a measurable effect on a prespecified primary metric.
>
> **Evaluation:** Define independent patient grouping, fold-local preprocessing, fair controls, and independent external testing.
>
> **Uncertainty:** Quantify paired differences with an analysis that respects patient clusters; separate sampling uncertainty from training randomness.
>
> **Limitations:** Explain what result would fail to support the hypothesis, what remains untested, and whether the observed effect is clinically meaningful.

---

## Closing learning reflection

**What I can now explain confidently:** patient leakage, the basic distinction between discrimination and calibration, clinical sensitivity/precision tradeoffs, class weighting, and the value of unimodal versus multimodal comparisons.

**What I should not yet present as mastered:** frequentist confidence intervals, nested model-selection details, patient-cluster uncertainty implementation, distribution-shift diagnosis, and falsifiable methodological contributions.

**Next step:** revise the high-priority gaps, implement the patient bootstrap, and practice one complete problem → mechanism → hypothesis → controlled comparison → uncertainty analysis narrative.
