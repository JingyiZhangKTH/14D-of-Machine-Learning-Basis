# Day 1 ML Interview Study Notes

**Topic:** Machine learning fundamentals, mathematical intuition, and rigorous medical ML evaluation  
**Format:** Original practice question → my original answer → corrected, interview-ready answer → key lesson  
**Status:** 10/10 questions completed

> **How to revise:** Hide the **Corrected answer** under each question, answer aloud in 60–90 seconds, then compare. On a second pass, reproduce the key equations and identify what evidence an ML researcher would require.

## Contents

1. [Training fit vs. generalization](#q1-training-fit-vs-generalization)
2. [Bias–variance tradeoff](#q2-biasvariance-tradeoff)
3. [Overfitting and early stopping](#q3-overfitting-and-early-stopping)
4. [Linear vs. logistic regression](#q4-linear-vs-logistic-regression)
5. [L1 vs. L2 regularization](#q5-l1-vs-l2-regularization)
6. [Grouped cross-validation and leakage](#q6-grouped-cross-validation-and-leakage)
7. [Class imbalance and clinical metrics](#q7-class-imbalance-and-clinical-metrics)
8. [Calibration vs. discrimination](#q8-calibration-vs-discrimination)
9. [Gradient descent and backpropagation](#q9-gradient-descent-and-backpropagation)
10. [Rigorous model comparison and ablations](#q10-rigorous-model-comparison-and-ablations)
11. [Formula sheet](#formula-sheet)
12. [My recurring misconceptions](#my-recurring-misconceptions)
13. [Spaced-repetition checklist](#spaced-repetition-checklist)

---

## Q1. Training fit vs. generalization

### Question

Two models predict cardiovascular disease from wearable sensor data from 1,000 patients:

|Model|Training accuracy|Test accuracy|
|-|-:|-:|
|A: Neural network|99%|72%|
|B: Logistic regression|85%|83%|

1. Which model would you prefer and why?
2. What explains Model A's large training/test performance gap?
3. What would you check before using Model B clinically?

### My original answer

> 1. Model B, because it performs better on the unseen dataset, although the training accuracy isn't the best.
>
> 2. the model is underfitting or overfitting?
>
> 3. Do some external validation to see how the performance is.

<details>
<summary><strong>Corrected answer — reveal after trying</strong></summary>

**1.** Provisionally choose **Model B**: it has higher held-out accuracy (83% vs. 72%) and a smaller train/test gap (2 vs. 27 percentage points). The goal is generalization, not memorizing the training examples.

**2.** Model A shows a pattern **consistent with overfitting**: very low training error and much higher test error. Its complexity, lack of regularization, small sample size, patient-specific patterns, or shift between train and test populations could contribute. The gap alone does not prove which cause dominates.

**3.** Before clinical use, verify patient-disjoint splits, absence of preprocessing leakage, class balance, sensitivity/specificity, AUROC/AUPRC, calibration, confidence intervals, subgroup robustness, and independent external validation.

For accuracy, a simple *generalization gap* measure is $G=\mathrm{Accuracy}_{train}-\mathrm{Accuracy}_{test}$. Here, $G_A=0.27$ and $G_B=0.02$. A smaller gap is useful evidence but does not alone guarantee high absolute performance.

</details>

**Takeaway:** Generalization measures how well learned patterns transfer to unseen cases. Overfitting means overly strong training fit relative to out-of-sample performance; underfitting means insufficient fit even on training data.

---

## Q2. Bias–variance tradeoff

### Question

You have 200 patients and 100 physiological features. Compare (A) a simple linear model and (B) a 10-hidden-layer deep neural network:

1. Which generally has higher bias, and why?
2. Which generally has higher variance, and why?
3. Which might benefit more from increasing the sample to 20,000 patients?

### My original answer

> 1.  higher bias: model a, linear models may underfit the data.
>
> 2. higher variance: model b, deep neural network may learn too much about the errors.
>
> 3. model b, because the neural network can learn more information and not be overfitting.

<details>
<summary><strong>Corrected answer — reveal after trying</strong></summary>

**1.** Typically **Model A has higher approximation bias** because a linear function is less flexible and may miss nonlinear relations unless appropriate features are engineered.

**2.** Typically **Model B has higher estimation variance**, especially with only 200 patients: a highly flexible model may learn sample-specific noise, so its predictions can be more sensitive to the choice of training patients.

**3.** Model B may benefit more from a large dataset because additional observations help constrain a high-capacity model and may enable reliable learning of nonlinear relationships. This is a tendency, not a guarantee: architecture, regularization, optimization, and the true signal matter.

For squared-error regression at fixed $x$ under standard assumptions:

$$\mathbb E\[(Y-\hat f(x))^2\]=\operatorname{Bias}\[\hat f(x)\]^2+\operatorname{Var}\[\hat f(x)\]+\sigma^2.$$

**Bias:** systematic deviation of the average fitted prediction from the target function. **Variance:** variability of fitted predictions over alternative training samples. This classical decomposition should not be applied uncritically to classification accuracy.

</details>

**Takeaway:** Bias is *systematic model error*; variance is *training-sample sensitivity*. Deep networks do not inevitably have high variance, but model flexibility increases overfitting risk when data are limited.

---

## Q3. Overfitting and early stopping

### Question

An LSTM's validation loss reaches its lowest point around epoch 30, then rises while training loss continues falling until epoch 100.

1. What is happening after epoch 30 and why?
2. Name three strategies to reduce overfitting and explain each.
3. Which checkpoint would early stopping select?

### My original answer

> 1. after epoch 30, the model is overfitting, because the model learns too much information from the trainingset, and its flexbility drops, so it doesn't work well in the validatio set, thus cause the validation loss increases.
>
> 2. add more variable samples / early stopping / decrease the learning rate
>
> 3. at the epoch 30

<details>
<summary><strong>Corrected answer — reveal after trying</strong></summary>

**1.** The model **begins to overfit** (under the assumed comparable data distributions). Continued training fits idiosyncrasies or noise in the training sample. **The architecture's flexibility does not drop**; generalization deteriorates.

**2.** Three defensible strategies:

* **More representative and diverse training patients:** can decrease estimation variance and improve coverage of the target population.
* **Early stopping:** save the checkpoint with best validation performance, rather than always selecting the last epoch.
* **Regularization (e.g., L2 or dropout):** limits certain overly flexible solutions and can improve out-of-sample performance. Reducing model capacity is another option.

**Important correction:** Lowering the *learning rate* changes optimization step size. It can stabilize training but **does not automatically prevent overfitting**, especially if training continues longer.

**3.** Choose approximately **epoch 30**, where validation loss is lowest. In practice, use a patience rule and restore the best checkpoint. Do not select checkpoints based on the final test set.

</details>

**Takeaway:** Overfitting means an increasing generalization gap; early stopping is a form of regularization through model selection over training time.

---

## Q4. Linear vs. logistic regression

### Question

For binary cardiovascular-disease prediction, consider (A) linear regression $\hat y=w^Tx+b$ and (B) logistic regression $p=\sigma(w^Tx+b)$.

1. Why is ordinary linear regression generally unsuitable for disease-probability prediction?
2. What is the purpose of the sigmoid?
3. Why use binary cross-entropy (BCE) rather than mean squared error (MSE) for logistic regression?

### My original answer

> 1. usually the diseases won't strictly be linear with features, if use linear regression will cause underfitting.
>
> 2. to regulate the output to (0,1)
>
> 3. because it's more flexible and get better performance

<details>
<summary><strong>Corrected answer — reveal after trying</strong></summary>

**1.** Ordinary linear regression produces **unbounded values**: a fitted value can be $-0.4$ or $1.6$, which cannot be interpreted as a probability. Logistic regression instead models a Bernoulli conditional probability. A logistic model is still *linear in the log-odds*; it does not automatically model arbitrary nonlinear feature interactions.

**2.** The sigmoid maps any real-valued logit $z$ to $(0,1)$:

$$\sigma(z)=\frac{1}{1+e^{-z}}, \qquad \log\frac{p}{1-p}=w^Tx+b.$$

**3.** For a Bernoulli label $y\in{0,1}$:

$$P(y\mid x)=p^y(1-p)^{1-y}.$$

Taking the negative logarithm yields:

$$L_{\mathrm{BCE}}=-\[y\log p+(1-y)\log(1-p)\].$$

Thus BCE is the **negative log-likelihood of the Bernoulli model**, not merely a more flexible loss. It is a proper scoring rule and produces a convenient derivative with a sigmoid logit:

$$\frac{\partial L_{\mathrm{BCE}}}{\partial z}=p-y.$$

</details>

**Takeaway:** Probability range, Bernoulli likelihood, and log-odds explain why logistic regression with BCE is a principled binary classifier. A linear probability model is possible but has limitations.

---

## Q5. L1 vs. L2 regularization

### Question

You train logistic regression with 500 patients and 100 physiological features. The model overfits. Compare:

$$L_{L1}=L_{\mathrm{BCE}}+\lambda\sum_j|w_j|,\quad L_{L2}=L_{\mathrm{BCE}}+\lambda\sum_jw_j^2.$$

1. Why add a regularization term?
2. What is the main difference in how L1 and L2 affect weights?
3. If only 10 of 100 features are truly useful, which would you initially prefer?

### My original answer

> 1. to prevent the model to be too dependent to the training set, and avoid overfitting. the regularization add something to loss, encourage it to find more suitable weights
>
> 2. the L1 regularization is linear, and the L2 regularization is squared, L1 allows two zeros
>
> 3. L2?

<details>
<summary><strong>Corrected answer — reveal after trying</strong></summary>

**1.** Regularization trades some training-data fit for a preference toward parameter configurations that may generalize better:

$$\hat w=\arg\min_w\big[L_{train}(w)+\lambda R(w)\big].$$

It may introduce bias while reducing variance. The strength $\lambda$ must be tuned using validation data.

**2.** **L1** uses $\sum_j|w_j|$ and can make coefficients **exactly zero** (sparsity / implicit feature selection). **L2** uses $\sum_jw_j^2$ and generally **shrinks** coefficients toward zero without making them exactly zero. For a single nonzero weight, the derivative of the L1 penalty is $\operatorname{sign}(w)$, while the derivative of the squared L2 penalty is $2w$; the L1 penalty has a kink at zero.

**3.** If the true signal is sparse, **L1 is a reasonable initial choice**. But with highly correlated variables (e.g., wearable biomarkers), L1 may choose predictors unstably; L2 or **Elastic Net** (an L1+L2 combination) may perform better. Compare using patient-level validation and examine feature-selection stability.

**Caution:** Scale numeric predictors appropriately before penalizing coefficient sizes, and do not interpret selected predictive features as causal biomarkers.

</details>

**Takeaway:** **L1 = sparsity**, **L2 = smooth shrinkage**, **Elastic Net = combination**. No penalty universally wins.

---

## Q6. Grouped cross-validation and leakage

### Question

There are 200 patients, each contributing 10 windows (2,000 observations). The goal is to generalize to **new patients**.

1. Why is an 80/20 random **window-level** split risky?
2. Which split would you use instead?
3. What is k-fold cross-validation, and how would you implement it here?

### My original answer

> 1. because if use same patient data for both training and testing data, the model may already saw the same patterns, so it's a kind of data leakage. That may cause high performance.
>
> 2. i will use the patient level split, because it can make sure each single patient data is fresh for the model, so we can know it's true performance.
>
> 3. to avoid the random good performance, by averaging the performance, we can know the model's true performance. and it can avoid the influence of bad split of data. i will split the 200 patients' data to 5 folds, each of them contains 40 patients' data, and for each experiment, i use 4 of 5 folds for training, and one of them for testing. And go 5 rounds, making sure each fold has been the testing set. and then average the performance of 5 fold.

<details>
<summary><strong>Corrected answer — reveal after trying</strong></summary>

**1.** Repeated measurements from the same patient share distinctive physiological signatures. Placing their windows on both sides lets a model benefit from **patient-specific information** during evaluation, making new-patient performance look artificially good.

**2.** Split **by patient ID**, not window, so all windows from each patient remain in a single partition. This better estimates transfer to unseen patients. It still does not guarantee performance at a new hospital or future time period.

**3.** Use **grouped 5-fold cross-validation**: split the 200 patients into five disjoint sets (about 40 patients each); train on four sets and validate on the remaining set, repeating five times. All windows from a patient move together. Average the five metrics (and examine variability); CV reduces reliance on one particular split but does not reveal the exact “true performance.”

```python
from sklearn.model_selection import GroupKFold

cv = GroupKFold(n_splits=5)
for train_idx, val_idx in cv.split(X, y, groups=patient_ids):
    # Fit preprocessing and model using train_idx only.
    # Evaluate on val_idx.
    pass
```

**Research-level guardrails:** Fit imputation/scaling/feature selection within each training fold; separate hyperparameter tuning from final testing (nested CV or untouched patient-level test set); report variability/uncertainty. For predicting future outcomes in *known* patients, a carefully designed time-based evaluation may instead match the target use case.

</details>

**Takeaway:** Choose a split that matches the **intended generalization target**; random CV is not necessarily valid for repeated patient measurements.

---

## Q7. Class imbalance and clinical metrics

### Question

A test set contains 1,000 patients: 50 positive and 950 negative. The model predicts **everyone negative**: $TP=0, FN=50, FP=0, TN=950$.

1. Is its 95% accuracy good for screening?
2. Calculate recall (sensitivity) and specificity.
3. If missing cases is costly, prioritize precision or recall? What's the trade-off?

### My original answer

> 1. 95% accuracy seems  good, but it's actually not. Since it predicted every patient healthy, and didn't find out the patients with disease, it is not a good clinical model, actually, it's meaningless.
>
> 2. recall  = 0, specificity = 1.
>
> 3. recall, because it pay attention to how many patients with disease were identified in the group of all patients with disease. the tradeoff is it would pay attention to false negative samples.

<details>
<summary><strong>Corrected answer — reveal after trying</strong></summary>

**1.** No: it detects **zero** true cases, so it is ineffective as a screening classifier at this threshold, despite 95% overall accuracy. Accuracy is dominated by the majority class.

**2.**

$$\mathrm{Recall}=\frac{TP}{TP+FN}=\frac0{50}=0; \qquad \mathrm{Specificity}=\frac{TN}{TN+FP}=\frac{950}{950}=1.$$

Precision, $TP/(TP+FP)$, is **undefined** because the model predicts no positives (some libraries report 0 by convention).

**3.** Prioritize **recall/sensitivity** when false negatives are particularly harmful, while acknowledging that lowering the positive-decision threshold generally increases false positives and may reduce precision and specificity. Threshold selection should weigh the consequences of both types of error and clinical workflow capacity.

**Evaluation:** report recall, specificity, precision, AUROC/AUPRC, and calibration as relevant. AUPRC is particularly informative for rare positive outcomes, though it depends on prevalence.

</details>

**Takeaway:** For imbalanced screening, accuracy alone can be dangerously misleading. Clinical costs determine which errors matter most.

---

## Q8. Calibration vs. discrimination

### Question

Two models both have AUROC 0.90. Among 100 patients each assigned approximately 80% risk, the observed event rates are 78% for Model A and 45% for Model B.

1. Which looks better calibrated for these patients?
2. What is the distinction between discrimination and calibration?
3. Why does calibration matter in clinical decisions?

### My original answer

> 1. model A appears better calibrated, because the observed event frequency is closer to the predicted risk.
>
> 2. the discrimination is a measurement of how well the model can separate positive and negative samples, the calibration is how the predicted possibility is close to the true possibility (to what extent we can trust).
>
> 3. because too high probability for positive samples may cause more smaples are predicted as disease, which may cause extra or unnecessary treatment for false positives.

<details>
<summary><strong>Corrected answer — reveal after trying</strong></summary>

**1.** **Model A** is better calibrated **for this probability group**: 78% is much closer to 80% than 45%. One bin does *not* determine which is better calibrated overall.

**2.** **Discrimination** assesses how well positive cases are ranked above negative cases (e.g., AUROC). **Calibration** assesses whether assigned probabilities agree with observed frequencies across groups of predictions. Ideal calibration informally means:

$$P(Y=1\mid\hat p=p)=p.$$

A monotonic transformation can preserve ranking/AUROC while changing calibration.

**3.** Accurate **absolute risks** matter when thresholds guide monitoring or treatment. **Overestimated** risk can cause unnecessary interventions; **underestimated** risk can cause missed opportunities for beneficial care. Check reliability plots, calibration intercept/slope, and Brier score; recalibration on separate development data may help. Brier score evaluates probabilistic accuracy but is **not purely a calibration metric**.

$$\mathrm{Brier}=\frac1N\sum_{i=1}^N(\hat p_i-y_i)^2.$$

</details>

**Takeaway:** AUROC asks **who ranks higher**; calibration asks **whether the probabilities mean what they say**.

---

## Q9. Gradient descent and backpropagation

### Question

Gradient descent is $\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t)$. Given $L(w)=(w-3)^2$, $w_0=0$, and $\eta=0.1$:

1. Why subtract rather than add the gradient?
2. What happens with a too-large or too-small learning rate?
3. Compute the gradient and updated weight after one step.

### My original answer

> 1. because we want to find the parameters that decrease the loss of wrong predictions, so we decrease the weight of wrong direction, which makes the loss smaller.
>
> 2. if the learning rate is too large, it may miss the smallest gradient point. if it is too small, it may never or take a very long time to reach the smallest gradient point.
>
> 3. current gradient: -6, new weight: 0.6

<details>
<summary><strong>Corrected answer — reveal after trying</strong></summary>

**1.** The gradient points in the direction of steepest **local increase** in a differentiable function. The negative gradient points toward steepest local **decrease** (in Euclidean parameter space). Individual weights may **increase or decrease** depending on gradient sign.

**2.** Too large a step may overshoot, oscillate, or diverge; too small a step can make progress extremely slow. The objective is to reach a **loss minimum**, not “the smallest gradient point.” A zero gradient can also occur at a saddle point or maximum.

**3.**

$$\frac{dL}{dw}=2(w-3)=2(0-3)=-6.$$
$$w_1=0-0.1(-6)=\boxed{0.6}.$$

The loss falls from $L(0)=9$ to $L(0.6)=5.76$.

**Backpropagation extension:** Gradient descent **uses** gradients; backpropagation **computes** them efficiently through the chain rule. For $z=wx$, $p=\sigma(z)$, and BCE:

$$\frac{\partial L}{\partial w}=\frac{\partial L}{\partial p}\frac{\partial p}{\partial z}\frac{\partial z}{\partial w}=(p-y)x.$$

</details>

**Takeaway:** Optimization moves parameters; backpropagation computes derivatives. The sign of the gradient determines the update direction.

---

## Q10. Rigorous model comparison and ablations

### Question

For prediction of 30-day cardiovascular complications from wearables and EHR data:

|Model|AUROC|AUPRC|
|-|-:|-:|
|Logistic regression|0.82|0.41|
|Random forest|0.85|0.45|
|Multimodal transformer|0.88|0.49|

These results come from one random train/test split with repeated observations per patient.

1. Can we conclude the transformer is genuinely better?
2. How would you evaluate rigorously?
3. Which ablations test the multimodal contribution?
4. What if the transformer has higher AUROC but worse calibration than logistic regression?

### My original answer

> 1. i cannot conclude, because the results are just from one single training testing split, we should do a cross validation to examine which model is the best.
>
> 2. use a five fold cross validation , and introduce the val set to improve and choose the best parameters, and then examine which model is teh best.
>
> 3. to do the experiments which only use one of the multimodal data, to examine which datat modal contribute teh most. but the training and testing and all parameters should remain the same as the multimodal experiments, only change the dataset components.
>
> 4. i would check the data and label distribution, if it's a very rare disease, i would look at the auprc. and if the transformers' calibration is too much shifted from the observed event, i would choose logistic regression if it has acceptable recall.

<details>
<summary><strong>Corrected answer — reveal after trying</strong></summary>

**1.** **No.** One random split may be unstable, and repeated patient windows on both sides could cause leakage. A 0.03 AUROC improvement over the random forest might not be robust, statistically distinguishable, or clinically meaningful.

**2.** Define the **target population and prediction time**; split by patient; reserve an untouched test cohort or use nested grouped cross-validation. Tune each model within training/development data; fit all preprocessing inside the training folds; compare on *identical held-out patients*. Report AUROC, AUPRC, clinically relevant sensitivity/specificity, calibration, confidence intervals on **paired differences** (e.g., patient-level bootstrap), repeated seeds, subgroup performance, and ideally external validation.

**3.** Compare **wearables only**, **EHR only**, **simple early/late fusion**, and the **proposed transformer fusion**. Optionally test missing modalities, reduced labels, and alternative temporal encoders. Keep dataset definition, target, patient splits, reporting metrics, and tuning budget comparable. Do **not** insist that parameter counts and architectures be literally identical when input modalities change.

**4.** Compare clinical utility at relevant decision thresholds, false positive/negative costs, AUPRC if the event is rare, and calibration. Attempt **recalibration** of the transformer on separate validation data, then evaluate on untouched test data. Choose based on overall evidence, reliability, deployment constraints, and clinical context—not AUROC or calibration alone.

**Research framing:** The question is not simply *“Does the transformer have the highest number?”* but *“Is the measured gain robust, attributable to the method, and useful for the intended clinical decision?”*

</details>

**Takeaway:** Valid scientific comparisons require fair baselines, leakage-resistant design, uncertainty quantification, ablation studies, and appropriate clinical endpoints.

---

## Formula sheet

|Concept|Formula / principle|
|-|-|
|Gradient descent|$\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t)$|
|Sigmoid|$\sigma(z)=1/(1+e^{-z})$|
|Logistic log-odds|$\log(p/(1-p))=w^Tx+b$|
|Bernoulli likelihood|$P(y\mid x)=p^y(1-p)^{1-y}$|
|Binary cross-entropy|$-[y\log p+(1-y)\log(1-p)]$|
|Logistic BCE gradient (logit)|$\partial L/\partial z=p-y$|
|L1 objective|$L+\lambda\sum_j|w_j|$|
|L2 objective|$L+\lambda\sum_jw_j^2$|
|Accuracy|$(TP+TN)/(TP+TN+FP+FN)$|
|Precision|$TP/(TP+FP)$ (undefined if denominator = 0)|
|Recall / sensitivity|$TP/(TP+FN)$|
|Specificity|$TN/(TN+FP)$|
|F1|$2\cdot\mathrm{Precision}\cdot\mathrm{Recall}/(\mathrm{Precision}+\mathrm{Recall})$ when defined|
|Brier score|$N^{-1}\sum_i(\hat p_i-y_i)^2$|
|Mean CV metric|$K^{-1}\sum_{k=1}^K M_k$|
|Squared-error bias–variance|$\mathbb E\[(Y-\hat f(x))^2\]=\operatorname{Bias}^2+\operatorname{Variance}+\sigma^2$ (standard assumptions)|

## My recurring misconceptions

* [ ] **Bias–variance:** I initially said deep networks generally have *higher bias* and linear models *higher variance*. Usually the classical tendency is the reverse.
* [ ] **Overfitting:** I said model flexibility “drops” after more training. Its capacity doesn't necessarily change; generalization worsens as fitting continues.
* [ ] **Learning rate:** I suggested reducing it as a direct regularizer. Smaller steps alone do not guarantee less overfitting.
* [ ] **Logistic regression:** I cited nonlinear disease relationships as the main reason to prefer it. The core justification is **probabilistic outputs and Bernoulli modeling**.
* [ ] **BCE:** I said it is “more flexible.” Its important justification is **negative Bernoulli log-likelihood**.
* [ ] **L1 vs. L2:** I initially preferred L2 for a sparse-signal scenario. **L1 encourages exactly zero coefficients**.
* [ ] **CV interpretation:** I described CV as revealing “true performance.” It yields an **estimate with uncertainty**, and depends on proper grouping and model-selection procedures.
* [ ] **Clinical metrics:** Improving recall generally decreases false negatives but may **increase false positives**, rather than merely “focusing on false negatives.”
* [ ] **Calibration:** Both **overestimated and underestimated risks** can produce harmful clinical decisions.
* [ ] **Ablations:** Keep evaluation conditions comparable, but architectural parameters cannot always remain identical when removing modalities.

## Spaced-repetition checklist

Use this self-test several times. For each question, mark **R** (recall correctly without notes), **P** (partial), or **N** (not yet).

|Question|Review 1|Review 2|Review 3|Review 4|
|-|-|-|-|-|
|Q1 Generalization|||||
|Q2 Bias–variance|||||
|Q3 Overfitting|||||
|Q4 Logistic/BCE|||||
|Q5 L1/L2|||||
|Q6 Grouped CV|||||
|Q7 Imbalance|||||
|Q8 Calibration|||||
|Q9 Gradient descent|||||
|Q10 Research evaluation|||||

**Suggested cadence:** First review within 24 hours, second after 3 days, third after 7 days, fourth after 14 days. Give extra practice to **Q2, Q4, and Q5**. Each time, answer before expanding the solution.

### Helpful references

* [Stanford CS229 — Machine Learning](https://cs229.stanford.edu/)
* [Introduction to Statistical Learning (Python)](https://www.statlearning.com/)
* [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
* [scikit-learn: Common pitfalls and leakage](https://scikit-learn.org/stable/common_pitfalls.html)
* [Dive into Deep Learning](https://d2l.ai/)

---

*Prepared from my Day 1 practice conversation as a personal revision resource. Original answers intentionally retain their wording and uncertainties so I can compare my progress over time.*
