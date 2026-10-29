# Day 4 --- Representation Learning and Rigorous Biomedical ML

**Purpose:** Revision report with the ten questions, my answers,
corrected answers, weaknesses, and recommended resources.

**Overall estimated score: 6.2/10.** This is a teaching estimate, not a
standardized assessment.

## Executive summary

Day 4 covered representation learning, self-supervised learning,
contrastive learning, fine-tuning, representation collapse, linear
probes, multimodal fusion, distribution shift, and experimental design.

### Strengths

-   You understand that complex models are not automatically better.
-   You repeatedly recognized the importance of patient-level splits and
    external validation.
-   You proposed useful ablations and unimodal baselines.
-   You understand that discrimination, calibration, and clinical
    thresholds are different.
-   You question whether an apparent improvement is genuine or caused by
    shortcuts.

### Main weaknesses to revise

1.  Mathematical precision: cosine similarity, contrastive loss,
    temperature, embedding variance, and linear probing.
2.  Self-supervised learning: distinguish targets constructed from data
    from downstream clinical labels.
3.  Leakage across the whole pipeline, including unlabeled pretraining
    and temporal alignment.
4.  Missing features versus missing modalities versus time-misaligned
    data.
5.  Experimental design that isolates the contributions of pretraining,
    fusion, and model capacity.
6.  Statistical uncertainty and external evaluation without tuning on
    the final test set.

------------------------------------------------------------------------

# Question 1 --- Representation learning

## Question

1.  What is a representation and representation learning?
2.  Why can learned representations help?
3.  How do supervised and self-supervised learning differ?
4.  How would you evaluate representation quality?

## My answers

1.  A representation is a numerical matrix representing a feature or
    pattern. Representation learning converts raw data into useful
    patterns; a classifier uses provided features.
2.  Learned representations may discover fine-grained patterns missed by
    manual feature engineering, although more complexity is not always
    better.
3.  Supervised learning uses labels and features; self-supervised
    learning can use a small labeled set plus unlabeled data.
4.  I would use external validation, control data format and collection,
    and change patients to test generalization.

## Correct answers

An encoder maps input (x) to a representation (z=f\_`\theta`{=tex}(x)).
A representation is a numerical encoding; representation learning learns
the mapping from data. It may capture useful nonlinear and temporal
patterns, but it can also encode noise, artifacts, or hospital-specific
shortcuts.

Supervised learning uses labels for the target task. Self-supervised
learning creates training targets from the data itself, such as
predicting masked heart-rate values or forecasting future signals. It is
not defined by combining a few labeled examples with many unlabeled
examples.

Evaluate with a frozen-encoder linear probe, downstream fine-tuning,
simple baselines, patient-level splits, external validation, and checks
for shortcut learning and collapse.

**Key lesson:** Good reconstruction does not guarantee good clinical
prediction.

------------------------------------------------------------------------

# Question 2 --- Self-supervised learning

## Question

1.  Identify a valid self-supervised task.
2.  Define the input, target, and loss for masked heart-rate
    reconstruction.
3.  Can reconstruction help while clinical prediction remains poor?
4.  How should two encoders be compared with 500 labeled patients?

## My answers

1.  Task B, because it needs unlabeled data.
2.  Input: masked and unmasked heart rate; target: masked values; loss:
    squared distance between predictions and true values.
3.  Yes, because reconstruction and complication prediction are
    different tasks.
4.  Split 500 patients into an 80% validation set and 20% test set,
    fine-tune, compare, and use patient-level splitting.

## Correct answers

A self-supervised task constructs targets from the input data. For
masked reconstruction, let (M) be the masked time points:

\[
`\mathcal `{=tex}L\_{`\mathrm{MSE}`{=tex}}=`\frac{1}{|M|}`{=tex}`\sum`{=tex}\_{t`\in `{=tex}M}(x_t-`\hat{x}`{=tex}\_t)\^2.
\]

The loss should be calculated on the intended masked targets. A model
may reconstruct signals well but learn features that do not predict
complications.

For a fair comparison, split patients into training, validation, and
test sets (or use grouped/nested cross-validation if appropriate). Train
on the training patients, tune on validation patients, and evaluate on
untouched test patients. Use identical partitions for both encoders and
include a comparable model trained from scratch.

If unlabeled data overlap with held-out labeled patients, exclude those
patients from pretraining when claiming strict unseen-patient
generalization. A target-adaptation setting can be studied separately if
it reflects the intended deployment.

**Key lesson:** Define the pretraining target, loss, downstream task,
and evaluation split separately.

------------------------------------------------------------------------

# Question 3 --- Contrastive learning

## Question

1.  Explain positive and negative pairs and suitable time-series
    augmentations.
2.  What does contrastive learning optimize?
3.  What is a false negative?
4.  How would you evaluate the encoder for complication prediction?

## My answers

1.  Positive pairs could be similar patient data; negatives are very
    different data. Augmentations include jittering, cropping, scaling,
    time warping, and masking. Cropping may remove useful information.
2.  Positive pairs should have closer embeddings and negatives farther
    embeddings.
3.  A false negative might result from bad augmentation or data
    selection; I described it as a positive pair predicted as negative.
4.  Compare downstream AUROC/AUPRC with patient-level
    train/validation/test splits. I asked whether labels should be
    binary.

## Correct answers

A positive pair consists of views intended to preserve relevant
semantics, often two suitable augmentations of one signal. Two different
patients are not necessarily clinically dissimilar. Augmentations must
preserve the physiological information relevant to the task; cropping
could remove an arrhythmia episode.

A common contrastive objective is InfoNCE:

\[ `\mathcal `{=tex}L_i=-`\log`{=tex}
`\frac{\exp(s(z_i,z_i^+)/\tau)}`{=tex}
{`\sum`{=tex}\_j`\exp`{=tex}(s(z_i,z_j)/`\tau`{=tex})}. \]

It encourages the positive to receive higher relative similarity than
negative examples.

A **false negative** is a pair treated as negative despite sharing
relevant semantics. This differs from an augmentation that destroys
information in a supposed positive pair.

Complication labels can be binary if the endpoint is defined as yes/no
within a fixed prediction horizon; other clinical outcomes can be
continuous, multiclass, or time-to-event. Use patient-level evaluation
and compare with a from-scratch baseline.

**Key lesson:** Pair construction encodes assumptions about which
similarities matter.

------------------------------------------------------------------------

# Question 4 --- Contrastive loss and embedding geometry

## Question

1.  Why is a negative pair with cosine similarity 0.75 difficult?
2.  How should contrastive loss change as positives become more similar
    and negatives less similar?
3.  What does temperature control?
4.  How would you investigate patient-identity shortcuts?

## My answers

1.  A negative pair with similarity 0.75 is challenging because its
    similarity is high. I described similarity as distance.
2.  Loss decreases when the model becomes closer to the correct label.
3.  Temperature controls loss scale and speed. I thought high
    temperature might make loss approach zero and low temperature might
    stop it decreasing quickly.
4.  Excellent training but poor downstream results might indicate
    identity leakage. I suggested checking age, sex, BMI, disease
    history, and doing ablations.

## Correct answers

Cosine similarity measures the angle between vectors:

\[ `\operatorname{sim}`{=tex}(z_i,z_j)=
`\frac{z_i^\top z_j}{\|z_i\|\|z_j\|}`{=tex}. \]

For unit-normalized embeddings:

\[ \|z_i-z_j\|\_2\^2=2-2`\operatorname{sim}`{=tex}(z_i,z_j). \]

Thus, higher cosine similarity corresponds to smaller Euclidean distance
only under this normalization.

InfoNCE uses a softmax over similarity scores divided by temperature
(`\tau`{=tex}). Lower temperature makes the distribution sharper and
often emphasizes hard negatives; higher temperature makes it softer.
Temperature does not directly set the speed of loss decrease or
guarantee a particular final loss.

Distinguish identity information, shortcut learning, and leakage. Test
patient-level splits, train probes for hospital/device identity, inspect
downstream prediction, ablate suspected artifacts, and evaluate on
external patients. Predictable patient identity is not automatically
harmful; the question is whether it undermines the intended clinical
use.

**Key lesson:** Learn the cosine formula, normalized-distance
relationship, and temperature intuition.

------------------------------------------------------------------------

# Question 5 --- Transfer learning and fine-tuning

## Question

1.  Compare frozen, full, and partial fine-tuning.
2.  Which would you try first with 500 labels?
3.  Explain overfitting and catastrophic forgetting.
4.  Design a fair comparison.

## My answers

1.  Frozen trains only the classifier; full fine-tuning updates all
    layers; partial fine-tuning is between them. I initially compared
    freezing with validation.
2.  I would start with partial fine-tuning because the labeled set is
    small. Full fine-tuning may help when more data are available or the
    pretraining objective differs.
3.  Fine-tuning may overwrite pretrained information. I would monitor
    train/validation curves and use a reasonable learning rate and
    number of layers.
4.  Split patients into train/validation/test and compare recall, F1,
    and AUROC.

## Correct answers

If the encoder is (f\_`\theta`{=tex}) and the classifier is
(g\_`\phi`{=tex}), then
(`\hat `{=tex}p=g\_`\phi`{=tex}(f\_`\theta`{=tex}(x))).

-   Frozen: (`\theta`{=tex}) is fixed; optimize (`\phi`{=tex}).
-   Full fine-tuning: optimize both (`\theta`{=tex}) and (`\phi`{=tex}).
-   Partial fine-tuning: optimize (`\phi`{=tex}) and selected layers of
    (`\theta`{=tex}).

A frozen encoder is a training strategy, not a validation procedure.
With 500 labels, begin with a frozen baseline and compare partial and
full fine-tuning empirically.

**Overfitting** is poor generalization despite good training
performance. **Catastrophic forgetting** is substantial erosion of
capabilities or information learned during pretraining. Mitigations
include smaller learning rates for pretrained layers, freezing early
layers, regularization, and early stopping.

Use identical patient partitions and comparable tuning budgets. Include
a from-scratch baseline and report AUROC, AUPRC where appropriate,
calibration, clinically relevant threshold metrics, and uncertainty.

**Key lesson:** Do not assume full fine-tuning is best simply because it
is more adaptable.

------------------------------------------------------------------------

# Question 6 --- Representation collapse and shortcut learning

## Question

1.  Define collapse mathematically.
2.  How can it be detected?
3.  How can it be prevented?
4.  How would you investigate hospital-specific shortcuts?

## My answers

1.  Collapse might be caused by a poor activation function or
    standardization. It harms prediction because useful information was
    not learned.
2.  Check means, variances, and distances. Low variance means embeddings
    are close and might cause majority-class predictions.
3.  Use appropriate activations and scale inputs.
4.  Strong hospital clustering may indicate shortcuts. I would not
    remove all hospital information without investigating the cause.

## Correct answers

Collapse occurs when many different inputs map to nearly identical
embeddings:

\[ f\_`\theta`{=tex}(x_i)`\approx `{=tex}c. \]

For embedding dimension (j), inspect variance across patients:

\[
v_j=`\frac{1}{n}`{=tex}`\sum`{=tex}\_{i=1}^n(z\_{ij}-`\bar `{=tex}z_j)^2.
\]

Near-zero variance across most dimensions is a warning sign. Also
inspect pairwise distances and downstream utility. Low variance is not
the only possible failure mode, and nonzero variance does not prove
clinical usefulness.

More targeted anti-collapse methods include contrastive learning with
suitable pairs, variance regularization (VICReg), and
stop-gradient/target-network approaches (BYOL). Input scaling and
activation choice can help optimization but do not guarantee prevention.

To investigate hospital shortcuts, train probes to predict
hospital/device identity, check external clinical performance, inspect
missingness and device artifacts, and ablate suspicious inputs. High
hospital predictability alone does not prove harmful shortcut learning.

**Key lesson:** Learn at least one explicit anti-collapse mechanism and
how to diagnose collapse.

------------------------------------------------------------------------

# Question 7 --- Evaluating learned representations

## Question

1.  What is a linear probe, and why freeze the encoder?
2.  What if Encoder B wins a linear probe but Encoder A wins after
    fine-tuning?
3.  How would you test transfer and shortcuts?
4.  How would you compare encoders fairly with 500 labels?

## My answers

1.  A linear probe is a linear classifier that tests whether useful
    information was learned; freeze the encoder and compare AUROC/AUPRC.
2.  B may use a shortcut while A may learn more useful information. I
    would inspect reliability and the downstream objective.
3.  Test different physiological data and random seeds; a sharp
    performance drop could indicate shortcuts.
4.  Use patient-level splits, tune seeds/learning rate/regularization,
    report AUROC/AUPRC, calibration and discrimination, and inspect
    probabilities and statistical tests.

## Correct answers

With frozen representation (z=f\_`\theta`{=tex}(x)), a linear probe
predicts:

\[
`\hat `{=tex}p(y=1`\mid `{=tex}x)=`\sigma`{=tex}(w\^`\top `{=tex}z+b).
\]

Only (w,b) are trained. It tests whether target information is
accessible to a linear classifier, not whether the representation
contains every possible useful signal.

A linear probe measures accessibility; fine-tuning measures
adaptability. A reversed ranking does not by itself prove shortcut
learning.

Use hospital/device probes, subgroup analysis, external evaluation, and
targeted ablations. Different random seeds test sensitivity to
stochastic training, not external generalization by themselves.

Use the same patient splits, preprocessing, downstream task, and
comparable tuning budgets. Tune on validation data only and estimate
paired differences on the same held-out patients, for example with
patient-level bootstrap confidence intervals.

**Key lesson:** A linear probe and full fine-tuning answer different
questions.

------------------------------------------------------------------------

# Question 8 --- Multimodal representation learning

## Question

1.  Compare early, intermediate, and late fusion.
2.  Handle missing modalities.
3.  Demonstrate that multimodal learning adds value.
4.  Prevent leakage and evaluate robustness.

## My answers

1.  Early fusion concatenates features; intermediate fusion encodes
    modalities and concatenates representations; late fusion combines
    modality-specific predictions.
2.  Remove patients with too much missing data, impute remaining values,
    perhaps with logistic regression. Zero-filling can misalign data.
3.  Compare multimodal models and baselines; remove one modality at a
    time while keeping patient splits fixed.
4.  Make patient vectors the same length, prevent partition leakage, and
    test on external data with different missingness, hospital, and
    collection practices.

## Correct answers

-   **Early fusion:** combine input features before the main model;
    simple but sensitive to scaling and alignment.
-   **Intermediate fusion:** encode modalities separately and combine
    representations; supports modality-specific and cross-modal patterns
    but is more complex.
-   **Late fusion:** combine modality-specific predictions; modular but
    may lose fine-grained interactions.

Distinguish missing individual features, missing entire modalities, and
time-misaligned observations. Excluding patients with missing data may
create selection bias if missingness relates to illness. Fit imputation
on training data only; consider missingness indicators,
modality-availability masks, modality dropout, and realistic
missing-modality evaluation. Zero may be a valid value, so zero-filling
without an indicator can confuse absence with observation.

To establish value, compare unimodal models, multimodal models,
leave-one-modality-out ablations, simple baselines, and pretrained
versus from-scratch models. Keep patient splits, outcome definitions,
prediction time, and tuning budgets comparable.

Matching vector dimensions is not enough: align modalities by patient,
timestamps, prediction window, and availability at prediction time. Do
not use notes written after the outcome to predict that outcome.

**Key lesson:** Representation alignment and clinical/temporal alignment
are different.

------------------------------------------------------------------------

# Question 9 --- Distribution shift and robust representations

## Question

1.  Explain covariate, label, and concept shift.
2.  Diagnose an external performance drop.
3.  Choose among recalibration, threshold changes, fine-tuning, domain
    adaptation, and robust training.
4.  Design a rigorous external evaluation.

## My answers

1.  Older population and sampling frequency are covariate shift; higher
    prevalence is label shift; different laboratory recording is concept
    shift.
2.  Check devices, demographics, missingness, prevalence, and outcome
    definitions, then align data.
3.  Recalibrate if probabilities disagree with observations; change the
    threshold to alter predicted positives; fine-tune with Hospital B
    data; use domain adaptation and robust training.
4.  Compare models on held-out data with AUROC/AUPRC, subgroup metrics,
    and paired patient-level bootstrap confidence intervals.

## Correct answers

Let (X) be features and (Y) the outcome:

-   Covariate shift: (P_A(X)`\neq `{=tex}P_B(X)), while
    (P(Y`\mid `{=tex}X)) is assumed stable.
-   Label shift: (P_A(Y)`\neq `{=tex}P_B(Y)), while (P(X`\mid `{=tex}Y))
    is assumed stable.
-   Concept shift:
    (P_A(Y`\mid `{=tex}X)`\neq `{=tex}P_B(Y`\mid `{=tex}X)).

Different laboratory units or recording procedures are
measurement/feature shifts and do not automatically establish concept
shift. Changed outcome definitions may create outcome-measurement
problems.

Diagnose data harmonization, feature distributions, missingness,
prevalence, label definitions, calibration, discrimination, and subgroup
performance before choosing a remedy.

Recalibration adjusts predicted probabilities, not necessarily ranking.
Threshold changes alter the sensitivity/precision trade-off, not AUROC.
Fine-tuning and domain adaptation need target data and can cause
negative transfer. Robust training does not guarantee immunity to unseen
shifts.

For external evaluation, keep Hospital B untouched unless a separate
adaptation cohort is available. If adapting, use separate adaptation and
final test cohorts. Compare models on the same final test patients and
estimate paired differences. A confidence interval containing zero does
not prove equivalence.

**Key lesson:** Diagnose the shift first; then select the method that
addresses that specific problem.

------------------------------------------------------------------------

# Question 10 --- Design a rigorous representation-learning study

## Question

Given 20,000 unlabeled wearable patients, 500 labeled patients with
wearable/EHR/notes data, and a separate Hospital B cohort:

1.  State a hypothesis and design leakage-free splits.
2.  Choose pretraining, fusion, fine-tuning, and missing-modality
    methods.
3.  Define baselines, ablations, and metrics.
4.  Quantify uncertainty and evaluate external generalization.

## My answers

1.  Hypothesis: self-supervised pretraining and multimodal fusion can
    improve prediction. Use 20,000 unlabeled patients for pretraining
    and 500 labeled patients for fine-tuning; split patients before
    preprocessing.
2.  Predict the next 30 days of heart rate, use intermediate fusion, and
    full fine-tuning. I repeated the same pretraining/fine-tuning split
    plan.
3.  Baselines: wearable-only, EHR-only, notes-only. Ablation: remove one
    modality at a time. Metrics: AUROC, AUPRC, sensitivity, specificity,
    and confidence intervals.
4.  Use paired patient-level bootstrap, compare the Hospital A model
    with the Hospital B fine-tuned model on a final test set, report
    subgroups and inspect false positives/negatives. I would consider
    the pipeline useful if AUROC, AUPRC, discrimination, and calibration
    were consistently good.

## Correct research-level answer

### 1. Testable hypotheses and splits

Separate the claims: - H1: Pretraining improves prediction compared with
the same architecture trained from scratch. - H2: Multimodal fusion
improves prediction compared with the strongest unimodal baseline.

Split labeled patients before window generation or preprocessing. Check
overlap between labeled and unlabeled cohorts. Exclude held-out
validation/test patients from pretraining when claiming strict
unseen-patient generalization. Define prediction time and outcome
horizon, and use only information available at prediction time.

### 2. Model

Pretrain a wearable encoder using masked reconstruction or forecasting.
Encode wearable, EHR, and notes separately, then use intermediate
fusion. Compare frozen, partial, and full fine-tuning. Handle missing
modalities with availability masks, modality dropout, and realistic
missingness tests. With 500 labeled patients, do not assume full
fine-tuning is best.

### 3. Baselines and metrics

Include unimodal models, multimodal models, from-scratch versus
pretrained variants, leave-one-modality-out ablations, and simple
conventional baselines. Report AUROC, AUPRC where appropriate,
sensitivity, specificity, precision, calibration, Brier score, and
uncertainty. A factorial comparison of pretraining (yes/no) and fusion
(unimodal/multimodal) helps separate their contributions.

### 4. External evaluation

Freeze the Hospital A model before evaluating on untouched Hospital B.
If adapting on Hospital B, create separate adaptation and final test
cohorts. Use paired patient-level bootstrap confidence intervals on the
same test patients. Evaluate prespecified subgroups, device effects,
missingness, and false positives/negatives. Define clinically meaningful
acceptance criteria before evaluation.

**Key lesson:** The strongest study isolates pretraining and fusion
effects, prevents leakage through unlabeled data, and protects the final
test set.

------------------------------------------------------------------------

# Core equations to memorize

## Representation

\[ z=f\_`\theta`{=tex}(x) \]

## Masked reconstruction

\[
`\mathcal `{=tex}L\_{`\mathrm{MSE}`{=tex}}=`\frac{1}{|M|}`{=tex}`\sum`{=tex}\_{t`\in `{=tex}M}(x_t-`\hat{x}`{=tex}\_t)\^2
\]

## InfoNCE contrastive loss

\[ `\mathcal `{=tex}L_i=-`\log`{=tex}
`\frac{\exp(s(z_i,z_i^+)/\tau)}`{=tex}
{`\sum`{=tex}\_j`\exp`{=tex}(s(z_i,z_j)/`\tau`{=tex})} \]

## Cosine similarity

\[ `\operatorname{sim}`{=tex}(z_i,z_j)=
`\frac{z_i^\top z_j}{\|z_i\|\|z_j\|}`{=tex} \]

For unit-normalized vectors: \[
\|z_i-z_j\|\_2\^2=2-2`\operatorname{sim}`{=tex}(z_i,z_j) \]

## Paired performance difference

\[ `\Delta`{=tex}=M_B-M_A \]

Estimate uncertainty in (`\Delta`{=tex}) by resampling the same held-out
patients for both models.

------------------------------------------------------------------------

# Targeted revision plan

Estimated time: four focused hours.

  ------------------------------------------------------------------------------
  Session                           Time Focus             Practice
  ---------------- --------------------- ----------------- ---------------------
  1                               60 min Self-supervised   Explain masked
                                         and contrastive   reconstruction,
                                         learning          forecasting, InfoNCE,
                                                           temperature, and
                                                           false negatives.

  2                               60 min Representation    Explain linear
                                         quality and       probing, collapse
                                         transfer learning diagnostics, and
                                                           frozen/partial/full
                                                           fine-tuning.

  3                               60 min Multimodal        Draw the data
                                         learning and      pipeline and mark
                                         leakage           patient IDs,
                                                           timestamps, masks,
                                                           and preprocessing.

  4                               60 min Experimental      Write a protocol
                                         design and        isolating pretraining
                                         uncertainty       and fusion, with a
                                                           protected external
                                                           test set.
  ------------------------------------------------------------------------------

## Checklist

-   [ ] I can distinguish self-supervised pretraining from downstream
    supervised learning.
-   [ ] I can explain contrastive loss, temperature, and false
    negatives.
-   [ ] I can explain linear probing and representation collapse.
-   [ ] I can distinguish missing features, missing modalities, and
    temporal misalignment.
-   [ ] I can prevent patient leakage through unlabeled pretraining.
-   [ ] I can design fair baselines and ablations.
-   [ ] I can estimate paired confidence intervals without using test
    data for tuning.
-   [ ] I can distinguish covariate, label, and concept shift.

------------------------------------------------------------------------

# Recommended learning sources

## Papers

1.  **SimCLR --- A Simple Framework for Contrastive Learning of Visual
    Representations**\
    https://arxiv.org/abs/2002.05709\
    Focus: augmentations, positive/negative pairs, contrastive loss,
    temperature.

2.  **TS2Vec --- Towards Universal Representation of Time Series**\
    https://arxiv.org/abs/2106.10466\
    Focus: time-series representations at different temporal scales and
    downstream evaluation.

3.  **VICReg --- Variance-Invariance-Covariance Regularization for
    Self-Supervised Learning**\
    https://arxiv.org/abs/2105.04906\
    Focus: representation collapse and variance regularization.

4.  **BYOL --- Bootstrap Your Own Latent**\
    https://arxiv.org/abs/2006.07733\
    Focus: self-supervised learning without explicit negative pairs.

5.  **Multimodal Machine Learning: A Survey and Taxonomy**\
    https://arxiv.org/abs/1705.09406\
    Focus: multimodal representation, alignment, fusion, and challenges.

6.  **From development to deployment: dataset shift, causality, and
    shift-stable models in health AI**\
    https://doi.org/10.1093/biostatistics/kxz041\
    Focus: distribution shift in health AI.

## Courses and implementation

-   **PyTorch Tutorials:** https://pytorch.org/tutorials/ --- implement
    an encoder, reconstruction loss, frozen parameters, and a linear
    probe.
-   **MIT Introduction to Deep Learning (6.S191):**
    https://introtodeeplearning.com/ --- review representations,
    optimization, and attention.
-   **Understanding Deep Learning, Simon J. D. Prince:**
    https://udlbook.github.io/udlbook/ --- mathematical foundations of
    neural networks, representations, and attention.

## Suggested order

1.  Read the relevant SimCLR sections.
2.  Read TS2Vec's method and evaluation sections.
3.  Read VICReg to understand collapse prevention.
4.  Implement a small PyTorch experiment with frozen and fine-tuned
    encoders.
5.  Read the multimodal survey and dataset-shift paper.
6.  Write a one-page protocol for the wearable/EHR/notes study.

------------------------------------------------------------------------

# Final assessment

**Overall estimated Day 4 score: 6.2/10.**

Your strongest skill is research-minded skepticism: you ask whether
results generalize, whether an ablation is fair, and whether a model is
using the right information.

Your biggest opportunity is methodological precision. For every
technique, be ready to explain: 1. What problem it addresses. 2. What
assumptions it makes. 3. How it is implemented without leakage. 4. What
experiment tests whether it helped. 5. What conclusion the evidence
actually supports.

You can proceed to Day 5 while revisiting this checklist. The goal is
not to memorize many methods; it is to design and defend a rigorous
experiment from start to finish.
