# Day 3 Study Report: Probability, Statistics, and Uncertainty in Medical ML

> **Purpose:** A GitHub-ready revision guide covering all 10 questions,
> my answers, corrected answers, mathematical intuition, weaknesses, and
> a revision plan.

## Executive summary

### What I did well

-   I usually calculated expectation, variance, standard error,
    confidence intervals, and z-statistics correctly.
-   I understood why prevalence matters for positive predictive value
    and why internal performance does not guarantee external
    generalization.
-   I recognized the importance of patient-level evaluation when
    patients contribute multiple wearable windows.
-   I made a sensible recommendation not to deploy a model immediately
    when external discrimination and calibration were worse.

### Main weaknesses

1.  Interpreting p-values and confidence intervals precisely.
2.  Comparing models using uncertainty in the *difference* between their
    scores.
3.  Distinguishing bootstrap uncertainty in fixed predictions from
    training and model-selection uncertainty.
4.  Understanding Bayes' theorem, priors, likelihoods, and posteriors.
5.  Designing external validation without compromising the test cohort.

------------------------------------------------------------------------

# Question 1 --- Bayes' theorem and base rates

## Question

A disease has 1% prevalence. A test has 90% sensitivity and 90%
specificity. In 10,000 people, calculate TP, FN, TN, FP, and PPV.
Explain why PPV differs from sensitivity.

## My answer

-   TP = 90, FN = 10, TN = 8,910, FP = 990.
-   PPV = 90 / 1,080 = 8.33%.
-   Low prevalence means there are many more people without disease than
    with disease.
-   I wrote an incorrect Bayes formula, but correctly distinguished
    `P(Disease | Positive)` from `P(Positive | Disease)`.

## Evaluation

The confusion matrix, PPV, and base-rate intuition were correct. The
formula needed correction.

## Correct answer

Of 10,000 people, 100 have the disease and 9,900 do not.

-   TP = 0.90 × 100 = 90
-   FN = 10
-   TN = 0.90 × 9,900 = 8,910
-   FP = 990

\[
PPV=P(D`\mid `{=tex}+)=`\frac{TP}{TP+FP}`{=tex}=`\frac{90}{1080}`{=tex}=0.0833.
\]

Bayes' theorem in sensitivity/specificity form is:

\[ PPV=`\frac{Se\cdot Prev}{Se\cdot Prev+(1-Sp)(1-Prev)}`{=tex}. \]

Here:

\[ PPV=`\frac{0.90(0.01)}{0.90(0.01)+0.10(0.99)}`{=tex}
=`\frac{0.009}{0.108}`{=tex}=8.33%. \]

**Why is PPV low?** Even with 90% specificity, the much larger
non-diseased population produces 990 false positives, compared with 90
true positives.

-   `P(Positive | Disease)` is sensitivity.
-   `P(Disease | Positive)` is PPV.

These conditional probabilities are not interchangeable.

**Remember:** Prevalence affects PPV and NPV. Sensitivity and
specificity are conditional on disease status.

------------------------------------------------------------------------

# Question 2 --- Probability distributions, expectation, and variance

## Question

Check whether a discrete probability distribution is valid, calculate
expectation and variance, and interpret the expected count of abnormal
wearable episodes.

## My answer

1.  Valid because each probability is between 0 and 1 and they sum to 1.
2.  `E[X] = 0.7`.
3.  `E[X²] = 1.1`, so `Var(X) = 1.1 − 0.49 = 0.61`.
4.  Zero is the most likely outcome, but nonzero episodes are possible,
    so the expectation is not zero.

## Evaluation

This was one of my strongest answers. The calculations and main
intuition were correct. The distribution implied by the moments is
`P(X=0)=0.5`, `P(X=1)=0.3`, `P(X=2)=0.2`.

## Correct answer

A valid discrete distribution satisfies:

\[ 0`\le `{=tex}P(X=x)`\le1`{=tex},`\qquad `{=tex}`\sum`{=tex}\_x
P(X=x)=1. \]

\[ E\[X\]=`\sum`{=tex}\_x xP(X=x)=0(0.5)+1(0.3)+2(0.2)=0.7. \]

\[ E\[X\^2\]=0^2(0.5)+1^2(0.3)+2\^2(0.2)=1.1. \]

\[ Var(X)=E\[X\^2\]-(E\[X\])^2=1.1-0.7^2=0.61. \]

The standard deviation is (`\sqrt{0.61}`{=tex}`\approx0.781`{=tex}).

**Interpretation:** 0.7 is the long-run average number of episodes per
observation period. It does not mean an individual patient experiences
0.7 episodes, nor is expectation necessarily the most likely outcome.

**Remember:** `E[X]` is a mean; `Var(X)` measures spread around that
mean.

------------------------------------------------------------------------

# Question 3 --- Sampling distributions and the Central Limit Theorem

## Question

Population heart rate has mean 72 bpm and standard deviation 12 bpm.
Calculate the expected sample mean and standard error for `n=64` and
`n=256`, then explain the CLT.

## My answer

-   `E[X̄] = 72`.
-   For `n=64`, `SE=1.5`.
-   For `n=256`, `SE=0.75`.
-   I initially confused the CLT with the range of individual heart
    rates, then recognized that individual measurements need not be
    normal.

## Evaluation

All calculations were correct. The main misconception was about what the
CLT describes.

## Correct answer

For independent observations:

\[ E\[`\bar `{=tex}X\]=`\mu=72`{=tex}. \]

\[ SE(`\bar `{=tex}X)=`\frac{\sigma}{\sqrt n}`{=tex}. \]

For 64 patients:

\[ SE=`\frac{12}{\sqrt{64}}`{=tex}=1.5`\text{ bpm}`{=tex}. \]

For 256 patients:

\[ SE=`\frac{12}{\sqrt{256}}`{=tex}=0.75`\text{ bpm}`{=tex}. \]

Four times the sample size halves the standard error because
`SE ∝ 1/√n`.

**Central Limit Theorem:** Under appropriate conditions, the sampling
distribution of the sample mean approaches a normal distribution as
sample size grows:

\[
`\bar `{=tex}X`\approx`{=tex}`\mathcal `{=tex}N`\left`{=tex}(`\mu`{=tex},`\frac{\sigma^2}{n}`{=tex}`\right`{=tex}).
\]

The CLT concerns **sample means**, not individual observations. It does
not say heart rates must lie between 60 and 84 bpm; those values are one
standard deviation below and above the mean.

**Remember:** Standard deviation describes variation among observations;
standard error describes variation in an estimator across repeated
samples.

------------------------------------------------------------------------

# Question 4 --- Confidence intervals

## Question

A sample of 64 patients has mean heart rate 75 bpm. The known population
standard deviation is 12 bpm. Use `z=1.96` to calculate a 95% CI, then
repeat for `n=256` assuming the same sample mean and standard deviation.

## My answer

-   For `n=64`: `SE=1.5`, `MOE=2.94`, CI `[72.06, 77.94]`.
-   I described the CI as a range calculated from sample data to
    estimate the population value.
-   For `n=256`: `SE=0.75`, `MOE=1.47`, CI `[73.53, 76.47]`.
-   I correctly said the interval estimates the population mean, not
    individual values.

## Evaluation

The calculations were correct. The formal repeated-sampling
interpretation needed refinement.

## Correct answer

For `n=64`:

\[ SE=`\frac{12}{\sqrt{64}}`{=tex}=1.5,
`\qquad `{=tex}MOE=1.96(1.5)=2.94. \]

\[ CI=75`\pm2.94`{=tex}=\[72.06,77.94\]. \]

For `n=256`:

\[ SE=`\frac{12}{16}`{=tex}=0.75,`\qquad `{=tex}MOE=1.96(0.75)=1.47. \]

\[ CI=\[75-1.47,75+1.47\]=\[73.53,76.47\]. \]

**Frequentist interpretation:** A 95% confidence interval comes from a
procedure that, under its assumptions, contains the fixed population
parameter in 95% of repeated samples in the long run. It is not formally
a 95% probability statement about the parameter after one particular
interval has been observed.

A larger sample generally narrows the interval, but does not guarantee a
representative sample or remove bias.

**Remember:** A confidence interval estimates a population parameter. A
prediction interval concerns a future individual observation and is
generally wider.

------------------------------------------------------------------------

# Question 5 --- Hypothesis testing and p-values

## Question

A reference population has mean heart rate 72 bpm. A sample of 64
patients has mean 75 bpm, with known population SD 12 bpm. Conduct a
two-sided z-test at `α=0.05`. The p-value is approximately 0.0455.

## My answer

-   I was unsure how to formulate the hypotheses.
-   `z=2`.
-   I incorrectly said `p<0.05` gave more than 95% confidence not to
    reject the null and referred to correlation.
-   I incorrectly described the p-value as the probability that we
    cannot reject the null.

## Evaluation

The z-statistic was correct. Hypothesis formulation and p-value
interpretation were major weaknesses.

## Correct answer

\[ H_0:`\mu=72`{=tex},`\qquad `{=tex}H_1:`\mu`{=tex}`\ne72`{=tex}. \]

\[ z=`\frac{\bar x-\mu_0}{\sigma/\sqrt n}`{=tex}
=`\frac{75-72}{12/\sqrt{64}}`{=tex}=`\frac{3}{1.5}`{=tex}=2. \]

For a two-sided test:

\[ p=P(\|Z\|`\ge2`{=tex}`\mid `{=tex}H_0)`\approx0.0455`{=tex}. \]

Because `0.0455 < 0.05`, reject `H₀` at the 5% significance level. The
data provide statistically significant evidence that the population mean
differs from 72 bpm. The observed mean is higher.

**A p-value is not:** - The probability that `H₀` is true. - The
probability that the result happened "by chance." - The probability that
the alternative is true. - The probability that we cannot reject `H₀`.

A p-value measures how unusual the observed statistic, or a more extreme
one, would be under the null hypothesis and the test assumptions.
Statistical significance does not automatically imply clinical
importance or causality.

**Remember:** State the hypotheses, calculate the statistic, interpret
the p-value under the null, and report effect size and uncertainty.

------------------------------------------------------------------------

# Question 6 --- Statistical power and sample size

## Question

A study tests an intervention effect of 3 bpm. Assume null effect 0,
population SD 12 bpm, `n=64`, two-sided `α=0.05`, and standard error 1.5
bpm. Explain Type I error, Type II error, power, and the effect of
sample size.

## My answer

-   Type I error: rejecting a true null; probability 5%.
-   Type II error: failing to detect a true effect.
-   Power is the probability of detecting a real effect. I thought power
    was low because the statistic was above the rejection threshold.
-   Larger sample size reduces SE and increases power.

## Evaluation

The error definitions were correct. The power reasoning needed
refinement: the expected statistic is only slightly above the threshold,
so sampling variability still causes many nonsignificant results.

## Correct answer

\[
`\alpha`{=tex}=P(`\text{reject }`{=tex}H_0`\mid `{=tex}H_0`\text{ true}`{=tex})=0.05.
\]

\[
`\beta`{=tex}=P(`\text{fail to reject }`{=tex}H_0`\mid `{=tex}H_1`\text{ true}`{=tex}),
`\qquad `{=tex}Power=1-`\beta`{=tex}. \]

The standardized effect is:

\[ `\frac{\delta}{SE}`{=tex}=`\frac{3}{1.5}`{=tex}=2. \]

The rejection threshold for a two-sided 5% z-test is approximately
`|z| > 1.96`. Under the alternative, the statistic is approximately
normal with mean 2 and SD 1, giving power of about **52%**.

The expected statistic is only just beyond the threshold; many repeated
samples will fall short.

An approximate sample size for 80% power is:

\[
n`\approx`{=tex}`\left`{=tex}(`\frac{(1.96+0.84)\sigma}{\delta}`{=tex}`\right`{=tex})\^2
=`\left`{=tex}(`\frac{2.80(12)}{3}`{=tex}`\right`{=tex})\^2`\approx126`{=tex}.
\]

This is an approximate calculation under simplifying assumptions.

**Remember:** Power depends on effect size, sample size, variability,
significance level, and design. A nonsignificant result does not prove
there is no effect.

------------------------------------------------------------------------

# Question 7 --- Bootstrap methods and uncertainty estimation

## Question

A model has AUROC 0.84 on a test cohort of 200 patients. Each patient
contributes multiple wearable windows, but the outcome is patient-level.
Another model has AUROC 0.81 on the same patients. Explain bootstrap
confidence intervals and paired comparison.

## My answer

-   I correctly chose patient-level resampling.
-   I proposed splitting into development/test sets, bootstrapping and
    retraining, then calculating test AUROC.
-   I said to use the same patients but suggested averaging predictions
    and probabilities.
-   I recognized that bootstrap does not capture all uncertainty and
    suggested checking distribution, definitions, and missingness.

## Evaluation

The sampling unit was correct. I mixed up test-sample uncertainty from
fixed predictions with uncertainty from retraining the model.

## Correct answer: fixed-prediction bootstrap

1.  Fit and select the model using development data.
2.  Obtain predictions on the independent test cohort.
3.  Sample 200 patient indices with replacement.
4.  Calculate AUROC from the sampled patients' labels and fixed
    predictions.
5.  Repeat, for example, 2,000 times.
6.  Use the 2.5th and 97.5th percentiles of the bootstrap AUROCs as a
    percentile-based 95% CI.

\[ CI\_{95%}=\[Q\_{0.025}(AUC\^*),Q\_{0.975}(AUC\^*)\]. \]

Patient-level resampling preserves the independent sampling unit.
Resampling correlated wearable windows independently can underestimate
uncertainty.

## Paired comparison

For every bootstrap replicate, use the same patient indices for both
models:

\[ `\Delta`{=tex}\_b^*=AUC\_{A,b}\^*-AUC\_{B,b}^\*. \]

Construct a CI for the paired difference. Do not average predicted
probabilities as the comparison statistic; the target here is the
difference in AUROC.

Bootstrapping fixed predictions estimates test-sample uncertainty
conditional on the fitted models. It does not fully capture training
randomness, hyperparameter-selection uncertainty, or generalization to a
new hospital. Those require repeated training/selection and external
validation.

**Remember:** Match the resampling unit to the independent sampling
unit, and distinguish test-sample uncertainty from uncertainty in the
complete learning pipeline.

------------------------------------------------------------------------

# Question 8 --- Paired comparisons and multiple testing

## Question

Four models evaluated on the same patients have AUROCs 0.78, 0.80, 0.82,
and 0.85. Explain how to compare them and calculate the probability of
at least one false positive across 10 independent tests at `α=0.05`,
assuming all nulls are true.

## My answer

-   I recognized the need to compare models on the same patients and use
    bootstrap CIs.
-   I understood overlapping individual CIs do not automatically
    establish no difference.
-   I incorrectly wrote `1 − (0.05)^10` for the multiple-testing
    probability.
-   I was unsure how to design a complete comparison plan.

## Correct answer: compare paired differences

If Model A has AUROC 0.85 and Model B has AUROC 0.78:

\[ `\Delta`{=tex}=AUC_A-AUC_B=0.07. \]

For each paired bootstrap replicate:

\[ `\Delta`{=tex}\_b^*=AUC\_{A,b}\^*-AUC\_{B,b}^\*. \]

Use the same resampled patients for both models, then construct a CI for
`Δ`. Overlapping individual CIs do not prove no difference. A CI for the
difference containing zero does not prove equivalence; equivalence
requires a prespecified margin and an appropriate test.

## Multiple testing

\[ P(`\text{no false positives}`{=tex})=(1-0.05)^{10}=0.95^{10}. \]

\[
P(`\text{at least one false positive}`{=tex})=1-0.95\^{10}`\approx0.4013`{=tex}.
\]

The probability is approximately **40.1%**.

Possible approaches: - Holm--Bonferroni or Bonferroni to control
family-wise error. - Benjamini--Hochberg to control false discovery rate
in suitable settings. - Prespecify a primary comparison and treat other
analyses as secondary or exploratory.

## A rigorous plan

1.  Prespecify the primary metric and comparison.
2.  Freeze the independent test cohort.
3.  Evaluate models on the same patients.
4.  Estimate the paired performance difference and its CI.
5.  Address multiplicity for additional confirmatory tests.
6.  Report effect size, uncertainty, and clinical relevance---not only
    p-values.

------------------------------------------------------------------------

# Question 9 --- Bayesian versus frequentist reasoning

## Question

A treatment-effect study reports a frequentist 95% CI `[0.2, 2.8]` and
Bayesian 95% credible interval `[0.3, 2.6]`. Explain the intervals,
Bayes' theorem, priors, and what it means that both intervals exclude
zero.

## My answer

-   I gave a mostly correct broad distinction between confidence and
    credible intervals.
-   I was unsure of Bayes' theorem.
-   I said different priors might change the sample distribution and
    model training.
-   I initially overlooked that neither interval includes zero.

## Correct answer

Bayes' theorem is:

\[
p(`\theta`{=tex}`\mid `{=tex}D)=`\frac{p(D\mid\theta)p(\theta)}{p(D)}`{=tex}.
\]

-   `p(θ)`: prior distribution.
-   `p(D | θ)`: likelihood.
-   `p(θ | D)`: posterior distribution.
-   `p(D)`: evidence/normalizing constant.

**Posterior = prior updated by evidence from the data.**

A prior is a distribution over unknown parameters or hypotheses; it does
not change the observed samples. Different reasonable priors may produce
different posteriors, especially with small or weakly informative
datasets. Prior sensitivity should be checked when conclusions may
depend on the prior.

**Confidence interval:** A procedure with 95% long-run coverage under
its assumptions.

**Credible interval:** Contains 95% of posterior probability conditional
on the data, prior, and model.

Both given intervals exclude zero. The frequentist interval provides
evidence against a zero effect under the corresponding test assumptions.
The Bayesian interval places at least 95% posterior mass on positive
values. Neither alone proves clinical importance or causality.

------------------------------------------------------------------------

# Question 10 --- Uncertainty and external validation

## Question

A multimodal model has internal AUROC 0.87 with good calibration,
external AUROC 0.76 with poor calibration, and external AUROC 0.76 after
recalibration with improved calibration. The external hospital has more
missing wearable data and lower disease prevalence. Explain uncertainty,
propose experiments, and decide whether to deploy.

## My answer

-   I described aleatoric uncertainty as random and epistemic
    uncertainty as caused by design, timing, or collection errors.
-   I recognized the external performance decline and that poor
    calibration can coexist with good discrimination, but confused
    recalibration with threshold adjustment.
-   I proposed checking missingness/prevalence, matching the external
    data distribution to the development data, estimating intervals, and
    examining subgroups.
-   I would not deploy immediately and would communicate
    false-positive/false-negative risks.

## Correct answer: uncertainty

-   **Aleatoric uncertainty:** Inherent outcome variability conditional
    on available information; e.g., similar patients may have different
    outcomes due to biological variability.
-   **Epistemic uncertainty:** Limited knowledge of the underlying
    relationship, insufficient training coverage, uncertain parameters,
    or model misspecification; e.g., few elderly patients in the
    training data.

A hospital difference may reveal distribution shift or model knowledge
gaps, but it is not automatically epistemic uncertainty. Measurement
noise and systematic errors should also be investigated.

## Correct answer: external performance

The drop from AUROC 0.87 to 0.76 suggests worse discrimination at the
external hospital.

Recalibration improves the relationship between predicted probabilities
and observed frequencies. A conventional strictly increasing
recalibration mapping preserves ranking and therefore does not usually
change AUROC. Changing the decision threshold changes binary
classifications but also does not change AUROC.

Discrimination, calibration, and clinical utility are different
properties.

## Rigorous investigation plan

1.  Check feature definitions, units, timestamps, sensor differences,
    outcome definitions, and missingness.
2.  Compare demographics, features, prevalence, and missingness across
    hospitals.
3.  Quantify uncertainty in AUROC, AUPRC, calibration, and performance
    differences with patient-level methods.
4.  Test prespecified subgroups and robustness to missingness.
5.  If recalibration is needed, use a dedicated adaptation cohort and
    evaluate on separate, representative patients.
6.  Assess false-positive/false-negative consequences at prespecified
    thresholds and consider prospective evaluation.

Do not modify the external test cohort merely to make it resemble the
development cohort. The external test should represent the target
deployment population.

A lower prevalence can reduce precision and change AUPRC even when
class-conditional ranking remains unchanged. A change in the
relationship between features and outcomes can also reduce
discrimination.

## Deployment recommendation

I would not recommend immediate deployment based only on these results.
I would investigate the performance drop, quantify uncertainty, validate
calibration independently, assess clinical utility, and communicate
remaining limitations and error risks to clinicians.

------------------------------------------------------------------------

# Formula sheet

## Bayes' theorem and PPV

\[ P(A`\mid `{=tex}B)=`\frac{P(A\cap B)}{P(B)}`{=tex} \]

\[ PPV=`\frac{Se\cdot Prev}{Se\cdot Prev+(1-Sp)(1-Prev)}`{=tex} \]

## Expectation and variance

\[ E\[X\]=`\sum`{=tex}\_x xP(X=x) \]

\[ Var(X)=E\[X\^2\]-(E\[X\])\^2 \]

## Standard error and confidence interval

\[ SE(`\bar `{=tex}X)=`\frac{\sigma}{\sqrt n}`{=tex} \]

\[
CI=`\text{estimate}`{=tex}`\pm`{=tex}`\text{critical value}`{=tex}`\times `{=tex}SE
\]

## Hypothesis testing and power

\[ z=`\frac{\bar x-\mu_0}{\sigma/\sqrt n}`{=tex} \]

\[ Power=1-`\beta`{=tex} \]

Approximate sample size for a two-sided z-test:

\[
n`\approx`{=tex}`\left`{=tex}(`\frac{(z_{1-\alpha/2}+z_{1-\beta})\sigma}{\delta}`{=tex}`\right`{=tex})\^2
\]

## Paired bootstrap comparison

\[ `\Delta`{=tex}\_b^*=AUC\_{A,b}\^*-AUC\_{B,b}^\* \]

## Multiple testing

\[ P(`\text{at least one false positive}`{=tex})=1-(1-`\alpha`{=tex})\^m
\]

## Bayesian inference

\[
p(`\theta`{=tex}`\mid `{=tex}D)=`\frac{p(D\mid\theta)p(\theta)}{p(D)}`{=tex}
\]

------------------------------------------------------------------------

# My strengths and weaknesses

## Strengths

-   Probability and basic calculations.
-   Expectation, variance, standard error, and confidence-interval
    arithmetic.
-   Confusion matrices and base-rate intuition.
-   Recognizing the importance of patient-level evaluation.
-   Caution about deploying a model with weaker external performance.

## Weaknesses to prioritize

1.  P-value and hypothesis-test interpretation.
2.  Formal meaning of confidence intervals.
3.  Distinguishing test uncertainty from training-pipeline uncertainty.
4.  Paired statistical comparisons.
5.  Multiple-testing correction.
6.  Bayesian priors and posteriors.
7.  External validation without compromising the test cohort.

------------------------------------------------------------------------

# Targeted revision plan

  -----------------------------------------------------------------------
  Priority                Topic                   Practice task
  ----------------------- ----------------------- -----------------------
  1                       Hypothesis testing      State hypotheses,
                                                  calculate z, interpret
                                                  p-values, explain Type
                                                  I/II errors

  2                       Bayesian inference      Explain prior,
                                                  likelihood, posterior,
                                                  and evidence from
                                                  memory

  3                       Paired bootstrap        Calculate paired AUROC
                                                  difference and CI using
                                                  patient-level
                                                  resampling

  4                       Multiple testing        Calculate family-wise
                                                  error for 5, 10, and 20
                                                  tests; explain Holm and
                                                  Benjamini--Hochberg

  5                       External validation     Draft a plan for
                                                  investigating a
                                                  performance drop at a
                                                  second hospital

  6                       Uncertainty             Explain what
                                                  fixed-prediction
                                                  bootstrap, repeated
                                                  seeds, repeated
                                                  training, and external
                                                  validation each capture
  -----------------------------------------------------------------------

### Self-test before moving on

1.  Why is a p-value not the probability that the null hypothesis is
    true?
2.  What is the difference between a confidence interval and a credible
    interval?
3.  Why compare two models using paired differences when evaluated on
    the same patients?
4.  What uncertainty does bootstrapping fixed predictions capture, and
    what does it miss?
5.  Why should an external test cohort not automatically be resampled to
    match the development distribution?

------------------------------------------------------------------------

# Recommended resources

-   [Seeing Theory](https://seeing-theory.brown.edu/) --- interactive
    probability and statistical inference.
-   [An Introduction to Statistical
    Learning](https://www.statlearning.com/) --- statistical learning,
    resampling, and model evaluation.
-   [scikit-learn: Probability
    Calibration](https://scikit-learn.org/stable/modules/calibration.html)
    --- calibration and probability reliability.
-   [scikit-learn:
    Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
    --- validation design and model selection.
-   [StatQuest on YouTube](https://www.youtube.com/@statquest) ---
    accessible explanations of statistics and machine learning.

## Final takeaway

Getting the calculation right is only half the work. I also need to
explain what the quantity measures, which assumptions it relies on, how
uncertainty should be estimated, and what conclusion the evidence
actually supports.
