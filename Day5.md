# Day 5 --- Temporal Machine Learning for Wearable and Clinical Data

> **Purpose:** A complete revision guide covering the ten Day 5
> questions, your answers, corrected answers, mathematical intuition,
> revision priorities, and recommended resources.
>
> **Research context:** Predicting cardiovascular complications from
> longitudinal wearable data such as heart rate, physical activity, and
> sleep.

## Day 5 overview

  Q    Topic                                                  Approx. score
  ---- ---------------------------------------------------- ---------------
  1    Temporal dependency, windowing, prediction horizon          6.5/10\*
  2    RNN, vanishing gradients, LSTM and GRU                      6.5/10\*
  3    Temporal CNN and receptive field                            6.5/10\*
  4    Self-attention and Q/K/V                                    6.5/10\*
  5    Positional encoding and irregular sampling                    7.5/10
  6    Missing-data mechanisms and imputation                        6.6/10
  7    Temporal leakage and evaluation design                        8.1/10
  8    Survival analysis and censoring                               6.6/10
  9    Sequence prediction, pooling, attention masks                 7.9/10
  10   Complete temporal-ML research design                          7.1/10

\*The first four scores are retrospective estimates because their
earlier detailed grading is not fully available in the current
conversation. The approximate overall assessment was **6.9/10**.

**Main progress:** You became much stronger at patient-level evaluation,
temporal leakage prevention, and designing a complete medical ML
pipeline.

**Main weaknesses:** Mathematical details of temporal architectures,
survival analysis, and distinguishing calibration from threshold
adjustment.

------------------------------------------------------------------------

# Question 1 --- Temporal dependency, windowing, splitting, and prediction horizon

## Question

1.  What are temporal dependency and autocorrelation?
2.  Why use overlapping windows, and what are their disadvantages?
3.  Is randomly splitting windows into train/test valid?
4.  How should the prediction horizon be defined?

## Your answer

-   You said five-minute measurements should be considered as a sequence
    and described temporal dependency as a relationship between
    measurements over time. You were unsure about autocorrelation.
-   You said overlapping windows preserve information and continuity,
    but use more memory and create redundancy.
-   You correctly rejected random window-level splitting and proposed
    patient-level splitting.
-   You said the model should use historical data and predict whether a
    complication occurs within the next seven days.

## Correct answer

**Temporal dependency** means measurements at different time points may
be statistically related. **Autocorrelation** measures the correlation
between a series and a lagged version of itself:

\[ `\rho`{=tex}(k)=`\operatorname{Corr}`{=tex}(X_t,X\_{t-k}) \]

For example, heart rate at 08:00 may correlate with heart rate at 08:05.

A sliding window of length (L) and stride (s) is:

\[ W_i=\[x\_{is},x\_{is+1},`\ldots`{=tex},x\_{is+L-1}\] \]

Overlapping windows ((s\<L)) preserve local transitions but share many
measurements. They are highly correlated and do not represent
independent patients.

**Correct splitting procedure:** 1. Split patients first. 2. Keep all
observations from each patient in one partition. 3. Generate windows
separately within each partition.

Define an index time (t). For 30 days of input and a seven-day outcome:

\[ X_t=X\_{\[t-30d,t\]} \]

\[ Y_t=I(`\text{first complication occurs in }`{=tex}(t,t+7d\]) \]

Only information available at or before (t) may enter the input.

**Remember:** Define the index time, input window, outcome window, and
event definition before modeling.

------------------------------------------------------------------------

# Question 2 --- RNN, vanishing gradients, LSTM, and GRU

## Question

1.  Explain an RNN hidden state.
2.  Explain vanishing gradients.
3.  Explain LSTM gates and states.
4.  Compare LSTM and GRU.

## Your answer

-   You correctly said an RNN uses the current input and previous hidden
    state to produce a new hidden state.
-   You initially associated vanishing gradients with hidden-state
    memory, but correctly recognized that gradients can become very
    small during backpropagation.
-   You said LSTM uses input and forget gates, but confused the cell
    state with the hidden state.
-   You said LSTM has a forget gate and GRU is simpler.

## Correct answer

A basic RNN updates:

\[ h_t=`\tanh`{=tex}(W_xx_t+W_hh\_{t-1}+b) \]

The hidden state is a learned summary of sequence information up to time
(t), not a perfect storage of every previous observation.

During backpropagation through time, gradients are repeatedly multiplied
by derivatives and recurrent matrices. A simplified expression is:

\[ `\frac{\partial L}{\partial h_t}`{=tex}`\propto`{=tex}
`\prod`{=tex}\_{j=t+1}\^{T}`\frac{\partial h_j}{\partial h_{j-1}}`{=tex}
\]

Repeated multiplication by values with magnitude below one can make
gradients extremely small. Early time steps then receive little learning
signal.

**Correction:** Vanishing gradients are not simply caused by a hidden
state using too much memory. They are an optimization problem.

An LSTM has a separate cell state (c_t) and hidden state (h_t):

\[ f_t=`\sigma`{=tex}(W_f\[x_t,h\_{t-1}\]+b_f) \]

\[ i_t=`\sigma`{=tex}(W_i\[x_t,h\_{t-1}\]+b_i) \]

\[ `\tilde `{=tex}c_t=`\tanh`{=tex}(W_c\[x_t,h\_{t-1}\]+b_c) \]

\[ c_t=f_t`\odot `{=tex}c\_{t-1}+i_t`\odot`{=tex}`\tilde `{=tex}c_t \]

\[ o_t=`\sigma`{=tex}(W_o\[x_t,h\_{t-1}\]+b_o) \]

\[ h_t=o_t`\odot`{=tex}`\tanh`{=tex}(c_t) \]

-   Forget gate: how much old cell information to retain.
-   Input gate: how much new candidate information to write.
-   Candidate state: proposed new information.
-   Output gate: how much cell-state information to expose as hidden
    state.

A GRU has update and reset gates and does not use a separate cell state
in the same way. GRUs often have fewer parameters, but neither GRU nor
LSTM is universally better.

**Medical example:** A recurrent model may learn that a sustained change
in resting heart rate over several days is more informative than one
isolated measurement.

------------------------------------------------------------------------

# Question 3 --- Temporal CNN, receptive field, and dilation

## Question

1.  Explain a 1D temporal CNN and how it differs from an RNN.
2.  Define receptive field.
3.  Explain dilated convolution.
4.  Compare CNN and RNN for 90 days of wearable data.

## Your answer

-   You said a temporal CNN applies a kernel to adjacent points and can
    process multiple time points in parallel.
-   You defined receptive field as the input region that can influence
    an output.
-   You worried that stacking convolutional layers would create too many
    parameters and overfitting.
-   You said dilation inserts gaps between kernel positions to capture
    longer-range information.
-   You initially preferred LSTM but wanted to compare computational
    efficiency, long-term dependencies, and overfitting.

## Correct answer

A temporal convolution is a **learned weighted combination**, not
necessarily an average:

\[ y_t=`\sum`{=tex}*{j=0}\^{k-1}w_jx*{t-j}+b \]

For stride-one, non-dilated convolutions with kernel size (k), stacking
(L) layers gives:

\[ R=1+L(k-1) \]

For kernel size 3, one, two, and three layers have receptive fields 3,
5, and 7.

For a single dilated layer:

\[ R=1+(k-1)d \]

where (d) is dilation. With (k=3), dilation 1, 2, and 3 gives receptive
fields 3, 5, and 7.

**Correction:** More layers do not automatically mean an enormous
parameter count. Parameters depend on channels, kernel sizes, and
architecture. More layers can increase capacity and overfitting risk,
but parameter count and receptive field are different concepts.

  -----------------------------------------------------------------------
  Temporal CNN                        RNN/LSTM/GRU
  ----------------------------------- -----------------------------------
  Parallel across time positions      Sequential hidden-state dependency

  Often efficient                     Can be slower for long sequences

  Needs adequate receptive field      Carries context through recurrent
                                      state

  Can model local and dilated         Can model sequential dependencies
  patterns                            
  -----------------------------------------------------------------------

Compare compact temporal CNN and GRU/LSTM baselines before assuming a
Transformer is necessary.

------------------------------------------------------------------------

# Question 4 --- Self-attention and Q/K/V

## Question

1.  Explain self-attention.
2.  Explain Query, Key, and Value.
3.  Compare attention with RNNs.
4.  Can attention weights be interpreted as clinical causes? How can
    shortcuts be investigated?

## Your answer

-   You described self-attention as assigning weights to time points to
    focus on relevant information.
-   You described query as current-token information, keys as
    information from other tokens, and values as information associated
    with keys.
-   You remembered similarity and softmax but divided by the square root
    of the value.
-   You correctly noted that Transformers can process tokens in parallel
    and that measurements before a complication do not necessarily cause
    it.
-   You suggested removing highly attended data to investigate
    shortcuts.

## Correct answer

The input is projected into:

\[ Q=XW_Q,`\quad `{=tex}K=XW_K,`\quad `{=tex}V=XW_V \]

Scaled dot-product attention is:

\[ Attention(Q,K,V)=
softmax`\left`{=tex}(`\frac{QK^T}{\sqrt{d_k}}`{=tex}`\right`{=tex})V \]

-   Query: what a token is looking for.
-   Key: what a token offers for matching.
-   Value: information combined using the attention weights.

**Correction:** The denominator is (`\sqrt{d_k}`{=tex}), the square root
of the key dimension---not the value.

Transformers can compute many attention comparisons in parallel during
training. Full self-attention has approximately quadratic
sequence-length cost:

\[ O(T\^2) \]

A high attention weight does not prove causality or clinical importance.
It may reflect a predictive signal, correlation, or a data-collection
shortcut.

Investigate shortcuts using external validation, controlled
perturbations, ablations, device/hospital comparisons, and checks for
patient identity or missingness patterns. Removing attended time points
can be a sensitivity experiment, but it does not prove a shortcut.

------------------------------------------------------------------------

# Question 5 --- Positional encoding and irregular sampling

## Question

1.  Why do Transformers need positional information?
2.  Compare sinusoidal and learned positional encoding.
3.  How should irregularly sampled data be represented?
4.  Design a Transformer input for wearable data.

## Your answer

-   You correctly said position helps distinguish tokens and that
    changing order can change sequence meaning.
-   You described sinusoidal and learned embeddings and their main
    advantages/disadvantages.
-   You proposed timestamps and missingness indicators, and considered
    using positions 1, 2, 4, 5 if position 3 is missing.
-   You proposed concatenating measurements in time order and adding
    timestamps, daily/monthly information, and missingness indicators.

## Correct answer

Self-attention alone does not inherently encode token order. Positional
information helps distinguish:

\[ \[60,70,80\]`\neq[80,70,60]`{=tex}\]

A sinusoidal encoding uses deterministic sine and cosine functions:

\[
PE(pos,2i)=`\sin`{=tex}`\left`{=tex}(`\frac{pos}{10000^{2i/d}}`{=tex}`\right`{=tex})
\]

\[
PE(pos,2i+1)=`\cos`{=tex}`\left`{=tex}(`\frac{pos}{10000^{2i/d}}`{=tex}`\right`{=tex})
\]

Learned positional embeddings are trainable vectors associated with
positions.

  -----------------------------------------------------------------------
  Sinusoidal                          Learned
  ----------------------------------- -----------------------------------
  Fixed functions                     Trainable vectors

  No learned positional parameters    Adds parameters

  Can be evaluated beyond training    Often tied to a learned maximum
  positions                           length

  Structured pattern                  Adapts to training task
  -----------------------------------------------------------------------

**Important distinction:** - Position = sequence order. - Timestamp =
actual time. - Time delta = elapsed time. - Missingness indicator =
whether a measurement was observed.

For irregular data, useful inputs can include actual timestamps, time
since previous observation, time since last valid measurement, modality
identity, and missingness indicators. Positions 1, 2, 4, 5 preserve an
ordinal gap but do not necessarily encode the actual elapsed time.

A possible token representation is:

\[ z_t=E\_{`\text{measurement}`{=tex}}(x_t)+E\_{`\text{time}`{=tex}}(t)
+E\_{`\text{modality}`{=tex}}(m_t)+E\_{`\text{missingness}`{=tex}}(r_t)
\]

Do not automatically resample every modality to the same frequency if
its original timing is informative.

------------------------------------------------------------------------

# Question 6 --- Missing-data mechanisms and imputation

## Question

1.  Explain MCAR, MAR, and MNAR.
2.  Compare common imputation strategies.
3.  Can missingness itself be informative?
4.  How do you prevent preprocessing leakage?

## Your answer

-   You correctly defined MCAR, MAR, and MNAR. You used sensor
    disconnection as an MCAR example and forgetting to wear a device as
    a MAR example.
-   You explained mean/median imputation, forward filling, linear
    interpolation, and model-based imputation, with reasonable
    advantages and limitations.
-   You recognized that missingness can be informative but should be
    investigated.
-   You proposed calculating the median separately for training,
    validation, and test sets.

## Correct answer

**MCAR:** Missingness is independent of observed and unobserved data.
Example: a random hardware failure unrelated to patient or physiological
characteristics.

**MAR:** After conditioning on observed variables, missingness does not
depend on the missing value itself. Example: older patients (age is
observed) are more likely to forget to wear a device.

**MNAR:** Missingness still depends on the unobserved value or
information, even after conditioning on observed variables. Example: a
patient removes a wearable specifically when experiencing an unusually
high heart rate, and this behavior is not fully explained by observed
data.

These are assumptions about the missingness process and cannot always be
distinguished from observed data alone.

  -----------------------------------------------------------------------
  Method                  How it works            Main limitation
  ----------------------- ----------------------- -----------------------
  Mean/median             Replace missing values  Can reduce variability
                          with training-set       and ignore time
                          statistic               

  Forward fill            Carry previous value    Misleading during rapid
                          forward                 changes

  Linear interpolation    Estimate between        Uses future
                          observed points         observation; may leak
                                                  in real-time tasks

  Model-based             Predict missing values  More complex and can
                          using other information introduce imputation
                                                  error
  -----------------------------------------------------------------------

Missingness indicators can be useful:

\[ R_t=1`\text{ if observed; }`{=tex}0`\text{ if missing}`{=tex} \]

But missingness may encode hospital-specific practice, device adherence,
or workflow shortcuts.

**Important correction:** Fit imputation statistics on training data
only, then apply the same statistics to validation and test data:

\[ Median\_{`\text{train}`{=tex}}`\rightarrow`{=tex}
{`\text{train, validation, test}`{=tex}} \]

Within cross-validation, fit imputation separately inside each training
fold.

------------------------------------------------------------------------

# Question 7 --- Prediction time, temporal leakage, and evaluation design

## Question

1.  Why define an index time?
2.  Why are overlapping windows dangerous when randomly split?
3.  When should patient-level and temporal splitting be used?
4.  Design a leakage-free evaluation.

## Your answer

-   You correctly identified the index time and the risk of using future
    points in interpolation.
-   You explained how overlapping windows can share measurements across
    train/test.
-   You proposed patient-level splitting and both patient-level and
    temporal splitting for future deployment six months later.
-   You proposed 30-day inputs, seven-day outcomes, training-set median
    imputation, hyperparameter tuning,
    AUROC/AUPRC/sensitivity/specificity, and patient-level bootstrap
    confidence intervals.

## Correct answer

At index time (t):

\[ X_t=X\_{\[t-30d,t\]},`\qquad`{=tex}
Y_t=I(`\text{event in }`{=tex}(t,t+7d\]) \]

Any feature recorded after (t) is unavailable and must be excluded.

Examples of leakage: - Future heart-rate measurements. - Interpolation
using a future observation. - Diagnosis codes entered after the event. -
Preprocessing statistics estimated using test data.

**Patient-level split** evaluates generalization to unseen patients.
**Temporal split** evaluates generalization to a later calendar period.
Use both when they match the intended deployment scenario.

A robust protocol: 1. Define index time and outcome horizon. 2. Split
patients before window generation. 3. Fit preprocessing inside training
folds only. 4. Tune on development data. 5. Lock the model and evaluate
once on the untouched test set. 6. Report discrimination, calibration,
subgroup performance, and uncertainty. 7. Conduct external validation.

**Correction:** Random seeds are for stability assessment, not for
selecting the best test result. Tune genuine hyperparameters (learning
rate, dropout, weight decay, architecture size) on development data.

------------------------------------------------------------------------

# Question 8 --- Binary classification, survival analysis, and censoring

## Question

1.  Compare fixed-horizon classification and survival analysis.
2.  Explain right censoring.
3.  Define survival and hazard functions.
4.  Design a clinical model with incomplete follow-up.

## Your answer

-   You correctly distinguished a seven-day binary outcome from
    time-to-event prediction.
-   You recognized that a patient with only three days of follow-up
    cannot be confidently labeled negative for a seven-day outcome.
-   You described survival as remaining event-free and hazard as the
    chance of an event at each moment.
-   You proposed recording follow-up time and using survival analysis
    for incomplete follow-up.

## Correct answer

For a fully observed seven-day outcome:

\[ Y=I(T`\leq7`{=tex}),`\qquad `{=tex}P(T`\leq7`{=tex}`\mid `{=tex}X) \]

Survival analysis models event time while retaining censored
observations.

\[ `\tilde `{=tex}T_i=`\min`{=tex}(T_i,C_i) \]

\[ `\delta`{=tex}\_i=I(T_i`\leq `{=tex}C_i) \]

-   (`\delta=1`{=tex}): event observed.
-   (`\delta=0`{=tex}): right-censored.

A censored patient contributes the information that they remained
event-free until their last follow-up. They are not automatically a
negative case.

**Survival function:**

\[ S(t)=P(T\>t) \]

Thus:

\[ P(T`\leq `{=tex}t)=1-S(t) \]

**Hazard function:**

\[ h(t)=`\lim`{=tex}\_{`\Delta `{=tex}t`\to0`{=tex}}
`\frac{P(t\leq T<t+\Delta t\mid T\geq t)}{\Delta t}`{=tex} \]

Hazard is an instantaneous event rate conditional on remaining
event-free, not an ordinary probability.

Relationship:

\[ S(t)=`\exp`{=tex}`\left`{=tex}(-`\int`{=tex}\_0\^t
h(u),du`\right`{=tex}) \]

For an observed event at time (t), a likelihood contribution is:

\[ f(t)=h(t)S(t) \]

For a patient censored at time (c), it is:

\[ S(c) \]

Useful survival metrics: - C-index. - Time-dependent AUROC. - Brier
score or integrated Brier score. - Time-specific survival calibration.

IPCW methods can account for censoring under appropriate assumptions. If
death prevents the cardiovascular complication from occurring or being
observed, consider competing-risk methods.

**Key correction:** Do not discard all incomplete-follow-up patients
automatically, and do not claim to know their actual future event time.
Estimate survival probabilities, hazards, or risk by a specified
horizon.

------------------------------------------------------------------------

# Question 9 --- Sequence-to-one, temporal pooling, and attention masks

## Question

1.  Compare sequence-to-one and sequence-to-sequence.
2.  Explain last-token, mean, max, and attention pooling.
3.  Compare causal and bidirectional attention.
4.  Design a temporal prediction head.

## Your answer

-   You correctly defined sequence-to-one and sequence-to-sequence and
    chose sequence-to-one for one seven-day complication prediction.
-   You correctly described the four pooling methods at a high level.
-   You correctly said causal attention restricts access to later tokens
    and bidirectional attention accesses both directions.
-   You correctly recognized that bidirectional attention can be used if
    the entire sequence is historical at prediction time.
-   You proposed modality representations, concatenation,
    sequence-to-one, sigmoid, BCE, and future masking.

## Correct answer

**Sequence-to-one:**

\[ \[x_1,`\ldots`{=tex},x_T\]`\rightarrow`{=tex}`\hat `{=tex}y \]

Example: 30 days of wearable data produce one seven-day complication
risk.

**Sequence-to-sequence:**

\[
\[x_1,`\ldots`{=tex},x_T\]`\rightarrow[\hat y_1,\ldots,\hat y_T]`{=tex}\]

Example: predict a future heart-rate value at each time point.

These describe input/output structure, not a particular architecture.

Given hidden states (H=\[h_1,`\ldots`{=tex},h_T\]):

**Last-token pooling**

\[ z=h_T \]

**Mean pooling**

\[ z=`\frac{1}{T}`{=tex}`\sum`{=tex}\_{t=1}\^{T}h_t \]

**Max pooling**

\[ z_j=`\max`{=tex}*t h*{t,j} \]

**Attention pooling**

\[ e_t=v\^T`\tanh`{=tex}(Wh_t) \]

\[
`\alpha`{=tex}\_t=`\frac{\exp(e_t)}{\sum_j\exp(e_j)}`{=tex},`\qquad`{=tex}
z=`\sum`{=tex}\_t`\alpha`{=tex}\_t h_t \]

Attention pooling learns weights over time. It does not guarantee that
rare peaks are retained, and attention weights do not prove causality.

**Causal attention** prevents a token from seeing later tokens.
**Bidirectional attention** allows each token to see the entire supplied
sequence.

If prediction occurs at the end of Day 30 and input contains only Days
1--30, bidirectional attention is valid. If making a prediction at Day
10, the model must not access Days 11--30.

A binary prediction head can be:

1.  Modality-specific encoders.
2.  Intermediate fusion.
3.  Temporal Transformer.
4.  Temporal pooling.
5.  MLP.
6.  Sigmoid output.

\[ p=`\sigma`{=tex}(z)=`\frac{1}{1+e^{-z}}`{=tex} \]

\[ L=-\[y`\log`{=tex}(p)+(1-y)`\log`{=tex}(1-p)\] \]

For survival prediction, use a survival/hazard output and
survival-appropriate loss rather than ordinary binary BCE alone.

Keep separate: - Prediction cutoff: excludes data after index time. -
Attention mask: restricts token-to-token access. - Patient split:
prevents patient overlap. - Temporal split: tests later-period
generalization.

------------------------------------------------------------------------

# Question 10 --- Complete temporal-ML research design

## Question

Design a study using 500 patients, 90 days of wearable measurements,
three modalities, missing data, incomplete follow-up, and an external
hospital.

1.  Formulate the hypothesis and prediction task.
2.  Choose an architecture and temporal representation.
3.  Design experiments and baselines.
4.  Define metrics, uncertainty, subgroup analysis, and clinical
    reliability.

## Your answer

### 1. Hypothesis and task

You proposed testing whether multimodal temporal representation learning
improves seven-day cardiovascular complication prediction compared with
conventional ML and unimodal models. You chose survival analysis because
some follow-up is incomplete. You defined 30 days of input and seven
days of future outcome.

### 2. Architecture

You proposed comparing LSTM, temporal CNN, Transformer, and
combinations; imputing missing values; aligning irregular data; encoding
timestamps; using intermediate fusion and attention pooling; and
producing a sequence-to-one sigmoid output.

### 3. Experimental design

You proposed patient-level splitting, temporal splitting when future
deployment is relevant, training-set median imputation, tuning
epochs/layers/learning rate/dropout/weight decay, conventional and
deep-learning baselines, unimodal experiments, modality ablations, and
external validation.

### 4. Evaluation

You proposed AUROC, AUPRC, sensitivity, specificity, calibration, paired
patient-level bootstrap, subgroup analysis, and threshold changes.

## Correct answer

### 1. Testable hypothesis

Example primary hypothesis:

> Multimodal temporal learning improves seven-day complication
> discrimination compared with the strongest unimodal baseline, measured
> by AUPRC in an independent patient-level test cohort.

Prespecify the primary metric and comparison. Secondary hypotheses may
address calibration, missingness robustness, and external
generalization.

### 2. Prediction definition

At index time (t):

\[ X_t=X\_{\[t-30d,t\]} \]

For fully observed binary outcomes:

\[ Y_t=I(`\text{complication in }`{=tex}(t,t+7d\]) \]

If follow-up is incomplete, use survival methods or improve outcome
ascertainment. Do not label a patient with only three days of follow-up
as a seven-day negative.

### 3. Architecture and baselines

With 500 patients, model complexity must be justified by the number of
independent patients and observed events, not by the number of
overlapping windows.

  -----------------------------------------------------------------------
  Model group                         Candidates
  ----------------------------------- -----------------------------------
  Conventional ML                     Logistic regression, Elastic Net,
                                      SVM, random forest

  Recurrent                           GRU, LSTM

  Convolutional                       Temporal CNN

  Attention                           Transformer

  Clinical comparator                 Existing clinical risk score, if
                                      available
  -----------------------------------------------------------------------

Use modality-specific encoders, intermediate fusion, timestamps/time
deltas, and missingness indicators where appropriate. Compare mean,
last-token, and attention pooling.

### 4. Experimental protocol

-   Split patients before generating windows.
-   Keep every patient's windows in one partition.
-   Use chronological evaluation if future deployment is the goal.
-   Fit imputation, scaling, and feature selection on training data
    only.
-   Tune using grouped cross-validation on development data.
-   Use multiple seeds to assess stability, not to select the best test
    result.
-   Compare unimodal and multimodal models on identical partitions.
-   Perform one-modality-at-a-time ablations.
-   Lock the model before external validation.

### 5. Metrics and uncertainty

For binary classification: - AUROC and AUPRC. - Sensitivity and
specificity at prespecified thresholds. - Precision and F1 where
useful. - Brier score. - Calibration intercept and slope.

For survival: - C-index. - Time-dependent AUROC. - Brier
score/integrated Brier score. - Time-specific calibration.

For paired model comparison, bootstrap the same patients and calculate:

\[ `\Delta`{=tex}=M_A-M_B \]

Report a confidence interval for (`\Delta`{=tex}), not only separate
confidence intervals for each model.

### 6. Calibration versus threshold adjustment

**Threshold adjustment** changes the classification decision, e.g. from
0.5 to 0.2. It does not change the underlying probabilities.

**Recalibration** changes predicted probabilities so they better agree
with observed frequencies. Methods include Platt scaling, isotonic
regression, and logistic recalibration.

Estimate recalibration parameters on a suitable calibration set, never
the final test set.

### 7. Clinical reliability

Investigate subgroup performance, calibration, false
positives/negatives, missingness, device differences, hospital-specific
shortcuts, decision-curve net benefit, workflow compatibility, and
prospective performance.

Good retrospective AUROC alone is not enough for clinical deployment.

------------------------------------------------------------------------

# Core knowledge checklist

## Time-series foundations

-   Temporal dependency and autocorrelation.
-   Sliding windows, window length, and stride.
-   Patient-level versus chronological splitting.
-   Prediction index time and outcome horizon.
-   Sequence-to-one versus sequence-to-sequence.

## Temporal architectures

-   RNN hidden-state update.
-   Vanishing and exploding gradients.
-   LSTM input, forget, and output gates.
-   GRU update and reset gates.
-   Temporal convolution.
-   Receptive field and dilation.
-   Query, Key, and Value.
-   Scaled dot-product attention.
-   Causal versus bidirectional attention.
-   Sinusoidal versus learned positional encoding.
-   Temporal pooling.

## Missing data and leakage

-   MCAR, MAR, and MNAR.
-   Mean/median imputation, forward fill, interpolation, model-based
    imputation.
-   Missingness indicators.
-   Training-only preprocessing.
-   Leakage from future observations.
-   Leakage from overlapping windows.
-   Patient identity and hospital-specific shortcuts.

## Survival analysis

-   Event time and censoring time.
-   Right censoring and event indicator.
-   Survival and hazard functions.
-   Relationship between hazard and survival.
-   C-index, time-dependent AUROC, and Brier score.
-   Competing risks.

## Research design

-   Primary and secondary hypotheses.
-   Conventional and deep-learning baselines.
-   Unimodal and multimodal ablations.
-   Patient-level bootstrap.
-   External validation.
-   Calibration versus threshold adjustment.
-   Subgroup evaluation and clinical utility.

------------------------------------------------------------------------

# Your most important misconceptions to correct

  -----------------------------------------------------------------------
  Misconception                       Correct understanding
  ----------------------------------- -----------------------------------
  RNN hidden state stores all         It is a learned summary of prior
  previous data perfectly             sequence information.

  Vanishing gradients happen because  They arise from repeated gradient
  hidden states use too much memory   propagation through recurrent
                                      computation.

  CNN convolution calculates averages It calculates learned weighted
                                      combinations.

  More CNN layers necessarily mean an Parameter count depends on
  enormous parameter count            channels, kernels, and
                                      architecture.

  Attention divides QK similarity by  It divides by (`\sqrt{d_k}`{=tex}),
  the square root of V                the square root of key dimension.

  Sequence position equals elapsed    Position is order; timestamps and
  time                                time deltas represent actual time.

  Imputation statistics should be     Fit on training data and apply the
  calculated separately for           same transformation to held-out
  train/validation/test               partitions.

  No observed complication means no   With incomplete follow-up, the
  complication                        patient may be right-censored.

  Hazard is an ordinary event         Hazard is an instantaneous event
  probability                         rate conditional on remaining
                                      event-free.

  Bidirectional attention always      It is valid when the entire input
  leaks future data                   is historical at prediction time.

  Attention weights prove clinical    They do not establish causality or
  causality                           reliable explanations.

  Changing threshold recalibrates     Threshold adjustment changes
  probabilities                       decisions; recalibration changes
                                      probabilities.

  More windows mean more independent  Overlapping windows from one
  patients                            patient are correlated
                                      observations.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# Targeted revision plan

## Priority 1 --- Temporal architecture mathematics (2--3 hours)

Derive and explain: 1. RNN hidden-state equation. 2. LSTM gate
equations. 3. GRU update/reset intuition. 4. Temporal convolution. 5.
Receptive field calculation. 6. Scaled dot-product attention.

You should be able to explain every term in each equation.

## Priority 2 --- Survival analysis (2--3 hours)

Focus on: 1. Right censoring. 2. Survival and hazard functions. 3.
Likelihood contributions for events and censored observations. 4. Cox
proportional hazards. 5. C-index and time-dependent AUROC. 6. Brier
score and competing risks.

Self-test: Explain why a patient with three days of event-free follow-up
cannot be labeled a seven-day negative.

## Priority 3 --- Temporal evaluation and missingness (1--2 hours)

Practice preventing patient overlap, overlapping-window leakage,
future-data leakage, preprocessing leakage, and hospital-specific
shortcuts. Explain patient-level versus temporal generalization.

## Priority 4 --- Clinical model reliability (1--2 hours)

Revise AUROC versus AUPRC, calibration intercept/slope, Brier score,
decision thresholds, recalibration, bootstrap confidence intervals, and
decision-curve analysis.

------------------------------------------------------------------------

# Recommended resources to read and watch

## 1. RNN, LSTM, GRU, and temporal CNN

**Read** - [Dive into Deep Learning --- Recurrent Neural
Networks](https://d2l.ai/chapter_recurrent-neural-networks/index.html) -
[Understanding LSTM Networks --- Christopher
Olah](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) -
[Dive into Deep Learning --- Convolutional Neural
Networks](https://d2l.ai/chapter_convolutional-neural-networks/index.html)

**Watch** - [StatQuest: LSTM
explained](https://www.youtube.com/results?search_query=StatQuest+LSTM+explained) -
[StatQuest: RNN
explained](https://www.youtube.com/results?search_query=StatQuest+recurrent+neural+network+explained)

## 2. Transformers and attention

**Read** - [Attention Is All You
Need](https://arxiv.org/abs/1706.03762) - [The Illustrated Transformer
--- Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

**Watch** - [3Blue1Brown: Attention in
Transformers](https://www.youtube.com/results?search_query=3Blue1Brown+attention+transformers) -
[StatQuest:
Transformers](https://www.youtube.com/results?search_query=StatQuest+transformers+explained)

Prioritize Q/K/V, positional information, attention masks, and
self-attention's computational cost.

## 3. Missing data

**Read** - [Sterne et al. --- Multiple imputation for missing data in
epidemiological and clinical
research](https://www.bmj.com/content/338/bmj.b2393)

Focus on the assumptions behind MCAR, MAR, and MNAR. Imputation does not
automatically remove missing-data bias.

## 4. Survival analysis

**Read** - [Lifelines
documentation](https://lifelines.readthedocs.io/en/latest/) -
[Scikit-survival
documentation](https://scikit-survival.readthedocs.io/en/stable/user_guide/00-introduction.html)

**Watch** - [StatQuest: Kaplan--Meier and survival
analysis](https://www.youtube.com/results?search_query=StatQuest+Kaplan+Meier+survival+analysis) -
[StatQuest: Cox proportional hazards
model](https://www.youtube.com/results?search_query=StatQuest+Cox+proportional+hazards+model)

This is your highest-priority reading section.

## 5. Clinical prediction and research methodology

**Read** - [TRIPOD+AI reporting
guideline](https://www.bmj.com/content/385/bmj-2023-078378) - [Clinical
Prediction Models --- Ewout
Steyerberg](https://link.springer.com/book/10.1007/978-3-030-16399-0)

Focus on study design, validation, calibration, clinical usefulness, and
transparent reporting.

------------------------------------------------------------------------

# Final Day 5 self-assessment

Try to answer these without looking at your notes:

1.  Why can random splitting of overlapping wearable windows cause
    leakage?
2.  How does an LSTM cell state help with long-term dependencies?
3.  How does dilation expand a temporal CNN's receptive field?
4.  Why does scaled dot-product attention divide by
    (`\sqrt{d_k}`{=tex})?
5.  Why are timestamps and sequence positions not equivalent?
6.  Why should imputation statistics be fitted only on training data?
7.  What is the difference between patient-level and chronological
    validation?
8.  How does a right-censored patient contribute to a survival
    likelihood?
9.  Why can bidirectional attention be valid for a historical input
    window?
10. Why is recalibration different from changing a classification
    threshold?

## Final takeaway

Your most important progress during Day 5 was learning to think about
temporal ML as a complete prediction problem rather than simply choosing
an architecture.

Your next step is to strengthen the mathematics behind methods you can
already describe.

**Prioritize these three concepts:** 1. Recurrent and attention
mechanisms mathematically. 2. Survival analysis and incomplete
follow-up. 3. Leakage-free temporal evaluation.

If you can explain these clearly, derive their basic equations, and
defend your experimental choices, you will have a much stronger
foundation for designing longitudinal medical ML studies.
