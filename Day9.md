# Day 9 --- Multimodal Learning for Medical AI

**Topics:** Multimodal fusion, cross-attention, missing modalities,
temporal learning, and integrated research design.

## Learning objectives

1.  Combine MRI, clinical text, wearable signals, and EHR data.
2.  Distinguish early, intermediate, and late fusion.
3.  Understand cross-attention and its matrix dimensions.
4.  Handle missing modalities and understand MCAR, MAR, and MNAR.
5.  Align asynchronous medical observations and model temporal
    dependencies.
6.  Prevent temporal information leakage.
7.  Design and evaluate a multimodal medical AI system.

------------------------------------------------------------------------

# Knowledge 1 --- Early, Intermediate, and Late Fusion

## Core knowledge

Multimodal fusion combines different data sources, such as MRI, clinical
reports, wearable signals, and EHR data.

### Early fusion

Combines inputs or early features before a shared model.

Example:

\[
X\_{`\text{fusion}`{=tex}}=\[X\_{`\text{image}`{=tex}};X\_{`\text{text}`{=tex}};X\_{`\text{wearable}`{=tex}}\]
\]

If the feature dimensions are 256, 256, and 64:

\[ 256+256+64=576 \]

**Advantages:** modalities can influence shared representation learning
early; implementation can be straightforward.

**Limitations:** different scales and formats require care; missing
modalities can be difficult; modality-specific structure may be lost if
combination is poorly designed.

### Intermediate fusion

Each modality first passes through its own encoder, then the learned
features are combined.

Example: 3D ViT for MRI, BERT for text, and an MLP or time-series
encoder for wearable signals.

**Advantages:** preserves modality-specific representation learning;
allows different encoders; can support cross-modal feature interactions.

**Limitations:** multiple encoders increase computation; fusion requires
careful design; simple concatenation does not automatically create
detailed token-level interaction.

### Late fusion

Independent models produce predictions, which are combined.

\[ P\_{`\text{final}`{=tex}}=`\sum`{=tex}\_i w_iP_i \]

Example:

\[
P\_{`\text{MRI}`{=tex}}=0.90,`\quad `{=tex}P\_{`\text{text}`{=tex}}=0.80,`\quad `{=tex}P\_{`\text{wearable}`{=tex}}=0.60
\]

Simple average:

\[ P\_{`\text{final}`{=tex}}=(0.90+0.80+0.60)/3=0.7667 \]

**Advantages:** can combine existing models; predictions can be
inspected separately; can be flexible when modalities are unavailable.

**Limitations:** fine-grained feature interactions may be lost;
probabilities should be comparable and calibrated; weights can overfit
or fail under distribution shift.

  -----------------------------------------------------------------------
  Property          Early fusion      Intermediate      Late fusion
                                      fusion            
  ----------------- ----------------- ----------------- -----------------
  Combination       Inputs/early      Learned features  Predictions
                    features                            

  Separate encoders Not always        Usually           Yes

  Interaction       Early/shared      Feature-level,    Limited
                    learning          depending on      
                                      mechanism         

  Missing           Can be difficult  Requires design   Often more
  modalities                                            flexible
  -----------------------------------------------------------------------

There is no universally best fusion strategy. Choice depends on the
task, sample size, modality alignment, missingness, data quality, and
computational resources.

## Knowledge 1 exercises and your answers

### Exercise 1 --- Identify fusion strategies

**Question:** Model A combines numerical inputs before a shared network;
Model B uses separate encoders and combines learned features; Model C
combines predictions from independent models.

**Your answers** 1. A = early fusion; B = intermediate fusion; C = late
fusion. 2. Intermediate fusion allows each modality to learn its own
representation first. 3. Late fusion is suitable for combining existing
models without retraining their encoders. 4. Intermediate fusion may
better capture cross-modal feature interactions.

**Review:** All correct.

**Refinement:** Intermediate fusion can support detailed interactions,
but simple concatenation does not automatically learn sophisticated
relationships. Cross-attention is one way to model token-level
interactions.

### Exercise 2 --- Weighted fusion

Given (P_A=0.85) and (P_B=0.65).

**Your answers and correct calculations**

Simple average:

\[ (0.85+0.65)/2=0.75 \]

Weights 0.6 and 0.4:

\[ 0.6(0.85)+0.4(0.65)=0.77 \]

Weights 0.8 and 0.2:

\[ 0.8(0.85)+0.2(0.65)=0.81 \]

All calculations were correct.

**Why weighted fusion may help:** one modality may be more informative
or reliable for a particular task.

**Why it can fail:** arbitrary or overfit weights, miscalibrated
probabilities, correlated model errors, and changing modality
reliability across hospitals.

**Correction:** Weighted fusion is a legitimate modelling strategy, not
inherently data manipulation. Select weights using training/validation
data, never by optimizing on the test set.

### Exercise 3 --- Research interpretation

**Question:** Early fusion AUROC 0.82, intermediate 0.89, late 0.85.
Researchers claim intermediate fusion is universally best.

**Your answer:** You disagreed because best AUROC does not guarantee
best AUPRC. You also called for calibration, sensitivity, specificity,
bootstrap confidence intervals, and consideration of computational cost
and complexity.

**Review:** Correct.

Additional considerations: - AUROC does not require positive prevalence
to be calculated; AUPRC is particularly sensitive to prevalence and
useful for imbalanced outcomes. - Dataset size affects metric
uncertainty. - Paired bootstrap resampling on the same held-out patients
can help compare models. - External validation, subgroup analysis,
missing-modality experiments, and clinical utility are also important. -
Late fusion is not necessarily cheaper because multiple encoders may
still be needed at inference.

**Key lesson:** The highest AUROC in one experiment does not establish a
universally superior architecture.

------------------------------------------------------------------------

# Knowledge 2 --- Cross-Attention and Modality Interaction

## Core knowledge

### Self-attention

Queries, keys, and values come from the same sequence:

\[ Q=XW_Q,`\quad `{=tex}K=XW_K,`\quad `{=tex}V=XW_V \]

\[
Attention(Q,K,V)=softmax`\left`{=tex}(`\frac{QK^T}{\sqrt{d_k}}`{=tex}`\right`{=tex})V
\]

### Cross-attention

Queries come from one modality; keys and values come from another.

Example: MRI features provide Q, and clinical text features provide K
and V.

Suppose:

\[
X_I`\in`{=tex}`\mathbb{R}`{=tex}\^{196`\times256`{=tex}},`\quad`{=tex}
X_T`\in`{=tex}`\mathbb{R}`{=tex}\^{32`\times256`{=tex}} \]

Then:

\[ QK\^T:(196`\times256`{=tex})(256`\times32`{=tex})=196`\times32`{=tex}
\]

The output is:

\[ AV:(196`\times32`{=tex})(32`\times256`{=tex})=196`\times256`{=tex} \]

Each MRI token can attend to all 32 text tokens.

Attention weights describe information routing, but they are not
automatically faithful clinical explanations or causal evidence.

### Unidirectional and bidirectional cross-attention

Unidirectional:

\[ O_I=CrossAttention(X_I,X_T,X_T) \]

MRI representations are updated using text.

Bidirectional:

\[ O_I=CrossAttention(X_I,X_T,X_T) \]

\[ O_T=CrossAttention(X_T,X_I,X_I) \]

Both modalities are updated. Bidirectional attention is not
automatically better and may increase computation and complexity.

### Multi-head cross-attention

For model dimension 256 and 8 heads:

\[ d_k=256/8=32 \]

Heads may learn different relationships, but they are not guaranteed to
correspond to predefined clinical concepts.

## Knowledge 2 exercises and your answers

### Exercise 1 --- Matrix dimensions

**Scenario:** 128 MRI tokens, 40 text tokens, embedding dimension 512, 8
attention heads.

Your submitted numbers were 65,536; 20,480; 5,120; 2,621,440; 5,120; and
40,960. Several were valid element or operation counts, but matrix
shapes and counts were mixed together.

**Correct derivation**

MRI feature matrix:

\[ X_I`\in`{=tex}`\mathbb{R}`{=tex}\^{128`\times512`{=tex}} \]

Elements: (128`\times512`{=tex}=65,536).

Text feature matrix:

\[ X_T`\in`{=tex}`\mathbb{R}`{=tex}\^{40`\times512`{=tex}} \]

Elements: (40`\times512`{=tex}=20,480).

Dimension per head:

\[ d_k=512/8=64 \]

For one head:

\[ Q`\in`{=tex}`\mathbb{R}`{=tex}\^{128`\times64`{=tex}},`\quad`{=tex}
K,V`\in`{=tex}`\mathbb{R}`{=tex}\^{40`\times64`{=tex}} \]

Attention matrix:

\[ QK\^T:(128`\times64`{=tex})(64`\times40`{=tex})=128`\times40`{=tex}
\]

Entries: (128`\times40`{=tex}=5,120).

Output for one head:

\[ (128`\times40`{=tex})(40`\times64`{=tex})=128`\times64`{=tex} \]

Output elements: (128`\times64`{=tex}=8,192).

Concatenated eight-head output:

\[ 128`\times`{=tex}(8`\times64`{=tex})=128`\times512`{=tex} \]

Elements: 65,536.

**Where two of your other numbers came from**

All attention-matrix entries across eight heads:

\[ 128`\times40`{=tex}`\times8`{=tex}=40,960 \]

Approximate scalar multiplications for QK products across eight heads:

\[ 128`\times40`{=tex}`\times64`{=tex}`\times8`{=tex}=2,621,440 \]

**Revision rule:** Write matrix shapes first; calculate element counts
and operation counts separately.

### Exercise 2 --- Architectural reasoning

**Your answers** 1. Model B uses cross-attention. 2. Model A combines
final embeddings and uses an MLP; Model B learns relationships between
modalities through cross-attention. 3. Cross-attention may help when
different MRI regions correspond to different clinical findings. 4.
Model B may be complex, computationally expensive, and prone to
overfitting. 5. You would not automatically choose Model B for a small
dataset.

**Review:** Strong reasoning.

**Refinement:** Model A's MLP can still learn relationships between
final embeddings. Model B offers more explicit token-to-token
interaction. Cross-attention can still be useful with small datasets if
pretrained encoders and suitable regularization are available.

### Exercise 3 --- Noisy reports and cross-attention

**Scenario:** A brain tumour model uses bidirectional cross-attention,
but 8% of reports contain incorrect anatomical descriptions. It achieves
higher AUROC than MRI-only. Researchers claim cross-attention improves
tumour detection.

**Your answer:** You rejected the conclusion because incorrect
image-report correspondence and noisy supervision can mislead learning.
You called for external validation, ablations, and clinical evaluation.
You also raised possible future-information leakage.

**Review and correction** - Incorrect report-image correspondence can
introduce noisy supervision. - Bidirectional cross-attention does not
inherently create temporal leakage. Leakage occurs when the model
receives information unavailable at prediction time, such as a report
written after that time or a target diagnosis included in the input. -
Compare MRI-only, text-only, concatenation, unidirectional
cross-attention, and bidirectional cross-attention. - Audit
correspondence and laterality. - Use external validation and clinically
meaningful metrics.

**Key lesson:** A performance gain does not prove that cross-attention
learned clinically meaningful relationships.

------------------------------------------------------------------------

# Knowledge 3 --- Missing Modalities and Robust Multimodal Learning

## Core knowledge

A modality may be entirely unavailable, or individual values within a
modality may be missing.

### MCAR, MAR, and MNAR

  -----------------------------------------------------------------------
  Mechanism               Meaning                 Example
  ----------------------- ----------------------- -----------------------
  MCAR                    Missingness unrelated   Random device
                          to observed or          disconnection
                          unobserved data         

  MAR                     Missingness depends on  Monitoring availability
                          observed variables      differs by observed age
                                                  group

  MNAR                    Missingness depends on  Monitoring is more
                          the missing value or    likely when underlying
                          unobserved information  SpO₂ is unusually low
  -----------------------------------------------------------------------

The classification depends on what is observed and conditioned on. If
missingness depends on an unobserved value, it may be MNAR; if the
relevant explanatory variables are observed and accounted for, the
mechanism may instead be MAR.

Complete-case analysis can introduce selection bias if patients with
complete records differ systematically from those without complete
records.

### Missing-modality strategies

**Modality dropout:** Randomly remove modalities during training so the
model learns to operate with incomplete inputs. It does not generate
missing values.

**Missingness indicators:** Explicitly identify which modalities or
measurements are available.

\[ M=\[1,0,1\] \]

This means MRI and EHR are available; text is missing.

A modality-level indicator differs from a feature-level indicator for a
missing laboratory or sensor value.

**Flexible modality-specific encoders:** Use separate encoders for MRI,
text, wearables, and EHR, then combine available representations using
availability-aware fusion. Separate encoders do not automatically
reconstruct missing modalities.

**Imputation and reconstruction:** Estimate missing measurements or
representations using statistical or deep-learning approaches. Examples
include mean/median, KNN, multiple imputation, autoencoders, variational
autoencoders, and generative models.

Imputed values are not equivalent to measured clinical values. Preserve
that distinction and evaluate uncertainty and reconstruction error.

### Robustness evaluation

Evaluate realistic combinations of available and missing modalities.
Measure AUROC, AUPRC, sensitivity, specificity, calibration, and
confidence intervals for each condition. Include external validation and
clinically relevant subgroups.

## Knowledge 3 exercises and your answers

### Exercise 1 --- Missing-data mechanisms

**Your answers** 1. Random technical disconnection: MCAR. 2. Missingness
related to observed age/monitoring practice: MAR. 3. Missingness related
to underlying low SpO₂: MNAR. 4. Removing incomplete patients may create
selection bias because examination intensity and data completeness can
depend on illness severity.

**Review:** All classifications were appropriate under the scenario
assumptions. Your selection-bias explanation was strong.

### Exercise 2 --- Missing-modality architecture

**Your answers** 1. Modality dropout holds out modalities during
training so the model gets used to missing inputs. 2. A missing-modality
indicator uses 0/1 to mark availability. 3. Flexible encoders may learn
dependencies between missing and complete data and help use time-series
information. 4. Imputation can overlook temporal dependencies, periodic
patterns, and peaks. 5. You would initially use missingness indicators
because they do not invent measurements or disrupt existing patterns.

**Corrections** - Modality dropout trains the model to operate without
an input; it does not generate missing values. - Flexible encoders
preserve modality-specific representations and allow available
representations to be fused. They do not automatically predict missing
values. - Missingness indicators are a sensible starting point. A
broader investigation could combine indicators, availability-aware
fusion, and modality dropout. - Imputation should account for temporal
structure and uncertainty.

### Exercise 3 --- Robustness claim

**Scenario**

  Condition             AUROC
  ------------------- -------
  All modalities         0.93
  MRI missing            0.75
  Text missing           0.91
  Wearables missing      0.88

Researchers claim robustness because complete-data AUROC is 0.93.

**Your answer:** You disagreed and identified modality dependence,
clinical availability patterns, MCAR/MAR/MNAR, modality dropout,
external validation, calibration, and deployment.

**Review:** Correct.

MRI-missing AUROC drops by:

\[ 0.93-0.75=0.18 \]

This suggests substantial dependence on MRI.

Further considerations: - Missing modalities may be correlated. For
example, wearable data and text describing wearable data may be jointly
unavailable. - Modality dropout trains for missing inputs; it does not
impute them. - Evaluate calibration separately for each missing-modality
pattern. - Test realistic hospital availability patterns and external
populations. - High complete-data performance does not establish
robustness or deployment readiness.

------------------------------------------------------------------------

# Knowledge 4 --- Multimodal Temporal Learning

## Core knowledge

Medical modalities are measured at different frequencies and times.

  Modality           Example frequency
  ------------------ -------------------------
  Wearable signals   Every minute
  SpO₂               Every five minutes
  Blood tests        Every eight hours
  Clinical notes     Irregularly
  MRI                Once every several days

### Static and dynamic information

Static information is relatively stable during the prediction period,
such as age and medical history.

Dynamic information changes over time, such as heart rate, SpO₂, blood
pressure, laboratory results, and clinical observations.

A heart rate of 115 bpm has different significance depending on whether
it is stable or has risen rapidly from 70 bpm.

### Temporal alignment strategies

**Resampling:** Convert measurements to a common interval. - Advantage:
easier alignment. - Limitation: may lose fine-grained patterns or
introduce artificial values.

**Time-window aggregation:** Summarize data over intervals using mean,
maximum, minimum, standard deviation, or trends. - Advantage: reduces
sequence complexity. - Limitation: may remove short-lived peaks and
patterns.

**Timestamp embeddings:** Supply actual timestamps or time differences.

\[ t=\[0,1,4,10\],`\quad `{=tex}`\Delta `{=tex}t=\[1,3,6\] \]

This helps distinguish observations one hour apart from observations six
hours apart.

**Asynchronous multimodal learning:** Preserve separate timestamps and
sequence lengths, then learn cross-modal relationships using temporal
attention.

### Temporal encoders

  -----------------------------------------------------------------------
  Model                   Main role               Considerations
  ----------------------- ----------------------- -----------------------
  LSTM                    Sequential time-series  Sequential computation;
                          representation          irregular time needs
                                                  handling

  GRU                     Gated sequential        Often a simpler
                          modelling               recurrent alternative

  TCN                     Temporal convolution    Receptive field must
                          and local patterns      cover relevant horizon

  Temporal Transformer    Attention across time   Needs explicit time
                          points                  information for
                                                  irregular data
  -----------------------------------------------------------------------

A standard LSTM is not bidirectional. A bidirectional LSTM processes
both forward and backward directions. For real-time prediction, it must
not use observations unavailable at prediction time.

### Temporal information leakage

For a prediction made at 12:00 Monday, a laboratory result collected at
16:00 Monday must not be used if it was unavailable at prediction time.

Prevent leakage by: 1. Defining prediction time and horizon. 2. Using
only information available by prediction time. 3. Verifying collection,
documentation, and availability timestamps. 4. Using patient-level
splitting. 5. Considering chronological validation. 6. Performing
external validation.

The time the information actually became available to the clinical
system matters, not just when it was eventually entered into a database.

## Knowledge 4 exercises and your answers

### Exercise 1 --- Temporal alignment

**Your answers** - Direct concatenation is difficult because modalities
have different sampling intervals, formats, and missing-value
patterns. - Resampling creates a common interval but may omit useful
observations and patterns. - Aggregation reduces sequence complexity. -
Actual timestamps reveal time gaps between admission, treatments, and
tests. - You would initially investigate timestamp embeddings.

**Review:** Correct reasoning.

**Additional detail:** Aggregation can produce clinically meaningful
summary statistics. Timestamp embeddings should be considered alongside
elapsed intervals, time since last observation, and missingness
indicators. Timestamp embeddings alone do not solve every alignment
problem.

### Exercise 2 --- Temporal architecture

**Your answers** 1. LSTM can capture temporal dependencies in wearable
data. 2. A time-aware Transformer can model irregular laboratory
observations and learn relationships using attention. 3. BERT extracts
contextual features from clinical notes. 4. Cross-attention learns
relationships between modalities. 5. The architecture may require
substantial computation, many parameters, and a larger dataset.

**Review:** Broadly correct.

**Corrections** - A standard LSTM is not bidirectional. A bidirectional
LSTM uses both directions and can leak future information if used
incorrectly for real-time prediction. - Attention weights do not
automatically represent clinical importance. - Time-aware Transformers
need explicit temporal information such as timestamps and elapsed
intervals. - Cross-attention requires careful alignment and leakage
prevention.

### Exercise 3 --- Temporal leakage

**Scenario:** A model predicts deterioration within 24 hours using the
previous 24 hours of heart rate, laboratory results, and clinical notes,
but also includes a laboratory result collected six hours after
prediction time. It achieves AUROC 0.96.

**Your answer:** You rejected deployment readiness and listed temporal
leakage, prediction time, prediction horizon, timestamp reliability,
patient-level and chronological splitting, external validation,
calibration, and clinical usefulness.

**Review:** Correct. The post-prediction laboratory result is future
information and must be excluded. The AUROC may be inflated and cannot
support deployment readiness.

------------------------------------------------------------------------

# Knowledge 5 --- Integrated Multimodal Medical AI Research Design

This section combines the previous four knowledge areas. It follows the
knowledge-first learning format rather than adding another question set.

## 1. Define the clinical research problem

Example objective: predict neurological deterioration in hospitalized
patients within the next 24 hours.

\[ X\_{`\leq `{=tex}t}`\rightarrow `{=tex}P(Y\_{t:t+24h}=1) \]

Define the target population, inclusion/exclusion criteria, outcome,
prediction time, prediction horizon, and intended clinical application
before choosing an architecture.

## 2. Modality-specific preprocessing

  -----------------------------------------------------------------------
  Modality                            Preprocessing considerations
  ----------------------------------- -----------------------------------
  MRI                                 Orientation, voxel spacing,
                                      intensity normalization,
                                      registration, spatial metadata

  Clinical text                       Tokenization, medical terminology,
                                      negation, uncertainty, timestamps

  Wearables                           Filtering, noise removal,
                                      missingness, temporal alignment

  EHR                                 Units, laboratory normalization,
                                      medication encoding, missingness,
                                      event timestamps
  -----------------------------------------------------------------------

Avoid introducing information unavailable at the intended prediction
time.

## 3. Data splitting and leakage prevention

Use patient-level splitting so all examinations from the same patient
remain in one partition for patient-independent evaluation.

An illustrative split is 70% training, 15% validation, and 15% testing.
These proportions are not universal.

Consider chronological splitting to evaluate future-period performance,
followed by external validation at an independent hospital.

## 4. Integrated architecture

  Modality        Encoder                  Main purpose
  --------------- ------------------------ --------------------------------------
  MRI             3D ViT                   Volumetric anatomical representation
  Clinical text   BERT                     Contextual language representation
  Wearables       LSTM or TCN              Temporal signal patterns
  EHR             Time-aware Transformer   Irregular clinical event modelling

Representations can be combined using cross-modal attention and gated
fusion.

\[ h\_{`\text{fusion}`{=tex}}=`\sum`{=tex}\_{i=1}\^{M}g_i h_i \]

Here (h_i) is a modality representation and (g_i) is its learned
contribution. A learned gate is not automatically a trustworthy
explanation of clinical modality importance.

## 5. Training and optimization

A multi-task objective may combine contrastive learning, classification,
segmentation, and reconstruction:

\[ L\_{`\text{total}`{=tex}}=
`\lambda`{=tex}*1L*{`\text{contrastive}`{=tex}}
+`\lambda`{=tex}*2L*{`\text{classification}`{=tex}}
+`\lambda`{=tex}*3L*{`\text{segmentation}`{=tex}}
+`\lambda`{=tex}*4L*{`\text{reconstruction}`{=tex}} \]

Possible strategies: - Transfer learning. - Careful fine-tuning. -
Different learning rates for pretrained and new layers. - AdamW. -
Learning-rate scheduling. - Gradient clipping. - Modality dropout. -
Early stopping. - Weight decay.

Use ablations to establish whether each component contributes.

## 6. Evaluation

  Task                 Metrics
  -------------------- -------------------------------------------
  Classification       AUROC, AUPRC
  Clinical detection   Sensitivity, specificity
  Segmentation         Dice, IoU, HD95
  Retrieval            Recall@K
  Calibration          Brier score, calibration curves
  Reliability          Confidence intervals
  Clinical utility     Decision-curve analysis where appropriate

Evaluate performance under different missing-modality conditions and on
external populations.

A high AUROC does not automatically mean a model is calibrated,
clinically useful, or ready for deployment.

## 7. Ablation studies

Recommended comparisons: 1. MRI only. 2. Clinical text only. 3.
Wearables only. 4. MRI + text. 5. MRI + wearables. 6. All modalities
with concatenation. 7. All modalities with cross-attention. 8. All
modalities with cross-attention and missing-modality training.

Compare performance, calibration, model size, inference time, and
external validation.

------------------------------------------------------------------------

# Day 9 revision checklist

## Knowledge 1 --- Fusion strategies

-   [ ] Understand early, intermediate, and late fusion.
-   [ ] Understand feature concatenation.
-   [ ] Understand weighted late fusion.
-   [ ] Understand fusion trade-offs.

## Knowledge 2 --- Cross-attention

-   [ ] Understand self-attention versus cross-attention.
-   [ ] Calculate Q, K, and V matrix dimensions.
-   [ ] Distinguish matrix shapes from element counts and operation
    counts.
-   [ ] Understand unidirectional and bidirectional attention.
-   [ ] Understand multi-head cross-attention.

## Knowledge 3 --- Missing modalities

-   [ ] Understand MCAR, MAR, and MNAR.
-   [ ] Understand modality dropout.
-   [ ] Understand missingness indicators.
-   [ ] Understand imputation and its limitations.
-   [ ] Evaluate realistic missing-modality combinations.

## Knowledge 4 --- Temporal learning

-   [ ] Understand different sampling frequencies.
-   [ ] Understand temporal alignment.
-   [ ] Understand resampling and time-window aggregation.
-   [ ] Understand LSTM, TCN, and temporal Transformers.
-   [ ] Understand timestamp embeddings and time differences.
-   [ ] Prevent temporal information leakage.

## Knowledge 5 --- Research design

-   [ ] Define the clinical prediction task and horizon.
-   [ ] Design modality-specific preprocessing.
-   [ ] Use patient-level and appropriate chronological splitting.
-   [ ] Design multimodal architecture and fusion.
-   [ ] Understand multi-task learning and optimization.
-   [ ] Evaluate discrimination, calibration, and clinical utility.
-   [ ] Perform ablations and external validation.

------------------------------------------------------------------------

# Recommended research materials

These are the papers and resources recommended during Day 9.

## 1. Attention Is All You Need

**Vaswani et al. (2017)**

Foundation for self-attention, multi-head attention, and Transformers.

https://arxiv.org/abs/1706.03762

Study focus: Q, K, V projections, scaled dot-product attention, and
multi-head attention.

## 2. Multimodal Transformer for Unaligned Multimodal Language Sequences

**Tsai et al. (2019)**

Relevant to cross-modal attention and asynchronous modalities.

https://arxiv.org/abs/1906.00295

Study focus: cross-modal attention and learning from unaligned
sequences.

## 3. Multimodal Machine Learning: A Survey and Taxonomy

**Baltrušaitis et al. (2019)**

A broad review of multimodal representation, alignment, fusion, and
co-learning.

https://arxiv.org/abs/1705.09406

Study focus: major challenges and design choices in multimodal learning.

## 4. Recurrent Neural Networks for Multivariate Time Series with Missing Values

**Che et al. (2018) --- GRU-D**

Relevant to missingness patterns, elapsed time, and clinical time-series
modelling.

https://arxiv.org/abs/1606.01865

Study focus: incorporating missingness and time intervals into recurrent
models.

------------------------------------------------------------------------

# Day 9 --- Key takeaways

1.  Early, intermediate, and late fusion combine modalities at different
    stages.
2.  Cross-attention allows one modality to use information from another
    at token or feature level.
3.  Matrix dimensions, element counts, and operation counts must be
    distinguished when calculating Transformer shapes.
4.  Missingness mechanisms can affect model bias and generalization.
5.  Modality dropout trains models to tolerate absent modalities; it
    does not impute missing values.
6.  Missingness indicators distinguish unavailable information from
    observed zero-valued measurements.
7.  Medical modalities often have different sampling frequencies and
    timestamps.
8.  Temporal models need actual time information when observations are
    irregular.
9.  Prediction-time definitions are essential for preventing temporal
    leakage.
10. Patient-level splitting, chronological evaluation, and external
    validation support more reliable generalization.
11. High benchmark performance does not establish clinical utility or
    deployment readiness.
12. Ablation studies help determine whether complex multimodal
    components genuinely contribute.

## Final reflection

The central lesson of Day 9 is that multimodal medical AI is not simply
about combining more data.

A reliable system must preserve important information within each
modality, understand how modalities relate, handle missing and
asynchronous observations, and avoid using information unavailable at
prediction time.

For future research discussions, keep asking: - What information does
each modality contribute? - Are the modalities correctly aligned? - Why
is information missing? - Does the model understand actual time
intervals? - Could the model be using leaked or misleading
information? - Does each architectural component provide measurable
benefit? - Does the model generalize to other hospitals? - Is the model
clinically useful and appropriately calibrated?

**Day 9 completed.**
