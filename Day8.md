# Day 8 --- Computer Vision and Multimodal Medical AI

**Programme:** 14-day Medical ML learning programme\
**Topics:** MRI representation, CNNs, segmentation, transfer learning,
augmentation, Vision Transformers, and multimodal image--text learning.

## 1. Learning objectives

By the end of Day 8, the goal was to: 1. Explain MRI images, volumes,
voxel spacing, and preprocessing. 2. Understand convolution, output
dimensions, parameter counts, receptive fields, CNNs, and U-Net. 3.
Evaluate segmentation using Dice, IoU, and sensitivity. 4. Understand
transfer learning, freezing, fine-tuning, and catastrophic forgetting.
5. Choose medically valid augmentations and avoid leakage. 6. Explain
ViT patch embeddings, positional embeddings, CLS tokens, and attention
complexity. 7. Understand CLIP-style contrastive image--text learning
and medical NLP challenges. 8. Design rigorous medical computer-vision
and multimodal experiments.

------------------------------------------------------------------------

# 2. Knowledge 1 --- MRI representation and preprocessing

## Core knowledge

### Pixels, voxels, and spacing

-   **Pixel:** a 2D image element.
-   **Voxel:** a 3D volume element.
-   **Voxel spacing:** the physical dimensions represented by each
    voxel, usually in millimetres.

For example, spacing of `1 × 1 × 5 mm` means the through-plane spacing
is much larger than the in-plane spacing. Physical spacing matters for
anatomical distances, lesion size, volume measurements, and 3D models.

### MRI preprocessing

MRI intensity values are not generally standardized across scanners,
sequences, and hospitals. Depending on the task, preprocessing may
include:

-   Z-score normalization.
-   Min--max normalization, when appropriate.
-   Percentile-based intensity clipping.
-   Bias-field correction.
-   Orientation standardization and registration.
-   Resampling to a common physical voxel spacing.

Preprocessing must be justified by the imaging protocol. Resampling
changes the image grid; it cannot recover spatial information that was
never acquired.

### Distribution shift

Hospitals may differ in scanners, acquisition protocols, MRI sequences,
voxel spacing, reconstruction pipelines, preprocessing, and patient
populations. External validation is important.

## Exercise 1 --- Your answers and corrections

**Your answers** 1. Pixels are 2D units; voxels are 3D units. You noted
that voxel spacing matters for accurate anatomical interpretation. 2.
MRI intensity values are not standardized, so you proposed z-score,
min--max, percentile clipping, and bias-field correction where
appropriate. 3. You identified hospital-specific MRI standardization as
a source of distribution shift. 4. You proposed consistent image size,
anatomical position, standardization, normalization, and channels.

**What you got right** - Correct pixel/voxel distinction. - Correct
understanding of physical voxel spacing. - Correct recognition of MRI
intensity variability and cross-hospital distribution shift. -
Appropriate attention to preprocessing consistency.

**Corrections** - Do not assume every MRI should have the same channels.
T1, T2, FLAIR, and other sequences contain different information.
Preserve and document sequence identity. - Equal image dimensions do not
guarantee equal physical geometry. Record orientation, affine
transforms, and voxel spacing. - Check whether registration and
resampling distort small lesions.

**Research-level answer**

A rigorous MRI pipeline documents sequence type, scanner and acquisition
information, orientation, voxel spacing, intensity preprocessing, and
registration or resampling. It assesses whether preprocessing loses
small-lesion information. Generalization should be tested on independent
hospitals and acquisition protocols.

------------------------------------------------------------------------

# 3. Knowledge 2 --- CNNs, receptive fields, U-Net, and segmentation

## 3.1 Convolution output dimensions

For one spatial dimension:

\[ O = `\left`{=tex}`\lfloor `{=tex}`\frac{I+2P-K}{S}`{=tex}
`\right`{=tex}`\rfloor`{=tex}+1 \]

Where (I) is input size, (P) padding, (K) kernel size, (S) stride, and
(O) output size. Apply separately to height and width.

With stride 1 and suitable "same" padding, spatial dimensions can be
preserved.

## 3.2 Convolution parameter count

\[
`\text{Parameters}`{=tex}=K_hK_wC\_{`\text{in}`{=tex}}C\_{`\text{out}`{=tex}}+C\_{`\text{out}`{=tex}}
\]

The final term is one bias per output channel, if bias is enabled.

**Important correction:** Image height and width do not multiply the
number of convolution parameters. Kernel weights are shared across
spatial positions.

For a 3 × 3 kernel, 3 input channels, 32 output channels, and one bias
per output channel:

\[ 3`\times3`{=tex}`\times3`{=tex}`\times32`{=tex}+32=896 \]

## 3.3 Receptive field

The receptive field is the region of the original input that can
influence a feature-map element.

For stride-1, kernel-3 convolutions, the receptive field grows:

-   First layer: 3 × 3.
-   Second layer: 5 × 5.
-   Third layer: 7 × 7.

General recurrence:

\[ J_l=J\_{l-1}S_l \]

\[ R_l=R\_{l-1}+(K_l-1)J\_{l-1} \]

Here (J_l) is the effective jump, (R_l) is receptive-field size, (S_l)
is stride, and (K_l) is kernel size.

## 3.4 2D CNN versus 3D CNN

  -----------------------------------------------------------------------
  Consideration           2D CNN                  3D CNN
  ----------------------- ----------------------- -----------------------
  Input                   Individual slices       Volume or 3D patches

  Spatial context         Within a slice          Across three dimensions

  Memory and compute      Usually lower           Usually higher

  Inter-slice continuity  Not directly modelled   Can be modelled
                          if slices are           
                          independent             

  Clinical use            Slice-level tasks or    Tasks needing
                          constrained resources   volumetric anatomy
  -----------------------------------------------------------------------

Neither is universally better. Compare them using patient-level splits.
A **2.5D model** is another option: adjacent slices are provided as
channels to a 2D model.

## 3.5 U-Net and skip connections

U-Net is an encoder--decoder CNN for biomedical segmentation.

-   **Encoder:** extracts features while reducing spatial resolution.
-   **Bottleneck:** learns abstract features.
-   **Decoder:** upsamples features to produce a segmentation map.
-   **Skip connections:** transfer high-resolution encoder features to
    corresponding decoder stages.

Skip connections combine semantic information from deeper layers with
spatial detail from earlier layers. A 3D U-Net can model volumetric
relationships across slices.

## 3.6 Classification versus segmentation

-   **Classification:** predicts a label for an image or patient,
    e.g. whether a tumour is present.
-   **Segmentation:** predicts a class for each pixel or voxel,
    identifying the tumour's location and boundary.

For 2D segmentation:
(`\hat `{=tex}Y`\in`{=tex}{0,1}\^{H`\times `{=tex}W}).\
For 3D segmentation:
(`\hat `{=tex}Y`\in`{=tex}{0,1}\^{D`\times `{=tex}H`\times `{=tex}W}).

A classification CNN is not a substitute for a segmentation architecture
when precise boundaries are required.

## 3.7 Segmentation metrics

Let TP be correctly predicted tumour voxels, FP be healthy voxels
incorrectly predicted as tumour, and FN be missed tumour voxels.

**Dice Similarity Coefficient**

\[ Dice=`\frac{2TP}{2TP+FP+FN}`{=tex} \]

**Intersection over Union**

\[ IoU=`\frac{TP}{TP+FP+FN}`{=tex} \]

Relationship:

\[ Dice=`\frac{2IoU}{1+IoU}`{=tex} \]

**Voxel-level sensitivity**

\[ Sensitivity=`\frac{TP}{TP+FN}`{=tex} \]

**Lesion-level sensitivity** measures the proportion of individual
lesions detected, rather than the proportion of tumour voxels detected.
Detecting 4 of 5 lesions gives (4/5=0.8).

A model may achieve high Dice on large lesions while missing small
lesions. When small-lesion detection matters, also report lesion-level
sensitivity and false positives per patient. HD95 can assess boundary
distance.

## Exercises and your answers

### Exercise 1 --- Output dimensions and parameters

**Your answers:** output height 224, width 224, 32 channels; parameter
expression `224 × 224 × 3 × 3 × 33`; you noted larger stride may miss
fine detail.

**Correction:** If the input is 224 × 224 and a 3 × 3 convolution uses
stride 1 with suitable same padding, the output can remain 224 × 224.
With 32 filters, output channels are 32. Parameter count depends on
kernel and channel counts, not image height and width. For 3 input
channels, 32 output channels, 3 × 3 kernel, and bias, the count is 896.

**Revision:** Separate feature-map dimensions from trainable parameter
count.

### Exercise 2 --- Receptive field

**Your answer:** You identified `3, 5, 7, ...` and a stride-1
recurrence.

**Correction:** For stride-1, kernel-3 convolutions, the receptive field
increases by 2 at each layer. Starting from 1 before the first
convolution gives 3, 5, 7, and so on. For non-unit stride, use the
general recurrence involving (J_l).

**Research interpretation:** Small receptive fields capture local
texture and boundaries; larger receptive fields capture broader
anatomical context. Medical models often need both.

### Exercise 3 --- 2D versus 3D CNN

**Your answers:** You initially selected a 2D CNN for fine-grained slice
details, recognized 3D CNN's greater memory and computational costs, and
proposed smaller strides and kernels.

**Correction:** A 2D CNN is a defensible baseline, but independently
processed slices cannot directly model inter-slice continuity. A 3D CNN
can model volumetric anatomy at greater cost. Compare 2D, 2.5D, and/or
3D models when justified. Smaller strides reduce downsampling; smaller
kernels focus on local regions. Neither guarantees lesion preservation.

### Exercise 4 --- U-Net

**Your answers:** You preferred comparing architectures; recognized
classification cannot produce precise segmentation boundaries; said skip
connections preserve details; preferred comparing 2D and 3D U-Net;
initially did not know Dice, IoU, and sensitivity.

**Correction:** U-Net is appropriate for segmentation. Skip connections
transfer high-resolution features to the decoder. 3D U-Net models
volumetric continuity but costs more memory and compute. 2D U-Net may be
useful under resource or annotation constraints.

**Metric calculation**

Given TP = 80, FP = 20, FN = 20:

\[ Dice=`\frac{160}{160+20+20}`{=tex}=0.8 \]

\[ IoU=`\frac{80}{80+20+20}`{=tex}=0.6667 \]

\[ Sensitivity=`\frac{80}{80+20}`{=tex}=0.8 \]

Your answers---Dice 80%, IoU 66.67%, sensitivity 80%---were all correct.

------------------------------------------------------------------------

# 4. Knowledge 3 --- Transfer learning and image augmentation

## 4.1 Transfer learning

Transfer learning starts from a model pretrained on another dataset and
adapts it to a target task:

\[
`\text{Pretrained model}`{=tex}`\rightarrow`{=tex}`\text{Fine-tuning}`{=tex}`\rightarrow`{=tex}`\text{Medical task}`{=tex}
\]

Pretraining may provide useful visual representations, but natural-image
features are not guaranteed to be optimal for MRI. Compare natural-image
pretraining, medical-image pretraining, and training from scratch when
feasible.

## 4.2 Freezing versus fine-tuning

Let the pretrained backbone be (f(x;`\theta`{=tex})) and the new head be
(g(h;`\phi`{=tex})).

-   **Frozen backbone:** (`\theta`{=tex}) remains fixed; only
    (`\phi`{=tex}) is optimized.
-   **Fine-tuning:** some or all of (`\theta`{=tex}) and (`\phi`{=tex})
    are optimized.
-   **Partial fine-tuning:** freeze early layers and unfreeze selected
    later blocks.

Freezing is faster and can reduce overfitting, but prevents backbone
adaptation. Fine-tuning allows domain adaptation but costs more and may
overfit.

## 4.3 Learning rates and catastrophic forgetting

Gradient descent:

\[
`\theta`{=tex}\_{t+1}=`\theta`{=tex}*t-`\eta`{=tex}`\nabla`{=tex}*`\theta `{=tex}L
\]

A common strategy is a smaller learning rate for pretrained layers and a
larger one for a newly initialized head. For example, (10\^{-5}) for the
backbone and (10\^{-3}) for the head. These are illustrative, not
universal values.

Aggressive updates may substantially change useful pretrained
representations, contributing to catastrophic forgetting.

## 4.4 Image augmentation

Augmentation creates transformed training examples and can improve
robustness. Examples include rotation, flipping, cropping, Gaussian
noise, intensity adjustment, and elastic deformation.

More augmentation does not always improve generalization.
Transformations must be plausible and preserve clinical meaning.

Risks: - Left--right flipping can alter laterality. - Large rotations
can create unrealistic orientation. - Cropping can remove a small
lesion. - Excessive noise can obscure pathology. - Excessive elastic
deformation can distort anatomy.

### Image-mask consistency

For segmentation, apply the same spatial transformation to image and
mask:

\[ X'=T(X),`\qquad `{=tex}Y'=T(Y) \]

Use appropriate interpolation: - MRI intensity image: continuous
interpolation, such as linear or cubic depending on the operation. -
Categorical mask: nearest-neighbour interpolation to preserve discrete
class labels.

## 4.5 Augmentation leakage

Correct sequence: 1. Split patients into training, validation, and test
partitions. 2. Keep all scans from one patient in one partition. 3.
Apply random augmentation only to training data. 4. Keep validation and
test data free of random training augmentation. 5. Evaluate on an
untouched test set after model selection.

Augmented versions are not independent new patients. If transformed
versions of the same original image appear in both training and test
sets, performance can be inflated.

## Exercises and your answers

### Exercise 1 --- Transfer learning

**Your answers:** You prioritized fine-tuning a pretrained ResNet for
1,500 labelled scans; correctly explained freezing and fine-tuning;
explained why the backbone may need a smaller learning rate; proposed a
fair comparison but suggested Dice and IoU for classification.

**Correction:** Compare training from scratch, frozen backbone plus new
head, and fine-tuned backbone. Use identical patient-level splits,
preprocessing, and fair hyperparameter selection. For image-level
classification, use AUROC, AUPRC, sensitivity, specificity, F1, and
calibration---not Dice or IoU unless the model also produces masks. Use
an untouched test set and external validation where possible.

### Exercise 2 --- Augmentation validity

**Your answers:** You said augmentation can increase diversity but
excessive augmentation can remove important information; identified
rotation and flipping as possible anatomical problems; initially missed
the image-mask alignment requirement; proposed controlled comparisons.

**Correction:** Augmentation should model plausible variation while
preserving the label. Apply identical spatial transformations to image
and mask, using nearest-neighbour interpolation for categorical masks.
Compare no augmentation, individual strategies, and selected
combinations under the same experimental protocol.

### Exercise 3 --- Augmentation and leakage

**Your answers:** You identified patient-level overlap and
augmented-image overlap; correctly proposed patient-level splitting
before augmentation; proposed no-augmentation and single-augmentation
ablations.

**Correction:** Also require a separate validation set, untouched test
set, multiple seeds where appropriate, confidence intervals, external
validation, and subgroup evaluation---especially for small lesions. A
good independent test score does not by itself establish cross-hospital
generalization.

------------------------------------------------------------------------

# 5. Knowledge 4 --- Vision Transformers (ViT)

## 5.1 Patch embeddings

A ViT divides an image into patches, flattens each patch, and projects
it into an embedding space.

For a 224 × 224 image with 16 × 16 patches:

\[ N=`\frac{224}{16}`{=tex}`\times`{=tex}`\frac{224}{16}`{=tex}=196 \]

For a grayscale patch:

\[ 16`\times16`{=tex}`\times1`{=tex}=256 \]

Each flattened patch has 256 values. If the embedding dimension is 768:

\[ E`\in`{=tex}`\mathbb{R}`{=tex}\^{256`\times768`{=tex}} \]

The projected embedding has dimension 768.

## 5.2 Positional embeddings and CLS token

Positional embeddings tell the Transformer where patches originated. A
learnable classification token (`[CLS]`) is added to the patch sequence.

For 196 patches:

\[ L=196+1=197 \]

The final CLS representation can be used for classification.

## 5.3 CNN versus ViT

  -----------------------------------------------------------------------
  Property                CNN                     ViT
  ----------------------- ----------------------- -----------------------
  Main operation          Convolution             Self-attention

  Locality                Built into convolution  Learned through
                                                  attention

  Global relationships    Accumulated through     Can directly connect
                          layers                  distant patches

  Spatial information     Convolutional structure Positional embeddings

  Data needs              Often effective with    Often benefits from
                          smaller datasets        large-scale pretraining
  -----------------------------------------------------------------------

CNNs have a useful local-pattern inductive bias. ViTs can model
long-range relationships but may need substantial pretraining. Neither
is universally superior.

## 5.4 Patch-size and attention-complexity trade-off

For a 224 × 224 image:

  Patch size     Patches   Sequence length with CLS
  ------------ --------- --------------------------
  16 × 16            196                        197
  8 × 8              784                        785

The attention matrix is (L`\times `{=tex}L).

For patch size 16:

\[ 197\^2=38,809 \]

For patch size 8:

\[ 785\^2=616,225 \]

Ratio:

\[ `\frac{616,225}{38,809}`{=tex}`\approx15.88`{=tex} \]

Reducing patch size from 16 to 8 quadruples patch count but increases
attention-matrix entries by about 16 times. Attention has a quadratic
sequence-length component:

\[ O(L\^2) \]

Smaller patches provide finer spatial granularity but do not
automatically improve lesion detection. They also increase computation
and memory requirements.

## 5.5 2D versus 3D ViT and physical spacing

A 2D ViT processes patches within slices. A 3D ViT processes volumetric
patches.

For a (128`\times256`{=tex}`\times256`{=tex}) volume and patch size
(8`\times16`{=tex}`\times16`{=tex}):

\[ N=16`\times16`{=tex}`\times16`{=tex}=4096 \]

\[ N\^2=16,777,216 \]

attention-matrix entries per head and sample.

Possible efficiency strategies include 3D crops, hierarchical attention,
window-based attention, patch merging, and efficient attention.

**Important MRI distinction:**

\[
`\text{Array position}`{=tex}`\neq`{=tex}`\text{Physical position}`{=tex}
\]

Ordinary 2D positional embeddings encode positions within a slice, not
physical distances between slices. Adjacent slices may be 1 mm apart in
one scan and 5 mm apart in another.

Potential improvements: - Resample to an appropriate common physical
spacing. - Use 3D patch embeddings and 3D positional representations. -
Incorporate physical-coordinate information where appropriate. -
Preserve affine and spatial metadata. - Compare 2D, 2.5D, and 3D
approaches.

Resampling a 5 mm acquisition to a 1 mm grid does not recover
information that was never acquired.

## Exercises and your answers

### Exercise 1 --- Patch dimensions

**Scenario:** Input 224 × 224 × 1, patch size 16 × 16, embedding
dimension 768, one CLS token.

**Your answers:** 196 patches; flattened patch dimension 256; projection
matrix 256 × 768; sequence length 197; smaller patches can preserve fine
detail but increase computation.

**Correct calculations:**

\[ N=14`\times14`{=tex}=196 \]

\[ 16`\times16`{=tex}`\times1`{=tex}=256 \]

\[ E`\in`{=tex}`\mathbb{R}`{=tex}\^{256`\times768`{=tex}} \]

\[ L=196+1=197 \]

Your core calculations and reasoning were correct.

### Exercise 2 --- Attention matrix

**Your answers:** Patch 16: 196 patches, sequence 197; patch 8: 784
patches, sequence 785; matrix entries 38,809 and 616,225; ratio 15.88;
compare models rather than automatically choosing smaller patches.

**Correction:** All calculations were correct. The matrix shapes are
(197`\times197`{=tex}) and (785`\times785`{=tex}). The comparison
concerns attention-matrix size and associated attention computation;
total Transformer computation also includes projections and feed-forward
layers.

### Exercise 3 --- 2D positional information and volumetric anatomy

**Your answers:** You recognized that 2D positional embeddings do not
encode physical distance between slices; independent slices cannot
represent the full 3D lesion structure; you proposed 3D patches and a 3D
ViT.

**Correction:** Your reasoning was correct. Add physical-spacing-aware
preprocessing, appropriate resampling, and physical-coordinate
representations. A 3D ViT using only voxel-grid indices still does not
automatically understand physical distances.

------------------------------------------------------------------------

# 6. Knowledge 5 --- Multimodal medical vision and CLIP

## 6.1 Image and text encoders

A CLIP-style model uses two encoders:

\[ f\_`\theta`{=tex}(I)`\rightarrow `{=tex}z_I \]

\[ g\_`\phi`{=tex}(T)`\rightarrow `{=tex}z_T \]

The image encoder creates an image embedding; the text encoder creates a
report embedding. Both are projected into a shared space.

Cosine similarity:

\[ sim(z_I,z_T)= `\frac{z_I\cdot z_T}{\|z_I\|\|z_T\|}`{=tex} \]

## 6.2 Contrastive learning

Contrastive learning increases similarity for corresponding image--text
pairs and decreases similarity for non-corresponding pairs.

For normalized embeddings, a temperature-scaled logit is:

\[ S\_{ij}=`\frac{z_{I_i}\cdot z_{T_j}}{\tau}`{=tex} \]

For a batch of (N) pairs, (S) has shape (N`\times `{=tex}N), with
correct pairs on the diagonal.

Image-to-text loss:

\[ L\_{I`\rightarrow `{=tex}T} =
-`\frac{1}{N}`{=tex}`\sum`{=tex}*{i=1}\^{N} `\log`{=tex}
`\frac{\exp(S_{ii})}`{=tex}
{`\sum`{=tex}*{j=1}\^{N}`\exp`{=tex}(S\_{ij})} \]

The text-to-image loss is calculated similarly:

\[ L=`\frac{L_{I\rightarrow T}+L_{T\rightarrow I}}{2}`{=tex} \]

In-batch negatives may include clinically similar reports. These false
negatives can create noisy supervision.

## 6.3 Medical language challenges

Medical reports contain clinically meaningful distinctions:

-   **Negation:** "No evidence of pneumonia."
-   **Uncertainty:** "Possible pneumonia."
-   **Laterality:** "Left-sided" versus "right-sided."
-   Multiple findings.
-   Incomplete descriptions.
-   Temporal mismatch between report and image.

A model that ignores negation or laterality may learn misleading
representations. A report should not automatically be treated as a
complete, perfect annotation of every visible finding.

## 6.4 Applications

-   Zero-shot image classification.
-   Image-to-report and report-to-image retrieval.
-   Medical image question answering.
-   Image representation pretraining from paired reports.
-   Fusion of images with clinical notes, EHR, or wearable data.

Good retrieval scores do not automatically establish clinical
reliability.

## Exercises and your answers

### Exercise 1 --- Medical image-text alignment

**Your answers:** You correctly described positive and negative pairs;
identified negation, uncertainty, and laterality; explained that
incorrect image-report matching creates misleading supervision; and said
retrieval alone is insufficient, proposing confidence intervals,
calibration, and external validation.

**Correction and additions:** - Evaluate negation and laterality
explicitly. - Verify image-report correspondence, examination dates, and
patient identity. - Investigate false negatives in contrastive
batches. - Evaluate downstream clinical tasks, not retrieval alone. -
Assess expert clinical agreement and factuality where relevant. - Use
calibration when probabilities inform clinical decisions.

### Exercise 2 --- Contrastive loss

Given logits:

\[ S=\[2,1,0\] \]

The correct report is the first candidate.

**Your answers:** Correct-pair probability 0.6652; loss 0.4077. With
logits (\[4,1,0\]), probability 0.9362 and loss 0.0659.

**Correct calculations:**

\[ P\_{`\text{correct}`{=tex}}= `\frac{e^2}{e^2+e^1+e^0}`{=tex}
`\approx0.6652`{=tex} \]

\[ L=-`\log`{=tex}(0.6652)`\approx0.4077`{=tex} \]

With logits (\[4,1,0\]):

\[ P\_{`\text{correct}`{=tex}}= `\frac{e^4}{e^4+e^1+e^0}`{=tex}
`\approx0.9362`{=tex} \]

\[ L=-`\log`{=tex}(0.9362)`\approx0.0659`{=tex} \]

All calculations were correct.

**Important interpretation:** The softmax probability is the probability
of selecting the correct report among the candidates in that comparison.
It is not automatically a calibrated 93.62% probability that a clinical
diagnosis is correct.

------------------------------------------------------------------------

# 7. Final integrated research-design exercise

## Scenario

A group proposes a multimodal brain tumour system using: - 12,000 MRI
volumes from three hospitals. - Radiology reports for most
examinations. - Different voxel spacing across hospitals. - Multiple
scans from some patients. - A limited number of expert tumour masks. - A
3D ViT image encoder. - A BERT text encoder. - CLIP-style contrastive
pretraining. - A tumour classification head. - A U-Net segmentation
branch.

Their preliminary results are excellent, but they randomly split MRI
scans into training and test sets.

## Your answers, corrections, and research-level recommendations

### 1. Data and preprocessing

**Your answer:** Patient-level train/validation/test splits; standardize
voxel spacing and formats; identify missing reports; check image-report
correspondence.

**Correction:** Your split and correspondence checks were correct.
Standardize orientation and resample to an appropriate common physical
spacing when justified. Preserve spatial metadata and inspect whether
resampling loses small-lesion information.

Do not invent missing clinical reports and treat them as ground truth.
Instead: - Record missingness explicitly. - Investigate missingness
patterns. - Use available image-only data where appropriate. - Train
with missing-modality strategies. - Evaluate performance when text is
unavailable.

### 2. Architecture and representation learning

**Your answer:** 3D ViT for volumetric self-attention, BERT for text
encoding, U-Net for fine-grained and volumetric segmentation; you
identified computational cost and parameter count as limitations.

**Correction:** The component roles were broadly correct: - 3D ViT
learns volumetric image representations. - BERT learns contextual
radiology-report representations. - 3D U-Net produces voxel-level
segmentation masks using an encoder--decoder and skip connections.

Also consider annotation scarcity, data requirements, and
clinical-language errors.

### 3. Training and optimization

**Your answer:** Start with transfer learning and fine-tune using a
small learning rate.

**Correction:** This is a sensible starting point. Also include: -
Contrastive pretraining on verified image-report pairs. - Classification
and segmentation objectives. - Missing-modality handling. - Appropriate
image augmentation. - Modality ablations. - Controlled fine-tuning and
learning-rate experiments.

A possible multi-task objective:

\[ L\_{`\text{total}`{=tex}}=
`\lambda`{=tex}*1L*{`\text{contrastive}`{=tex}}
+`\lambda`{=tex}*2L*{`\text{classification}`{=tex}}
+`\lambda`{=tex}*3L*{`\text{segmentation}`{=tex}} \]

The (`\lambda`{=tex}) terms control each objective's contribution.

### 4. Evaluation and clinical reliability

**Your answer:** AUROC, AUPRC, sensitivity, specificity, calibration,
and confidence intervals for classification; Dice, IoU, and sensitivity
for segmentation; similarity for image-text alignment; external hospital
validation.

**Correction:** This was your strongest section. A fuller protocol:

  Task                   Metrics
  ---------------------- --------------------------------------------------------
  Classification         AUROC, AUPRC, sensitivity, specificity, calibration
  Segmentation           Dice, IoU, lesion-level sensitivity, HD95
  Image-text retrieval   Recall@K, median rank
  Reliability            Confidence intervals, calibration, external validation

Also require: - Image-only versus text-only versus multimodal
ablations. - Missing-modality experiments. - Negation and laterality
checks. - Hospital-level subgroup analysis. - Assessment of downstream
clinical usefulness.

## Integrated research-level model answer

> I would first establish patient-level training, validation, and test
> splits, ensuring all examinations from the same patient remain in one
> partition. I would standardize MRI orientation, voxel spacing, and
> intensity preprocessing while preserving spatial metadata.
>
> Missing radiology reports should be explicitly identified rather than
> artificially reconstructed. I would investigate missingness patterns
> and design the model to support image-only and multimodal inference.
>
> Architecturally, I would use a 3D ViT to extract volumetric image
> representations, BERT to encode contextual clinical text, and a 3D
> U-Net segmentation branch to generate voxel-level tumour predictions.
>
> I would investigate contrastive pretraining to align image and text
> embeddings, followed by multi-task learning for classification and
> segmentation. Transfer learning and carefully controlled fine-tuning
> could reduce overfitting.
>
> For evaluation, I would report AUROC, AUPRC, sensitivity, specificity,
> and calibration for classification; Dice, IoU, and lesion-level
> sensitivity for segmentation; and Recall@K for image-text retrieval.
>
> Finally, I would perform modality ablations, missing-modality
> experiments, external hospital validation, and uncertainty analysis. I
> would assess whether representations preserve clinically important
> information such as negation, anatomical location, and lesion
> characteristics.

------------------------------------------------------------------------

# 8. Day 8 revision priorities

## Priority 1 --- Revisit carefully

1.  Convolution parameter counts versus feature-map dimensions.
2.  Receptive-field calculations, especially when stride is not 1.
3.  Dice, IoU, voxel-level sensitivity, and lesion-level sensitivity.
4.  Image-mask alignment and nearest-neighbour interpolation for
    categorical masks.
5.  Missing-modality strategies and multi-task learning.

## Priority 2 --- Strengthen mathematical understanding

1.  ViT patch projection dimensions.
2.  Attention matrix dimensions and quadratic complexity.
3.  Contrastive loss and temperature scaling.
4.  Physical coordinates versus voxel-grid positions.
5.  Multi-task loss weighting.

## Priority 3 --- Read research papers

1.  **U-Net --- Ronneberger et al. (2015):** Biomedical segmentation and
    skip connections.\
    https://arxiv.org/abs/1505.04597

2.  **Vision Transformer --- Dosovitskiy et al. (2021):** Patch
    embeddings and Transformer-based image classification.\
    https://arxiv.org/abs/2010.11929

3.  **CLIP --- Radford et al. (2021):** Contrastive image--text
    representation learning.\
    https://arxiv.org/abs/2103.00020

4.  **ConVIRT --- Zhang et al. (2022):** Contrastive learning from
    paired medical images and text.\
    https://arxiv.org/abs/2010.00747

------------------------------------------------------------------------

# 9. Day 8 key takeaways

-   MRI voxel spacing represents physical geometry and matters for
    volumetric learning.
-   CNN parameter counts depend on kernel and channel dimensions, not
    image height and width.
-   3D models can capture volumetric context but require more
    computation and careful spatial preprocessing.
-   U-Net skip connections help preserve spatial detail for
    segmentation.
-   Dice, IoU, and sensitivity measure different aspects of segmentation
    performance.
-   Transfer learning can help with small labelled datasets, but
    fine-tuning requires careful optimization.
-   Augmentation must preserve clinical meaning and be applied after
    patient-level splitting.
-   ViTs convert image patches into tokens and use positional embeddings
    and self-attention.
-   Smaller ViT patches increase spatial granularity but can
    dramatically increase attention computation.
-   Ordinary 2D positional embeddings do not encode physical distances
    between MRI slices.
-   CLIP-style contrastive learning aligns image and text
    representations.
-   Medical text introduces challenges involving negation, uncertainty,
    laterality, incomplete reports, and noisy image-report
    correspondence.
-   Strong benchmark performance does not automatically establish
    clinical reliability or benefit.

## Final reflection

The most important shift in Day 8 was moving from understanding
individual architectures to evaluating whether an architecture preserves
the information required by a clinical task.

For future research discussions, consistently ask:

1.  What information does the model receive?
2.  What information might preprocessing or architecture discard?
3.  Does the training signal accurately represent clinical truth?
4.  Is the evaluation metric appropriate for the clinical objective?
5.  Does the model generalize across patients, hospitals, and
    acquisition protocols?
6.  Does the model provide clinically meaningful benefit, rather than
    only a strong benchmark score?

**Day 8 completed. Next: Day 9 --- Multimodal Learning.**
