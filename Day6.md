# Day 6 --- Transformer Mathematics, Positional Encoding, and Optimization

> **Programme:** 14-day preparation for medical machine learning
> research\
> **Day 6 focus:** Attention mathematics, Transformer architecture,
> positional information, optimization, and practical training\
> **Status:** Completed
>
> This document records the concepts studied, the questions asked, my
> answers, corrections, and topics to revisit. It intentionally excludes
> exercise grades.

------------------------------------------------------------------------

## 1. Learning objectives

By the end of Day 6, I worked through five knowledge blocks:

1.  **Attention mathematics:** expectation, variance, scaled dot-product
    attention, and softmax behaviour.
2.  **Multi-head attention:** projection matrices, tensor dimensions,
    concatenation, output projection, and parameter counts.
3.  **Positional encoding:** absolute and relative positions, RoPE,
    ALiBi, and irregular clinical timestamps.
4.  **Transformer optimization:** residual connections, Pre-LN versus
    Post-LN, gradient clipping, and numerical stability.
5.  **Modern training in PyTorch:** autograd, chain rule, AdamW,
    learning-rate warmup, cosine decay, and gradient accumulation.

The main application throughout the day was modelling wearable
physiological data and clinical time series.

------------------------------------------------------------------------

# 2. Knowledge 1 --- Attention mathematics

## Concepts studied

### 2.1 Scaled dot-product attention

For query (q) and key (k), the unscaled attention score is:

\[ s=q\^`\top `{=tex}k=`\sum`{=tex}\_{i=1}\^{d_k}q_i k_i \]

The standard scaled score is:

\[ s\_{`\text{scaled}`{=tex}}=`\frac{q^\top k}{\sqrt{d_k}}`{=tex} \]

The scaling factor helps control the variance of attention logits as the
key dimension increases. This reduces the risk that softmax becomes
excessively saturated and produces very concentrated attention
distributions.

### 2.2 Expected value and variance

Under the simplifying assumptions that query and key components are
independent, have zero mean, and have unit variance:

\[ E\[q_i k_i\]=0 \]

\[ `\operatorname{Var}`{=tex}(q_i k_i)=1 \]

For the sum of (d_k) independent terms:

\[ E\[q\^`\top `{=tex}k\]=0,`\qquad`{=tex}
`\operatorname{Var}`{=tex}(q\^`\top `{=tex}k)=d_k \]

After dividing by (`\sqrt{d_k}`{=tex}):

\[
`\operatorname{Var}`{=tex}`\left`{=tex}(`\frac{q^\top k}{\sqrt{d_k}}`{=tex}`\right`{=tex})=1
\]

These are assumptions used to explain the scaling intuition; real
learned query and key components need not satisfy them exactly.

### 2.3 Softmax saturation

If attention logits become very large in magnitude or differ greatly
from one another, softmax can become highly concentrated. This can make
the distribution close to one-hot and reduce useful gradient flow
through the softmax.

Large feature dimensions can contribute to large unscaled dot products,
which is one reason the attention score is divided by
(`\sqrt{d_k}`{=tex}).

## Questions and my answers

**Question 1: How would you express the expectation and variance of the
query-key score and the scaled score?**

My initial answer:

-   (E\[q_i k_i\] = `\frac{1}{i}`{=tex}`\sum`{=tex}(q_i k_i))
-   (E\[s\] = `\frac{1}{d_k}`{=tex}`\sum`{=tex}(s d_k))
-   I wrote variance expressions using averages of squared deviations.

**Correction / revision:**

The notation in my answer mixed up a sample average with a probability
expectation. For a random variable (X), expectation is written (E\[X\]).
Under the usual simplifying assumptions for attention:

\[ E\[q_i k_i\]=0,`\quad `{=tex}`\operatorname{Var}`{=tex}(q_i k_i)=1 \]

\[
E\[q\^`\top `{=tex}k\]=0,`\quad `{=tex}`\operatorname{Var}`{=tex}(q\^`\top `{=tex}k)=d_k
\]

\[
E\[s\_{`\text{scaled}`{=tex}}\]=0,`\quad `{=tex}`\operatorname{Var}`{=tex}(s\_{`\text{scaled}`{=tex}})=1
\]

**Question 2: Why divide attention logits by (`\sqrt{d_k}`{=tex})?**

My answer: Without scaling, logits can become too large and softmax may
become nearly uniform or too concentrated.

**Correction:** The key risk from large-magnitude logits is softmax
saturation (a very peaked distribution), not a uniform distribution.
Very small logits tend to produce a more uniform softmax. Scaling helps
keep the logit variance at a manageable level.

**Question 3: Why can large attention logits be a problem?**

My answer: Large logits can make softmax too concentrated. I connected
this to high-dimensional wearable sensor representations.

**Refinement:** High dimensionality can increase the variance of an
unscaled dot product. Scaling controls this dimension-related effect.
The issue is not simply that wearable signals have many sensor channels;
it is the dimension of the query/key representation and the distribution
of its components.

## Revise

-   Distinguish expectation of a random variable from an empirical
    sample average.
-   Derive why the variance of a sum of independent, zero-mean terms is
    the sum of their variances.
-   Explain why scaling is by (`\sqrt{d_k}`{=tex}), not (d_k).
-   Distinguish a uniform softmax (similar logits) from a saturated
    softmax (one or a few logits dominate).
-   State the assumptions behind the variance derivation.

------------------------------------------------------------------------

# 3. Knowledge 2 --- Multi-head attention: dimensions and parameters

## Architecture used in the exercises

-   Sequence length: (T=24)
-   Model dimension: (d\_{`\text{model}`{=tex}}=512)
-   Number of heads: (h=8)
-   Per-head key and value dimensions: (d_k=d_v=64)

Input:

\[ X`\in`{=tex}`\mathbb{R}`{=tex}\^{24`\times512`{=tex}} \]

For one head:

\[ W_i^Q,W_i^K,W_i^V`\in`{=tex}`\mathbb{R}`{=tex}^{512`\times64`{=tex}}
\]

Thus:

\[ Q,K,V`\in`{=tex}`\mathbb{R}`{=tex}\^{24`\times64`{=tex}} \]

The attention score matrix is:

\[ QK\^`\top`{=tex}:
(24`\times64`{=tex})(64`\times24`{=tex})=24`\times24`{=tex} \]

After softmax and multiplication by (V):

\[ (24`\times24`{=tex})(24`\times64`{=tex})=24`\times64`{=tex} \]

Each head therefore produces an output of shape (24`\times64`{=tex}).

Concatenating eight heads along the feature dimension gives:

\[ 24`\times`{=tex}(8`\cdot64`{=tex})=24`\times512`{=tex} \]

The output projection has shape:

\[ W^O`\in`{=tex}`\mathbb{R}`{=tex}^{512`\times512`{=tex}} \]

The final output remains:

\[ (24`\times512`{=tex})(512`\times512`{=tex})=24`\times512`{=tex} \]

This shape allows a residual addition with the original input (X).

## Parameter calculation

For one head, each Q, K, or V matrix has:

\[ 512`\cdot64`{=tex}=32,768 \]

Q, K, and V for one head:

\[ 3`\cdot32`{=tex},768=98,304 \]

Across eight heads:

\[ 8`\cdot98`{=tex},304=786,432 \]

Output projection:

\[ 512`\cdot512`{=tex}=262,144 \]

Total, ignoring biases:

\[ 786,432+262,144=`\boxed{1,048,576}`{=tex} \]

This is the standard full-projection parameter count for Q, K, V, and
output projection with model dimension 512. It excludes biases,
feed-forward layers, normalization parameters, and other Transformer
components. If the implementation uses biases for all four projections,
it adds (4`\cdot512`{=tex}=2,048) parameters.

## Questions and my answers

**Question 1: What is the shape of (W_Q) for an input of shape
(24`\times512`{=tex}), with 64 output features per time point?**

My first answer was (24`\times64`{=tex}).

Correction: That is the shape of (Q), not the weight matrix. The correct
weight shape is:

\[ W_Q:512`\times64`{=tex} \]

I then correctly answered (W_K=512`\times64`{=tex}).

**Question 2: What is the shape of the attention score matrix
(S=QK\^`\top`{=tex})?**

My answer: (24`\times24`{=tex}). Correct.

**Question 3: What is the shape after multiplying attention weights by
(V)?**

My answer: (24`\times64`{=tex}). Correct.

**Question 4: What is the concatenated shape for eight heads?**

My answer: (8`\times24`{=tex}`\times64`{=tex}).

Correction: This describes a possible stacked representation.
Concatenation along the feature dimension gives a matrix of shape:

\[ 24`\times512`{=tex} \]

**Question 5: What is the output shape after the output projection?**

My answer: (24`\times512`{=tex}). Correct.

**Question 6: How many parameters are in one Q projection matrix?**

My first answer multiplied (512`\times64`{=tex}) twice.

Correction:

\[ 512`\times64`{=tex}=32,768 \]

**Question 7: How many Q, K, and V parameters are in one head?**

My answer: 98,304. Correct.

**Question 8: How many Q, K, and V parameters are there across eight
heads?**

My answer: 786,432. Correct.

**Question 9: How many parameters are in the output projection?**

My setup was (512`\times512`{=tex}), which gives 262,144 parameters.

**Question 10: What is the total attention projection parameter count?**

My answer: 1,048,576. Correct.

**Question 11: Why might multi-head attention be useful for wearable
data?**

My answer: Different heads may learn different information, such as
heart rate, SpO₂, activity, and fine-grained or coarse temporal
patterns.

**Refinement:** This is a useful intuition, but heads are not guaranteed
to specialize in individual sensors. Each head learns its own
projections and attention patterns over the input representations. A
head might capture short-term dependencies or relationships between
physiological variables, but this must be investigated empirically
rather than assumed.

## Revise

-   Matrix multiplication dimensions: inner dimensions must match; outer
    dimensions determine the output.
-   Distinguish a matrix's shape from its number of elements/parameters.
-   Distinguish stacking heads from concatenating heads.
-   Practise calculating parameters with and without biases.
-   Understand that attention heads may learn complementary patterns,
    but do not necessarily correspond to individual modalities or
    sensors.
-   Remember that attention weights are not automatically explanations
    of clinical reasoning.

------------------------------------------------------------------------

# 4. Knowledge 3 --- Positional encoding

## 4.1 Why positional information is needed

Self-attention without positional information does not intrinsically
know the order of input tokens. Reordering the input reorders the
corresponding outputs, rather than giving the model an independent
notion of sequence order.

For time series, order matters. An increasing heart rate and a
decreasing heart rate can have different clinical meanings even if they
contain the same values.

## 4.2 Absolute versus relative position

-   **Absolute positional encoding:** represents a token's position,
    such as index 0, 1, or 2.
-   **Relative positional encoding:** represents the positional
    relationship or distance between tokens, such as (j-i).

An index is not automatically a real timestamp.

For irregularly sampled data, token positions can be misleading. For
example:

  Measurement     Token index Actual time
  ------------- ------------- -------------
  A                         0 08:00
  B                         1 08:05
  C                         2 16:00

Index distances are both 1, but the actual time intervals are 5 minutes
and 475 minutes.

## Questions and my answers

**Question 1: Why is positional information necessary for wearable
sequences?**

My answer: It preserves the order of the sequence and helps the model
learn time-dependent information.

Correct.

**Question 2: What is the difference between absolute and relative
positional encoding?**

My answer: Absolute encoding is the real or set time/order of a sample;
relative encoding is how far one sample is from another.

Refinement: Absolute encoding often represents an index, not necessarily
real clock time. Relative encoding represents a relationship between
positions. Real timestamps or elapsed times need to be provided or
encoded explicitly.

**Question 3: Which might be useful for wearable data?**

My initial answer: Absolute positional encoding, because it has a real
timestamp for each data point.

Correction: Standard absolute positional encoding does not necessarily
contain real timestamps. For irregular wearable data, time-aware
relative information based on actual elapsed time may be especially
useful. Absolute clock time or time since admission may also matter, for
example to capture circadian patterns or clinical context. A combination
may be appropriate.

## 4.3 Rotary Position Embedding (RoPE)

RoPE rotates query and key vectors according to their positions. For a
two-dimensional component pair:

\[ R(`\theta`{=tex})=
```{=tex}
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}
```
\]

The key property is:

\[
R(m`\theta`{=tex})\^`\top `{=tex}R(n`\theta`{=tex})=R((n-m)`\theta`{=tex})
\]

Therefore, the interaction between rotated queries and keys depends on
the relative position difference.

### RoPE exercise

Given (m=5) and (n=12):

-   My answer: (n-m) is the relative position.
-   More precise numerical answer: (12-5=7).
-   If both positions move forward by 10, the difference remains 7.
-   My explanation: This lets the Transformer learn from relative token
    positions and may benefit long sequences.

Refinement: RoPE encodes relative position in attention interactions,
but does not guarantee arbitrary-length extrapolation. Long-context
performance depends on training length, frequency choices, and any
context-extension method.

## 4.4 ALiBi

ALiBi adds a distance-dependent bias to attention logits instead of
rotating Q and K.

For a causal attention head, a simplified expression is:

\[
S\_{ij}=`\frac{q_i^\top k_j}{\sqrt{d_k}}`{=tex}-m_h(i-j),`\quad `{=tex}j`\leq `{=tex}i
\]

where (m_h) is a head-specific slope.

### Irregular-time ALiBi exercise

Scenario: Measurements at 08:00, 08:05, and 16:00 are assigned token
positions 0, 1, and 2. Standard ALiBi uses token-index distance.

My answer: The Transformer does not know the actual time distance
between measurements, and clinical gaps can be useful information. I
proposed adding an indicator for distance from the beginning.

Refinement: Elapsed time since the beginning is useful absolute temporal
context, but pairwise elapsed time is also important. One possible
time-aware bias is:

\[ B\_{ij}\^{(h)}=-m_h g(\|t_i-t_j\|) \]

where (t_i,t_j) are actual timestamps and (g) could be a suitable
transformation such as (`\log`{=tex}(1+`\Delta `{=tex}t)). This is a
design proposal, not a universal best solution; it should be compared
experimentally.

## Revise

-   Absolute index versus actual timestamp.
-   Relative index distance versus actual elapsed time.
-   RoPE's rotation property and why the attention interaction depends
    on relative position.
-   ALiBi's additive bias and its head-specific slopes.
-   Why standard NLP positional methods may need adaptation for
    irregular EHR and wearable measurements.
-   Consider both time since observation start and pairwise elapsed
    time.
-   Compare positional strategies with controlled experiments and
    ablations.

------------------------------------------------------------------------

# 5. Knowledge 4 --- Transformer optimization

## 5.1 Residual connections and gradient flow

A residual block is:

\[ y=x+F(x) \]

Its derivative in the scalar case is:

\[ `\frac{dy}{dx}`{=tex}=1+`\frac{dF(x)}{dx}`{=tex} \]

For vector-valued functions:

\[ J_y=I+J_F \]

The identity term gives gradients a direct path through the block. This
can make deep networks easier to optimize, although residual connections
do not guarantee that gradients can never vanish or explode.

### Exercise: Residual derivative

Given:

\[ `\frac{dF(x)}{dx}`{=tex}=0.001 \]

My answers:

1.  With residual connection: (1.001).
2.  Without residual connection: (0.001).
3.  My explanation: Deep Transformers can suffer vanishing gradients,
    and the identity term gives a direct gradient path.

All three ideas were correct. The important qualification is that real
networks involve Jacobian products, so the identity path helps but does
not guarantee stable gradients.

## 5.2 Pre-LN versus Post-LN

Pre-LN:

\[ y=x+F(LN(x)) \]

Its derivative is:

\[ `\frac{dy}{dx}`{=tex}=I+J_FJ\_{LN} \]

Post-LN:

\[ y=LN(x+F(x)) \]

Its derivative is:

\[ `\frac{dy}{dx}`{=tex}=J\_{LN}(I+J_F) \]

Pre-LN has a more direct identity gradient path that bypasses both the
transformation and normalization. This often makes optimization easier
in deep Transformers, though architecture and training details matter.

### Exercise: Pre-LN versus Post-LN

My answers:

1.  Pre-LN provides the more direct identity path.
2.  It can be easier to train because the residual path bypasses the
    transformation and normalization.
3.  A 48-layer medical Transformer may face gradient vanishing or
    explosion and needs stable optimization.

Refinement: A large parameter count alone does not cause gradient
instability. Repeated transformations and Jacobian products are the more
direct mathematical explanation.

## 5.3 Gradient clipping

Global norm clipping limits the gradient magnitude:

\[
g'=g`\cdot`{=tex}`\min`{=tex}`\left`{=tex}(1,`\frac{c}{\|g\|_2}`{=tex}`\right`{=tex})
\]

It scales a gradient vector down if its norm exceeds the threshold,
while preserving its direction.

### Exercise 1

Given (g=\[6,8\]) and (c=2):

\[
\|g\|\_2=10,`\quad `{=tex}`\alpha=0.2`{=tex},`\quad `{=tex}g'=\[1.2,1.6\]
\]

### Exercise 2

Given (g=\[3,4\]) and (c=1):

My answers: - Original norm: 5. - I initially answered the scaling
factor as 0.4; correction: (1/5=0.2). - I initially gave a scalar rather
than a vector for the clipped gradient; correction: (g'=\[0.6,0.8\]). -
My explanation: Clipping prevents large gradients from causing
excessively large updates.

Important distinction: clipping controls excessively large gradients
after they are computed. It does not solve vanishing gradients.

## 5.4 Numerically stable softmax

Softmax is:

\[ softmax(z_i)=`\frac{e^{z_i}}{\sum_j e^{z_j}}`{=tex} \]

Large logits can cause exponential overflow. Since softmax is invariant
to subtracting the same constant from every logit, use:

\[ z'=z-`\max`{=tex}(z) \]

This keeps the largest shifted logit at zero and all others
non-positive.

### Exercises

For (z=\[1000,1001,1002\]):

-   My initial maximum answer was 1000; correction: the maximum is 1002.
-   My initial shifted vector was (\[0,1,2\]); correction:
    (\[-2,-1,0\]).

For (z=\[15,20,18\]):

-   My answer was (\[-5,0,-2\]), which is correct.

## Revise

-   Derive residual-block gradients and understand the identity path.
-   Compare the computational gradient paths in Pre-LN and Post-LN.
-   Practise global norm clipping, including the scaling factor and the
    resulting vector.
-   Distinguish gradient clipping from gradient normalization.
-   Understand softmax shift invariance and why subtracting the maximum
    prevents overflow.
-   Distinguish vanishing gradients from exploding gradients.

------------------------------------------------------------------------

# 6. Knowledge 5 --- Modern optimization and PyTorch

## 6.1 Autograd and the chain rule

A simple model:

\[ `\hat `{=tex}y=wx \]

\[ L=(wx-y)\^2 \]

The gradient is:

\[ `\frac{\partial L}{\partial w}`{=tex}=2(wx-y)x \]

PyTorch tracks operations when parameters have `requires_grad=True`.
Calling `loss.backward()` computes gradients using automatic
differentiation.

Example:

``` python
import torch

w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
y = torch.tensor(10.0)

prediction = w * x
loss = (prediction - y) ** 2

loss.backward()
print(w.grad)
```

### Exercise: Forward and backward pass

Given (w=2), (x=3), and (y=10):

My answers:

1.  Prediction: 6 --- correct.
2.  Loss: 16 --- correct.
3.  Gradient: I answered -12; correction: (-24).

Derivation:

\[ `\hat `{=tex}y=2`\cdot3`{=tex}=6 \]

\[ L=(6-10)\^2=16 \]

\[ `\frac{\partial L}{\partial w}`{=tex}=2(wx-y)x =2(6-10)(3)=-24 \]

With learning rate (`\eta=0.1`{=tex}), one gradient-descent update would
be:

\[ w\_{`\text{new}`{=tex}}=2-0.1(-24)=4.4 \]

## 6.2 Adam and AdamW

Adam maintains exponential moving averages of the gradient and squared
gradient:

\[ m_t=`\beta`{=tex}*1m*{t-1}+(1-`\beta`{=tex}\_1)g_t \]

\[ v_t=`\beta`{=tex}*2v*{t-1}+(1-`\beta`{=tex}\_2)g_t\^2 \]

It uses bias-corrected estimates to adapt parameter updates.

AdamW decouples weight decay from the adaptive gradient update. This
makes weight decay separate from Adam's moment calculations.

### Exercise: AdamW

My answers:

1.  Weight decay discourages the model from relying on excessively large
    weights.
2.  Adam with coupled L2 regularization incorporates the penalty into
    the gradient; AdamW applies weight decay separately.
3.  AdamW allows regularization strength to be controlled independently
    of the adaptive gradient update.

Refinement: Weight decay can help control overfitting, but does not
guarantee external generalization. External validation remains
essential.

PyTorch example:

``` python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)
```

## 6.3 Learning-rate warmup

A linear warmup schedule is:

\[
`\eta`{=tex}*t=`\eta`{=tex}*{`\max`{=tex}}`\frac{t}{T_{\text{warmup}}}`{=tex}
\]

It gradually increases the learning rate at the beginning of training.

### Exercise

Given (`\eta`{=tex}\_{`\max`{=tex}}=0.0001) and 1,000 warmup steps:

-   At step 500, my answer was 0.00005 --- correct.
-   At step 800, my answer was 0.00008 --- correct.
-   My explanation: Early parameters and representations are not yet
    well organized; large updates can destabilize training and gradients
    can fluctuate.

Refinement: The main concern is excessively large early parameter
updates, not necessarily large parameter values.

## 6.4 Cosine learning-rate decay

A cosine schedule is:

\[ `\eta`{=tex}*t=`\eta`{=tex}*{`\min`{=tex}}+
`\frac{1}{2}`{=tex}(`\eta`{=tex}*{`\max`{=tex}}-`\eta`{=tex}*{`\min`{=tex}})
(1+`\cos`{=tex}(`\pi `{=tex}p)) \]

where (p) is the fraction of the decay schedule completed.

### Exercise

Given (`\eta`{=tex}*{`\max`{=tex}}=0.0001),
(`\eta`{=tex}*{`\min`{=tex}}=0), and 50% decay progress:

-   (p=0.5) --- correct.
-   Learning rate: 0.00005 --- correct.
-   My explanation: Smaller learning-rate updates help refine parameters
    near the end of training.

Refinement: Cosine decay reduces the learning rate, not necessarily the
gradient magnitude. The actual parameter update also depends on the
gradient and optimizer.

## 6.5 Gradient accumulation

Gradient accumulation splits a large effective batch into smaller
mini-batches, accumulates gradients, and performs one optimizer update
after the accumulation steps.

Example:

-   Effective batch: 128 patients.
-   Mini-batch size: 32.
-   Accumulation steps: 4.
-   Optimizer updates after processing all 128 patients: 1.

Conceptually, for equal-sized mini-batches and appropriately averaged
losses:

\[ g\_{`\text{effective}`{=tex}}=`\frac{g_1+g_2+g_3+g_4}{4}`{=tex} \]

PyTorch example:

``` python
optimizer.zero_grad()

for batch in mini_batches:
    loss = model_loss(batch) / 4
    loss.backward()

optimizer.step()
```

### Exercise: Gradient accumulation

My answers:

1.  One optimizer update --- correct.
2.  Do not call `optimizer.step()` after each mini-batch because
    gradients should accumulate before the parameter update.
3.  It reduces the memory required for each mini-batch when the GPU
    cannot fit the full batch.

Refinement: Gradient accumulation reduces peak activation memory per
forward/backward pass, but model parameters, optimizer states, and
accumulated gradients still occupy memory. It approximates a larger
batch gradient, but can differ when batch-dependent operations or
stochastic layers are involved.

## Revise

-   Apply the chain rule carefully; do not omit factors.
-   Practise calculating gradients and then applying a gradient-descent
    update.
-   Explain Adam's first and second moments and bias correction.
-   Distinguish Adam with coupled L2 regularization from AdamW.
-   Calculate linear warmup and cosine decay values.
-   Distinguish learning-rate reduction from gradient-magnitude
    reduction.
-   Understand when gradient accumulation approximates a large batch and
    what memory it does and does not save.
-   Remember that PyTorch gradients accumulate in `.grad`; clear them
    before a new accumulation cycle with `optimizer.zero_grad()`.

------------------------------------------------------------------------

# 7. Consolidated revision checklist

Use this checklist for short review sessions over the next few days.

## Attention and tensor mathematics

-   [ ] Derive the expected value and variance of (q\^`\top `{=tex}k)
    under stated assumptions.
-   [ ] Explain why scaled dot-product attention divides by
    (`\sqrt{d_k}`{=tex}).
-   [ ] Explain softmax saturation versus a nearly uniform distribution.
-   [ ] Calculate Q, K, V, attention-score, and attention-output shapes.
-   [ ] Distinguish concatenation from stacking.
-   [ ] Calculate projection parameter counts with and without biases.

## Positional encoding and temporal data

-   [ ] Distinguish absolute indices, relative indices, timestamps, and
    elapsed time.
-   [ ] Explain the relative-position property of RoPE.
-   [ ] Explain how ALiBi adds a distance-dependent bias.
-   [ ] Identify why index-based positional methods can misrepresent
    irregularly sampled clinical data.
-   [ ] Propose and evaluate time-aware positional features or biases.

## Optimization

-   [ ] Derive the gradient of a residual block.
-   [ ] Explain the gradient-path difference between Pre-LN and Post-LN.
-   [ ] Calculate global norm clipping by hand.
-   [ ] Explain why clipping does not solve vanishing gradients.
-   [ ] Stabilize softmax by subtracting the maximum logit.
-   [ ] Calculate a simple model's loss and gradient using the chain
    rule.
-   [ ] Explain Adam's moments and AdamW's decoupled weight decay.
-   [ ] Calculate warmup and cosine-decay learning rates.
-   [ ] Explain gradient accumulation and its limitations.

------------------------------------------------------------------------

# 8. Recommended learning materials

These are selected to reinforce the concepts studied today. Prioritize
the original papers and official documentation over watching many
general tutorials.

## Foundational Transformer and attention papers

1.  **Attention Is All You Need** --- Vaswani et al. (2017)\
    The original Transformer paper. Focus on scaled dot-product
    attention, multi-head attention, positional encoding, residual
    connections, and LayerNorm.\
    https://arxiv.org/abs/1706.03762

2.  **RoFormer: Enhanced Transformer with Rotary Position Embedding**
    (2021)\
    Useful for understanding the mathematical motivation and
    relative-position property of RoPE.\
    https://arxiv.org/abs/2104.09864

3.  **Train Short, Test Long: Attention with Linear Biases Enables Input
    Length Extrapolation** (ALiBi, 2021)\
    Read the method section to understand how distance-dependent biases
    are added to attention scores.\
    https://arxiv.org/abs/2108.12409

## Optimization and normalization

4.  **Adam: A Method for Stochastic Optimization** --- Kingma and Ba
    (2014)\
    Focus on first and second moments, bias correction, and the update
    rule.\
    https://arxiv.org/abs/1412.6980

5.  **Decoupled Weight Decay Regularization** --- Loshchilov and Hutter
    (2017/2019)\
    The key paper behind AdamW. Focus on why L2 regularization and
    weight decay are not equivalent for adaptive optimizers.\
    https://arxiv.org/abs/1711.05101

6.  **Layer Normalization** --- Ba, Kiros, and Hinton (2016)\
    A useful reference for understanding normalization in sequence
    models.\
    https://arxiv.org/abs/1607.06450

## PyTorch practical resources

7.  **PyTorch Autograd documentation**\
    Practise computational graphs, `requires_grad`, `backward()`, and
    gradient inspection.\
    https://docs.pytorch.org/docs/stable/autograd.html

8.  **PyTorch Optimizer documentation**\
    Review optimizer updates, Adam, AdamW, and parameter groups.\
    https://docs.pytorch.org/docs/stable/optim.html

9.  **PyTorch learning-rate scheduler documentation**\
    Review warmup implementations, cosine schedules, and scheduler
    stepping.\
    https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

## Suggested practical exercises

-   Implement a single attention head in PyTorch and print the shapes of
    Q, K, V, attention scores, and outputs.
-   Implement multi-head attention for (d\_{`\text{model}`{=tex}}=512),
    (h=8), and verify the parameter count.
-   Compare Pre-LN and Post-LN Transformer blocks in a small experiment.
-   Implement global gradient norm clipping and inspect gradient norms
    before and after clipping.
-   Compare ordinary softmax with max-shifted softmax on large logits.
-   Train a small model using Adam and AdamW, keeping the data split and
    other hyperparameters fixed.
-   Plot a linear warmup plus cosine decay schedule.
-   Implement gradient accumulation and compare it with a single larger
    batch, noting any differences caused by batch-dependent layers or
    stochasticity.

------------------------------------------------------------------------

# 9. Medical ML research connections

The most important lesson from Day 6 is that Transformer design choices
should be evaluated in the context of the data-generating process.

For wearable and EHR data:

-   **Irregular sampling:** token index is not the same as elapsed
    clinical time.
-   **Missingness:** measurement absence may itself carry information
    and may be informative rather than random.
-   **Longitudinal structure:** temporal order and time gaps can affect
    clinical interpretation.
-   **Optimization:** unstable training can make model comparisons
    unreliable.
-   **Generalization:** a stable training loss does not establish
    external-hospital generalization.
-   **Interpretability:** attention patterns should not automatically be
    treated as faithful clinical explanations.
-   **Reproducibility:** report optimizer, learning-rate schedule,
    weight decay, clipping threshold, batch size, accumulation steps,
    seeds, and stopping criteria.

For a research paper, I should ask not only whether a Transformer
achieves a high AUROC, but also whether its temporal representation is
appropriate, its training is stable, its comparisons are fair, and its
performance generalizes to new patients and hospitals.

------------------------------------------------------------------------

# 10. Preparation for Day 7

**Next topic: Natural Language Processing (NLP).**

I will carry forward the Transformer foundations from Day 6 and apply
them to language.

Suggested Day 7 topics:

1.  Text preprocessing and tokenization.
2.  Subword tokenization, BPE, and WordPiece.
3.  Word and contextual embeddings.
4.  Language modelling and next-token prediction.
5.  NLP evaluation, perplexity, and medical-domain language models.

The aim is to connect Transformer mathematics to the way modern language
models represent and process clinical text.

------------------------------------------------------------------------

## Personal learning reflection

My conceptual understanding of Transformer architecture and optimization
is developing. I can now work through many tensor-shape and
parameter-count calculations, explain the purpose of multi-head
attention, and reason about positional encoding for irregular medical
time series.

My main area for improvement remains mathematical precision. I should
practise the chain rule, variance derivations, gradient clipping, and
optimizer update equations until I can derive them without relying on
memorized answers.

**Study principle for the next days:** learn the concept, derive the
equation, implement a small example, and then explain why it matters for
medical ML research.
