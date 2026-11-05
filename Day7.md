# Day7 --- NLP and LLM Foundations

**Programme:** 14-day medical machine learning learning programme\
**Day:** 7 of 14\
**Theme:** Natural Language Processing, language modelling, and medical
LLM evaluation

## Day 7 overview

The five knowledge blocks were:

1.  Text preprocessing and tokenization.
2.  Subword tokenization: BPE, WordPiece, and Unigram.
3.  Word embeddings: Word2Vec, GloVe, and contextual representations
    such as BERT.
4.  Language modelling: autoregressive prediction, masked language
    modelling, cross-entropy, perplexity, teacher forcing, and exposure
    bias.
5.  Medical LLM evaluation: BLEU, ROUGE-L, BERTScore, clinical
    factuality, hallucinations, and domain adaptation.

**Central lesson:** Fluent language generation and high text similarity
do not prove clinical correctness. Medical NLP systems must be evaluated
for factuality, preservation of clinical meaning, and generalization.

------------------------------------------------------------------------

# Knowledge 1 --- Text preprocessing and tokenization

## Key knowledge

A typical NLP pipeline is: raw clinical text →
normalization/preprocessing → tokenization → token IDs → embedding
lookup → Transformer processing.

Tokenization divides text into words, subwords, characters, punctuation,
or special tokens. Token IDs are categorical identifiers; they do not
inherently contain semantic meaning. An embedding layer maps IDs to
learned vectors.

Common special tokens include `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`,
`[MASK]`, `[BOS]`, and `[EOS]`; conventions vary by model. Padding makes
variable-length sequences the same length within a batch. An attention
mask identifies valid tokens and padding positions. A padding mask
differs from a causal mask, which prevents access to future positions
during autoregressive modelling.

Clinical preprocessing must preserve negation, uncertainty, temporality,
experiencer, abbreviations, medication names, dosages, units, and
measurements. For example, "has pneumonia" and "does not have pneumonia"
must remain distinguishable.

## Exercise 1 --- Clinical preprocessing

**Scenario:** A pipeline lowercases text, removes punctuation, removes
stop words including `not`, `no`, and `without`, tokenizes, maps tokens
to IDs, and pads sequences.

### 1. What is tokenization, and why convert text into token IDs?

**My answer:** \> Tokenization is turning texts to separated words,
subwords, or characters for easier processing for models. Because NLP
models only know the meaning when the text is converted to token IDs.

**Corrected answer:** Tokenization divides text into discrete units and
maps them to vocabulary IDs. IDs are categorical identifiers, not
semantic representations. An embedding layer maps IDs to learned
vectors, which are processed by the model.

### 2. Which preprocessing step is dangerous, and why?

**My answer:** \> Remove the stop words such as no and not. Because it
may change the meaning of text completely.

**Correct answer:** Removing negation words such as `no`, `not`, and
`without` can reverse clinical meaning. Negation must be preserved and
explicitly evaluated.

### 3. Why do we need padding and attention masks?

**My answer:** \> Because to process token groups within the same patch,
their length must be the same, so we padding the tokens. Attention mask
will tell the model which tokens are paddings.

**Corrected answer:** Sequences in a **batch** are commonly padded to a
common length so they can be processed together. An attention mask
identifies valid tokens and padding positions. "Batch," not "patch," is
the correct term here.

### 4. What safeguards would you implement?

**My answer:** \> I will pay more attention to stop words, medical
terminologies and other words containing important medical information,
use techniques to remain them as well as their order.

**Corrected answer:** Use conservative normalization; preserve negation,
abbreviations, medical terms, numbers, units, dosage, temporality,
uncertainty, and experiencer information. Document the pipeline and test
that preprocessing does not alter clinical meaning.

------------------------------------------------------------------------

# Knowledge 2 --- Subword tokenization

## Key knowledge

-   **BPE (Byte Pair Encoding):** Starts with small units and repeatedly
    merges frequent adjacent units.
-   **WordPiece:** Builds a subword vocabulary through merging, using a
    likelihood-based merge criterion rather than simply selecting the
    most frequent adjacent pair. It is associated with BERT.
-   **Unigram:** Starts with a relatively large vocabulary and
    progressively removes units to optimize a probabilistic tokenization
    model. It can assign probabilities to different valid segmentations.

  Method      Main principle
  ----------- ----------------------------------------------------
  BPE         Repeatedly merges frequent adjacent units
  WordPiece   Selects merges using a likelihood-based criterion
  Unigram     Learns a probabilistic vocabulary by pruning units

Subword tokenization helps represent rare and unseen words and reduces
the need for a separate vocabulary entry for every word. A subword is
not necessarily a meaningful medical morpheme. Fewer tokens can reduce
computational cost, but fewer tokens do not automatically mean better
representations. Vocabulary coverage, clinical terminology handling,
downstream performance, and compatibility with pretrained embeddings
matter.

A fair tokenizer comparison should control data splits, preprocessing,
architecture, embedding dimensions, training budget, and evaluation.
Assess token fertility, sequence length, vocabulary coverage,
truncation, downstream performance, and external generalization. Learn
tokenizer vocabulary from training text only. A pretrained model's
tokenizer cannot simply be replaced without addressing compatibility
with its embedding matrix and pretrained weights.

## Exercise 1 --- General-domain versus medical-domain tokenizer

### 1. Why is subword tokenization preferable to word-level tokenization?

**My answer:** \> Because lots of medical terminologies are complex and
combined by different subword groups, so it's more clear and convenient
for model to learn their meaning if they are tokenized by subgroup.

**Corrected answer:** Subword tokenization reduces vocabulary
requirements and lets rare or unseen medical terms be represented using
reusable units. It does not automatically make a term easier to
understand; semantic knowledge must be learned during training.

### 2. What is the difference between BPE and WordPiece?

**My answer:** \> BPE select the most frequent character merge as a
subword, WordPiece choose merges by likelihood criterion.

**Correct answer:** BPE repeatedly merges frequent adjacent units.
WordPiece selects merges using a likelihood-based criterion.

### 3. Does producing fewer tokens automatically mean the medical tokenizer is better?

**My answer:** \> No, because fewer tokens may means less detailed
information for models if there are many combined medical terminologies.
Less doesn't mean better, nor more.

**Correct answer:** Correct. Fewer tokens may improve efficiency, but
token count alone does not establish quality. Vocabulary coverage,
clinical terminology handling, and downstream performance must also be
evaluated.

### 4. How would you compare the tokenizers fairly?

**My answer:** \> I will use the same text input, same NLP models, and
same preprocessing way to train the model and deal with the input. And
use same evaluation methods to see which tokenizer works the best.

**Corrected answer:** Control the training corpus, data splits,
preprocessing, architecture, embedding dimension, training budget, and
evaluation data as much as possible. Train compatible models separately
where needed. Compare token fertility, sequence length, vocabulary
coverage, truncation, downstream performance, and external
generalization. Learn tokenizer vocabularies from training data only.

## Exercise 2.2 --- Unigram and subword regularization

### 1. Why might multiple segmentations help?

**My answer:** \> Multiple segmentations can generate different version
of subword tokens, which enrich the options for models to learn as much
information as they can. They can pick the most related token.

**Corrected answer:** Subword regularization exposes the model to
different valid segmentations of the same word. This acts as data
augmentation and may reduce dependence on one fixed segmentation. The
tokenizer samples segmentations according to its learned probability
distribution; the language model is not necessarily choosing the most
semantically related token.

### 2. What is a potential disadvantage?

**My answer:** \> One potential disadvantage is the model may won't
learn very consistent representation of words, which may affect the
performance because the model was exposed to many segmentations for each
word.

**Corrected answer:** Excessive segmentation variation can introduce
training noise and make stable representation learning harder,
especially for rare clinical terms. More fragmented sequences can also
increase computational cost. Controlled variation may be beneficial, so
its effect should be evaluated rather than assumed.

------------------------------------------------------------------------

# Knowledge 3 --- Word embeddings and contextual representations

## Key knowledge

### Word2Vec

Word2Vec learns embeddings through local context-prediction tasks: -
**CBOW:** Predicts a target word from surrounding context. -
**Skip-gram:** Predicts surrounding context words given a target word.

A simplified Skip-gram objective is:

\[ `\mathcal{L}`{=tex} =
-`\sum`{=tex}*t`\sum`{=tex}*{`\substack{-c\leq j\leq c\\j\neq0}`{=tex}}`\log `{=tex}P(w\_{t+j}`\mid `{=tex}w_t)
\]

Skip-gram can generate multiple target-context training pairs from each
occurrence of a word, providing several learning signals for rare terms.
CBOW is generally faster because it aggregates context to predict a
target.

### GloVe

GloVe (Global Vectors for Word Representation) learns embeddings from
global word co-occurrence statistics:

\[
J=`\sum`{=tex}*{i,j}f(X*{ij})`\left`{=tex}(w_i^T`\tilde `{=tex}w_j+b_i+`\tilde `{=tex}b_j-`\log `{=tex}X\_{ij}`\right`{=tex})^2
\]

Here, (X\_{ij}) is a co-occurrence count; (w_i) and (`\tilde `{=tex}w_j)
are word and context vectors; and (b_i,`\tilde `{=tex}b_j) are bias
terms.

**Important:** Both Word2Vec and GloVe produce static embeddings.
GloVe's use of global statistics does not make its embeddings
contextual.

### BERT and contextual embeddings

BERT uses bidirectional self-attention to produce context-dependent
token representations. The same word can have different representations
in different sentences.

\[ h_i\^{(0)}=E(token_i)+P(i) \] \[
h_i^{(l+1)}=`\operatorname{TransformerLayer}`{=tex}(h_i^{(l)}) \]

BERT's original input representation combines token, position, and
segment embeddings:

\[
E\_{`\text{input}`{=tex}}=E\_{`\text{token}`{=tex}}+E\_{`\text{position}`{=tex}}+E\_{`\text{segment}`{=tex}}
\]

Contextual representations can help with ambiguity, negation,
temporality, and experiencer information, but they do not guarantee
correct clinical interpretation.

  -----------------------------------------------------------------------------------
  Property            Word2Vec              GloVe                 BERT
  ------------------- --------------------- --------------------- -------------------
  Learning signal     Local context         Global co-occurrence  Masked language
                      prediction                                  modelling and
                                                                  contextual
                                                                  attention

  Static/contextual   Static                Static                Contextual

  Same word in        Same lookup vector    Same lookup vector    Context-dependent
  different sentences                                             

  Typical use         Word                  Word                  Contextual NLP
                      features/similarity   features/similarity   tasks
  -----------------------------------------------------------------------------------

## Exercise 1 --- Word2Vec versus BERT

**Scenario:** Compare "Patient was discharged from hospital" with
"Patient has purulent discharge from the wound."

### 1. Why would Word2Vec have difficulty distinguishing the meanings of "discharge"?

**My answer:** \> because the word2vec learn the representation of words
during the process of training, it's not dynamic. it will only produce
one same vector to represent a same word even if it has different
meanings in a context through the whole training set.

**Correct answer:** Word2Vec learns a fixed lookup embedding for each
vocabulary word. Context is used during training, but the resulting
embedding is static when used. `Discharge` receives the same lookup
vector in both sentences.

### 2. How does BERT produce different contextual representations?

**My answer:** \> the bert produce representations according to the word
or tokens adjacent to the target word. so it can find out the same word
contain different meanings.

**Corrected answer:** BERT's self-attention allows each token to
incorporate information from other tokens throughout the input sequence,
not only adjacent words.

### 3. Why are contextual embeddings important for medical NLP?

**My answer:** \> because medical nlp research vocabulary is diverse and
complex, same word may have many different meanings in one text, then
the different vector representation for the words is very important to
better understand the context and improve performance.

**Correct answer:** Clinical language contains semantic ambiguity,
abbreviations, negation, temporality, and experiencer information.
Contextual representations can help distinguish these situations, but
still require clinical evaluation.

## Exercise 2 --- CBOW versus Skip-gram

**Sentence:** "Patient developed acute kidney injury after surgery."

### 1. What does CBOW use to predict `injury`?

**My answer:** \> CBOW use the other words adjacent to the target word
to predict it.

**Correct answer:** CBOW uses surrounding context words to predict the
target word `injury`.

### 2. What does Skip-gram predict from `injury`?

**My answer:** \> skip gram try to predict the words around 'injury'.

**Correct answer:** Skip-gram uses `injury` as the target and predicts
surrounding context words such as `acute`, `kidney`, and `after`,
depending on the context window.

### 3. Why might Skip-gram help with rare medical terms?

**My answer:** \> skip gram can predict words around these rare medical
terms, so it may provide more example sentence for each term, which
enrich the training data. but CBOW can only predict the one target word,
which won't be very helpful.

**Corrected answer:** Skip-gram does not generate more sentences. Each
occurrence can generate multiple target-context training pairs,
providing several learning signals. This can help rare words. CBOW
generally aggregates context to predict one target and is often faster.
The relative performance is empirical, not guaranteed.

## Exercise 3 --- Word2Vec versus GloVe

### 1. What is the fundamental difference?

**My answer:** \> word2vec learns word embeddings from the adjacent
words, and only use same embedding for same word even if they have
different meanings. Glove learns the embeddings from the global training
set, and can represent different meanings with different embeddings for
one word.

**Correction:** The first part is correct, but the final claim is
incorrect. Both Word2Vec and GloVe produce static embeddings. Word2Vec
learns through context prediction; GloVe learns from global
co-occurrence statistics. Neither dynamically assigns different
contextual embeddings to the same word occurrence.

### 2. Why might GloVe be useful in clinical corpora?

**My answer:** \> yes, it can be helpful because GloVe can use the co
occurrence patterns to produce embeddings for words.

**Correct answer:** GloVe can learn statistical relationships from
global co-occurrence patterns in clinical text. Its usefulness depends
on corpus quality and representativeness.

### 3. Can GloVe distinguish "Patient has pneumonia" from "Patient does not have pneumonia"?

**My answer:** \> not really, because it can also struggle with negative
meaning and sentences with stop words.

**Correct answer:** Not reliably through its static word vectors. The
embedding for `pneumonia` is the same in both sentences. Standard GloVe
does not dynamically construct a sentence-contextual representation that
captures the effect of negation.

## Exercise 4 --- Static versus contextual embeddings

### 1. What happens to `discharge` in Word2Vec and GloVe?

**My answer:** \> in word2vec and glove, the word will have same
embedding because these methods represent the word in one training test
statically. not dynamic

**Correct answer:** The same vocabulary word has one fixed lookup
embedding within each model, regardless of sentence context.

### 2. What happens in BERT?

**My answer:** \> different, self attention can let the model learn the
embeddings by the context of the word. so it can diffrenciate the
different meaning of same word.

**Correct answer:** BERT produces context-dependent representations
through self-attention. The representation of `discharge` can differ
between the hospital-discharge sentence and the wound-discharge
sentence.

### 3. Why might BERT be a more suitable starting point for clinical information extraction?

**My answer:** \> because bert can produce representations by the words'
context, it's dynamic. also, bert can integrate negation to the word,
make same word different representations. word2vec and glove can only
represent same word using same embeddings and struggle with negations.

**Correct answer:** BERT's contextual representations can incorporate
information about negation, ambiguity, and surrounding clinical context.
It is a useful starting point for clinical extraction, but it does not
guarantee correct interpretation; clinical phenomena must be evaluated
explicitly.

------------------------------------------------------------------------

# Knowledge 4 --- Language modelling

## Key knowledge

### Autoregressive language modelling

GPT-style models predict the next token from preceding tokens:

\[ P(w_t`\mid `{=tex}w_1,`\ldots`{=tex},w\_{t-1}) \]

The joint probability of a sequence is factorized using the chain rule:

\[
P(w_1,`\ldots`{=tex},w_n)=`\prod`{=tex}*{t=1}\^{n}P(w_t`\mid `{=tex}w*{\<t})
\]

Causal masking prevents a position from attending to future tokens.

### Masked language modelling

BERT-style models hide selected tokens and predict them using
surrounding context. They can use both left and right context.

### Cross-entropy / negative log-likelihood

For a correct target token with predicted probability (p):

\[ L=-`\ln`{=tex}(p) \]

Higher probability assigned to the correct token produces lower loss.

### Perplexity

When average loss uses natural logarithms:

\[ PPL=`\exp`{=tex}(L\_{`\text{average}`{=tex}}) \]

Lower perplexity generally means the model assigns higher probability to
the observed sequence under the evaluation setup. It is not a direct
measure of clinical understanding. Comparisons across different
tokenizers require care because token-level losses and sequence lengths
can differ.

### Teacher forcing and exposure bias

During autoregressive training, teacher forcing supplies correct
preceding tokens. At inference, the model conditions on its own
generated tokens. An early incorrect token can influence later
predictions. This training--inference context mismatch is called
exposure bias.

A standard BERT masked-language model is not naturally optimized for
free-form autoregressive generation. An autoregressive or
encoder-decoder model is a more natural starting point for
discharge-summary generation.

## Exercise 1 --- Autoregressive versus masked language modelling

### 1. How does Model A learn next-token prediction?

**My answer:** \> model A use causal attention to predict the next token
based on possibilities using all previous tokens.

**Correct answer:** An autoregressive model estimates the probability of
the next token conditioned on preceding tokens. Causal masking prevents
access to future positions.

### 2. How does Model B predict masked tokens?

**My answer:** \> model b predicts masked tokens using tokens left and
right to the target token, so it's bidirection attention.

**Correct answer:** A masked language model predicts selected hidden
tokens using both left and right context.

### 3. What does the sequence probability equation represent?

**My answer:** \> it represent the possibility of the nth token is
calculated by all former token conditions together

**Corrected answer:** The full equation represents the joint probability
of the entire sequence, factorized into conditional probabilities for
every token position. The final factor is the probability of the nth
token given all preceding tokens.

### 4. Which model would you consider for discharge-summary generation?

**My answer:** \> i would use model B, because it can extract the useful
information using bidirectional attention, so it can perform better then
model A which just predicting the next token.

**Correction:** For generating a coherent discharge summary, an
autoregressive or encoder-decoder generative model is a more natural
starting point. BERT is useful for extracting diagnoses, medications,
and concepts, but a standard masked-language model is not directly
optimized for free-form generation. Evaluation must check factuality,
completeness, negation, medication accuracy, and unsupported statements.

## Exercise 2 --- Cross-entropy calculation

Given Model A probability 0.8 and Model B probability 0.2 for the
correct token:

### 1. Model A loss

**My answer:** (L=0.2231)

**Correct answer:**

\[ -`\ln`{=tex}(0.8)`\approx0.2231`{=tex} \]

### 2. Model B loss

**My answer:** (L=1.6094)

**Correct answer:**

\[ -`\ln`{=tex}(0.2)`\approx1.6094`{=tex} \]

### 3. Which model has lower loss, and why?

**My answer:** \> model a has lower loss, and minimizing the loss means
assign bigger probability to the correct tokens, so the LLM's
performance can be improved.

**Correct answer:** Model A has lower loss because it assigns a higher
probability to the correct token. Minimizing training loss encourages
higher probability for observed targets, but does not automatically
guarantee clinical reasoning, factuality, or external generalization.

## Exercise 3 --- Perplexity

Given average losses 0.4 for Model A and 0.8 for Model B:

### 1. Model A perplexity

**My answer:** 1.4918\
**Correct answer:** (e\^{0.4}`\approx1.4918`{=tex})

### 2. Model B perplexity

**My answer:** 2.2255\
**Correct answer:** (e\^{0.8}`\approx2.2255`{=tex})

### 3. Does lower perplexity prove better clinical reasoning?

**My answer:** \> no, because lower perplexity only means the model has
less uncertainty about the possibilities assigned to observed tokens. we
still need to consider clinical factual accuracy, negation preservation,
hallucination rates, clinical terminology, downstream task performance,
and generalization ability to external hospitals.

**Correct answer:** No. Perplexity measures how well a model predicts
observed tokens under a specified evaluation setup. It does not directly
establish clinical reasoning or factual reliability. Clinical
factuality, hallucinations, negation, downstream performance, and
external validation must also be assessed.

## Exercise 4 --- Teacher forcing and exposure bias

### 1. Why does teacher forcing use correct preceding tokens?

**My answer:** \> the model was given correct tokens during training,
and it used the correct tokens to predict the next token.

**Correct answer:** Teacher forcing supplies ground-truth preceding
tokens during training, allowing the model to learn next-token
conditional probabilities using correct contexts.

### 2. Why can an incorrect generated token affect subsequent predictions?

**My answer:** \> a incorrect generated token can influence the
possibility assigned to the next tokens during inference, wrongly
exposure

**Correct answer:** At inference, the model conditions on its own
previous outputs. An incorrect output changes the context for subsequent
predictions. This mismatch between training and inference contexts is
exposure bias.

### 3. What safeguards would you introduce for clinical generation?

**My answer:** \> i will check with the clinical factual observation,
correct the wrong clinical notes and recalibrate with the true clinical
obervation

**Corrected answer:** Compare generated statements with source patient
records; verify medication names, doses, units, routes, and frequencies;
check contradictions, negation, and temporality; allow abstention when
evidence is insufficient; and require clinician review for high-risk
use.

**Important:** Factual verification and calibration are different.
Factual verification checks whether a statement matches the record.
Calibration checks whether predicted probabilities correspond to
observed frequencies.

------------------------------------------------------------------------

# Knowledge 5 --- Medical LLM evaluation

## Key knowledge

Different NLP tasks require different evaluation methods:

  -----------------------------------------------------------------------
  Task                    Example                 Typical metrics
  ----------------------- ----------------------- -----------------------
  Classification          Sepsis classification   AUROC, AUPRC, F1
                          from notes              

  Named entity            Extract medication      Precision, recall, F1
  recognition             names                   

  Relation extraction     Identify drug--dose     Precision, recall, F1
                          relationships           

  Text generation         Generate discharge      BLEU, ROUGE, BERTScore,
                          summaries               factuality

  Question answering      Answer clinical         Accuracy, exact match,
                          questions               clinical correctness
  -----------------------------------------------------------------------

### Precision, recall, and F1

\[ Precision=`\frac{TP}{TP+FP}`{=tex} \] \[
Recall=`\frac{TP}{TP+FN}`{=tex} \] \[
F1=2`\frac{Precision\times Recall}{Precision+Recall}`{=tex} \]

### Generation metrics

-   **BLEU:** N-gram precision with a brevity penalty.
-   **ROUGE-L:** Longest common subsequence overlap.
-   **BERTScore:** Similarity using contextual token embeddings.

These measure aspects of textual similarity. They do not independently
establish clinical factuality. A generated sentence can have high
overlap with a reference while reversing its meaning through negation or
changing a medication dose.

### Clinical LLM evaluation

A rigorous evaluation should include factual consistency with source
records, unsupported-claim/hallucination rate, medication accuracy,
negation preservation, completeness, temporal accuracy, clinical
relevance, expert clinician assessment, external hospital validation,
statistical uncertainty, and subgroup analysis.

### Domain adaptation

-   **DAPT:** Continue language-model pretraining using domain-specific
    text.
-   **TAPT:** Continue pretraining using text related to the downstream
    task.
-   **Fine-tuning:** Train a pretrained model for a particular
    downstream objective.

Domain adaptation does not automatically improve performance. It must be
compared with suitable baselines and evaluated on independent data.

## Exercise 1 --- Critically evaluating a medical LLM paper

**Scenario:** A model generates discharge summaries and reports BLEU =
0.82, ROUGE-L = 0.89, and BERTScore = 0.94. The authors claim the model
is clinically reliable, but report no factuality, hallucination,
medication accuracy, negation, external validation, or clinician
evaluation.

### 1. What do the metrics measure, and why are they insufficient?

**My answer:** \> BLUE measures the consecutive overlap of original and
generated texts, ROUGE-L measures the longest overlap subsequence
similarity of the original and generated texts. Bert score measure the
similarity of originaland generated texts using embedding
representations. \> \> the are insufficient because we still need more
reports such as clinical factuality, hallucination rates, medical
accuracy, negation preservation, external hpspital validation, expert
clinician evaluation.

**Corrected answer:** BLEU measures n-gram precision with a brevity
penalty. ROUGE-L measures longest-common-subsequence overlap. BERTScore
measures contextual embedding similarity. These metrics assess textual
similarity, not whether generated statements are supported by the
patient record or clinically correct.

### 2. Identify methodological problems.

**My answer:** 1. No report for clinical factuality. 2. No report of
hallucination rate. 3. No report of negation preservation.

**Correct answer:** All three are important omissions. Additional
concerns include missing medication accuracy, no external validation, no
clinician assessment, and an unsupported conclusion that textual
similarity proves clinical reliability.

### 3. What additional clinical evaluations would you require?

**My answer:** \> clinical factuality, hallucination rates, medical
accuracy, negation preservation, external hpspital validation, expert
clinician evaluation.

**Correct answer:** Define measurable outcomes: proportion of generated
claims supported by records; unsupported claims per summary; exact
agreement on medication details; negation preservation rate;
completeness of important findings; temporal accuracy; and structured
clinician ratings with inter-rater agreement.

### 4. What would you require as a peer reviewer?

**My answer:** \> clinical factuality check, external hospital
validation, expert evaluation

**Corrected answer:** Require source-grounded factuality assessment;
medication and numerical-value verification; negation, temporality, and
completeness assessment; structured clinician evaluation with
inter-rater agreement; external validation on independent hospital data;
suitable baselines; confidence intervals and appropriate paired
comparisons; and subgroup and error analyses.

------------------------------------------------------------------------

# Key formulas to revise

## Cosine similarity

\[
`\operatorname{cosine}`{=tex}(A,B)=`\frac{A\cdot B}{\|A\|\|B\|}`{=tex}
\]

Measures angular similarity between vectors.

## Skip-gram objective

\[ `\mathcal{L}`{=tex} =
-`\sum`{=tex}*t`\sum`{=tex}*{`\substack{-c\leq j\leq c\\j\neq0}`{=tex}}`\log `{=tex}P(w\_{t+j}`\mid `{=tex}w_t)
\]

Predicts surrounding context words from the target word.

## GloVe objective

\[
J=`\sum`{=tex}*{i,j}f(X*{ij})`\left`{=tex}(w_i^T`\tilde `{=tex}w_j+b_i+`\tilde `{=tex}b_j-`\log `{=tex}X\_{ij}`\right`{=tex})^2
\]

Learns vectors from global co-occurrence statistics.

## Autoregressive sequence probability

\[
P(w_1,`\ldots`{=tex},w_n)=`\prod`{=tex}*{t=1}\^{n}P(w_t`\mid `{=tex}w*{\<t})
\]

Factorizes sequence probability into conditional next-token
probabilities.

## Cross-entropy / negative log-likelihood

\[ L=-`\ln`{=tex}(p\_{`\text{correct}`{=tex}}) \]

Higher probability for the correct token produces lower loss.

## Perplexity

\[ PPL=`\exp`{=tex}(L\_{`\text{average}`{=tex}}) \]

Exponentiated average negative log-likelihood when natural logarithms
are used.

## Precision, recall, and F1

\[
Precision=`\frac{TP}{TP+FP}`{=tex},`\qquad `{=tex}Recall=`\frac{TP}{TP+FN}`{=tex}
\] \[ F1=2`\frac{Precision\times Recall}{Precision+Recall}`{=tex} \]

------------------------------------------------------------------------

# Day 7 --- Revision priorities

## High priority

-   **Static versus contextual embeddings:** Word2Vec and GloVe both
    produce static embeddings; BERT produces contextual representations.
-   **Clinical LLM evaluation:** Turn broad concerns such as factuality
    and hallucinations into measurable evaluation criteria.
-   **Research design:** Expand short recommendations into controlled
    experiments, baselines, external validation, and statistical
    uncertainty.
-   **Generation versus understanding:** A model suited to extracting
    information is not automatically suited to generating long-form
    text.

## Medium priority

-   Language-model mathematics: chain rule, cross-entropy, and
    perplexity.
-   Teacher forcing and exposure bias.
-   Tokenizer comparisons and embedding-vocabulary compatibility.
-   Word2Vec and GloVe training objectives.

## Misconceptions corrected

1.  GloVe does not produce contextual embeddings; it is a static
    embedding method.
2.  Token IDs do not inherently contain semantic meaning.
3.  BERT's self-attention considers the full permitted input sequence,
    not only adjacent words.
4.  Fewer tokens do not automatically indicate a better tokenizer.
5.  Skip-gram creates multiple target-context training pairs; it does
    not generate extra example sentences.
6.  Bidirectional attention is not automatically better than causal
    attention; task requirements matter.
7.  Perplexity does not directly measure clinical understanding.
8.  Calibration does not verify whether a generated clinical statement
    is factually correct.
9.  Textual similarity does not establish clinical reliability.
10. Domain adaptation does not guarantee downstream improvement.

------------------------------------------------------------------------

# Recommended learning materials

1.  **Speech and Language Processing --- Daniel Jurafsky and James H.
    Martin** --- Tokenization, embeddings, language modelling, and NLP
    evaluation.\
    https://web.stanford.edu/\~jurafsky/slp3/

2.  **Attention Is All You Need** --- Transformer architecture and
    self-attention.\
    https://arxiv.org/abs/1706.03762

3.  **BERT: Pre-training of Deep Bidirectional Transformers for Language
    Understanding** --- Bidirectional contextual representations and
    masked language modelling.\
    https://arxiv.org/abs/1810.04805

4.  **ClinicalBERT** --- Domain adaptation for clinical notes.\
    https://arxiv.org/abs/1904.05342

## Suggested practical exercises

-   Inspect how a tokenizer segments medical terminology.
-   Compare Word2Vec CBOW and Skip-gram on a small corpus.
-   Compare static embeddings with contextual BERT representations.
-   Calculate cross-entropy and perplexity for sample predictions.
-   Evaluate generated clinical text with BLEU, ROUGE-L, and BERTScore.
-   Design a factuality checklist for generated discharge summaries.

## Final Day 7 takeaway

\[
`\text{Text}`{=tex}`\rightarrow`{=tex}`\text{Tokens}`{=tex}`\rightarrow`{=tex}`\text{Embeddings}`{=tex}`\rightarrow`{=tex}`\text{Contextual representations}`{=tex}`\rightarrow`{=tex}`\text{Language modelling}`{=tex}`\rightarrow`{=tex}`\text{Clinical evaluation}`{=tex}
\]

Understanding how an LLM processes language is only the beginning.
Medical ML research must also establish whether its representations and
outputs preserve clinically important information, generalize to new
populations and hospitals, and remain reliable in real-world settings.

**Next: Day 8 --- Computer Vision and Medical Imaging.**
