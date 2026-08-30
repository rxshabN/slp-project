# Multi-Turn Distress Trajectory Detection in Conversational Text for Crisis Triage Support

Rishab Nagwani · Hitakshi Sardana · Jhalak Malhotra · Anish Mohapatra

---

## 1. Introduction

Automated detection of psychological distress in text has converged on a single formulation. A model receives one unit of text — usually a social media post — and returns a label. Fifteen peer-reviewed studies from 2024–2026 surveyed for this project show that this task is close to saturated: performance on the Dreaddit benchmark sits in a band of roughly 0.80–0.88 F1 whether the classifier is a tuned logistic regression, a 9.5M-parameter transformer trained from scratch, or a domain-pretrained encoder such as MentalBERT. Chrifi Alaoui et al. [1] demonstrate parity with MentalBERT at roughly a twelfth of the parameters, which is strong evidence that further architectural work at this unit of analysis has small marginal return.

The problem is that the unit of analysis is wrong for the setting the work is meant to serve. Crisis helplines and text-based support services do not receive isolated posts; they receive conversations, and they already operate a prioritised queue rather than first-come-first-served ordering. Rana et al. [6] state the cost structure explicitly — misprioritisation produces delayed assistance for high-risk individuals, resource misallocation, deterioration and legal exposure — which is an asymmetric loss under a hard capacity constraint, not the symmetric objective that accuracy and macro-F1 encode.

Two findings make the mismatch concrete. Weilnhammer et al. [9], in *Nature Medicine*, scored over 90,000 conversational turns across 810 conversations on thirteen clinically relevant risk dimensions and found that risk accumulates gradually across turns rather than appearing abruptly, with escalation slopes that differ by vulnerability phenotype. Mansoor and Ansari [8] report a 7.2-day lead time from temporal analysis of posting history, establishing that trajectory carries information a single unit does not — though at a granularity of days rather than turns. A detector that reads only the current message is structurally blind to most of the signal these two studies measure.

The gap is recognised inside the literature but not closed. Chrifi Alaoui et al. [1] list the absence of temporal modelling among their stated limitations. Pereira et al. [7] name emotion dynamics modelling as an open challenge in emotion recognition in conversations, and note that the machinery for tracking affect across utterances is mature but has been benchmarked on general-purpose corpora such as IEMOCAP and MELD rather than distress or support conversations. Meanwhile the three reviewed studies that do use genuine crisis conversations [4], [5], [6] depend on institutional data held under confidentiality agreements, which is precisely why the open literature has retreated to public social media corpora.

This project closes the specific, testable part of that gap: it builds a system that models distress trajectory across the turns of a support conversation, evaluates that representation against a pointwise baseline on a shared target, reports performance under a fixed alert budget with calibrated confidence, and places explanation outside the real-time inference path.

### 1.1 Reformulation: risk to trajectory

The central design decision is to substitute the target. Clinical risk has no ground truth obtainable outside a crisis service. Trajectory does, because multi-turn support corpora record seeker-reported distress intensity before and after the conversation. ESConv [17] supplies exactly this: a pre-conversation and post-conversation self-rating on a 1–5 scale, for a corpus of emotional support dialogues that is public and reproducible.

The prediction target is therefore **non-improvement**: whether a conversation ends without the seeker's self-reported distress falling by at least one point. This is a defensible proxy for what a triage queue is actually ordering — conversations where the current exchange is not helping and a human should look sooner — and it makes no claim of clinical risk assessment.

---

## 2. Objectives

**O1 — Establish whether within-conversation trajectory beats a pointwise reading.** Run a controlled ablation in which four representations of the same turn-score sequence predict the same target: last-turn score, mean score, engineered trajectory features, and a learned sequence model. This comparison is absent from the reviewed literature and is directly testable. A null result is reportable.

**O2 — Evaluate against the triage objective rather than the classification convention.** Report recall on the non-improving class at fixed alert budgets (10%, 20%, 30% of conversations reviewable per shift) as the headline measure, with expected calibration error reported alongside, since the system surfaces confidence values to a human reviewer. No reviewed study reports either.

**O3 — Specify and measure an architecture where explanation does not occupy the synchronous path.** Compute attribution asynchronously, only for turns crossing the flag threshold, and report classifier latency and attribution latency as separate figures. The reviewed literature reports neither.

**O4 — Quantify the generalisation penalty rather than assuming it away.** Evaluate on a held-out split drawn from problem types unseen in training, and report the drop. Bhatt et al. [15] document class imbalance, monolingual restriction and temporal drift as recurring weaknesses; in-corpus scores systematically overstate deployed performance.

**O5 — Report the turn encoder honestly as a component, not a contribution.** Encoder capacity is not the variable under study. A small distilroberta-base encoder is used deliberately, following the finding [1] that large domain-pretrained encoders are not required for competitive post-level performance.

---

## 3. Methodology

### 3.1 System architecture

```
seeker turn ──▶ [ turn encoder ]──▶ calibrated score s_t ──┬──▶ [ trajectory layer ] ──▶ P(non-improving)
                 distilroberta                             │                                    │
                 + temperature                             │                                    ▼
                                                           │                          ranked triage queue
                                                           │
                                          s_t ≥ θ ──▶ [ async queue ] ──▶ [ occlusion attribution ]
                                                                                 (off critical path)
```

Three components, deliberately decoupled:

**Turn encoder.** distilroberta-base fine-tuned for binary stress on Dreaddit [16], then temperature-scaled on a held-out split. Dreaddit posts are long retrospective narratives; ESConv turns are short synchronous messages. To narrow that register gap, training posts are chunked into spans of at most three sentences, so the encoder sees turn-length input during supervision. Nothing is fine-tuned on ESConv text, so every turn score is an out-of-domain reading by construction — a limitation, but also what makes the transfer penalty measurable rather than hidden.

**Trajectory layer.** Consumes the per-conversation score sequence *s*₁…*s_T* and predicts non-improvement. Five arms, described in §4.2.

**Attribution service.** Leave-one-token-out occlusion, run on a background worker fed by a queue. Occlusion is chosen over integrated gradients (which needs backward passes) and over SHAP (whose pass count is not bounded in advance) because the number of forward passes is exactly one per token, which makes queue depth predictable under load.

### 3.2 Data

| Corpus | Role | Unit | Notes |
|---|---|---|---|
| Dreaddit [16] | turn-encoder supervision | post span | ~3.5k posts, binary stress label; chunked to ≤3 sentences for training, evaluated unchunked |
| ESConv [17] | trajectory training and evaluation | conversation | ~1.3k support dialogues; seeker pre/post intensity on 1–5 scale |

Label construction on ESConv: Δ = final − initial intensity. Label = 1 (non-improving) when Δ > −1, i.e. distress did not drop by at least one scale point. Conversations with fewer than four seeker turns are dropped, since a trajectory cannot be observed in three points. Conversations missing either survey field are dropped.

### 3.3 Splitting

All splits are at the conversation level, so no turn from a conversation appears on both sides. The in-distribution protocol is repeated stratified 5-fold cross-validation, 5 repeats (25 fits per arm), with out-of-fold predictions pooled for confidence intervals. The out-of-distribution protocol holds out entire problem types (`job crisis`, `academic pressure`) from training and tests only on them, which is a distributional shift in topic and vocabulary rather than a random resample.

### 3.4 Calibration

Every arm is put on a calibrated probability scale so the ECE comparison is fair. The logistic arms (A, B, C) are calibrated by construction. The sequence models hold out an inner 15% of each training fold purely to fit a scalar temperature, never touching the test fold. The turn encoder is temperature-scaled on the Dreaddit validation split.

---

## 4. Experimental Framework

### 4.1 Hypotheses

- **H1.** Trajectory representations (C, D1, D2) achieve higher recall@20% than the last-turn baseline (A) on the same target.
- **H2.** Order-sensitive representations (C, D1, D2) beat the order-free aggregate (B), isolating whether *sequence* matters or only *quantity* of distress. This is the sharper test; B is the harder baseline.
- **H3.** All arms degrade on the held-out problem types, and the trajectory arms degrade no worse than A — that is, trajectory signal is not merely topic memorisation.

### 4.2 Arms

| Arm | Representation | What it isolates |
|---|---|---|
| A | last-turn score only | the pointwise formulation the field deploys |
| B | mean score across turns | distress quantity, order discarded |
| C | 15 engineered features: last, mean, max, min, sd, first, last−first, OLS slope, slope over last 3 turns, final-third minus first-third, fraction above 0.5 and 0.6, longest consecutive-rise run, log turns, normalised area | explicit slope, recency and volatility |
| D1 | GRU over the score sequence | learned sequence structure |
| D2 | GRU over score + 16-dim PCA of turn embeddings | whether lexical content beyond the score adds anything |

D is kept small (64 hidden units, dropout 0.4). With ~1.3k conversations, an over-parameterised sequence model would make a null result uninterpretable — you would not know whether trajectory failed or the model simply could not be fit.

### 4.3 Metrics

**Primary.** Recall@budget for budgets of 10%, 20%, 30% — of the conversations that actually did not improve, the fraction ranked inside the top *b* fraction. Expected calibration error over 15 bins.

**Secondary.** Precision@budget, AUROC, AUPRC, Brier score.

**Reported alongside.** Base rate, and the random-ordering reference line (recall@*b* = *b*), because a triage system that does not beat random ordering under the operating budget has no operational value regardless of its AUROC.

### 4.4 Statistical protocol

Five seeds end to end, including encoder retraining. Metrics reported as mean ± sd across seeds. Paired bootstrap (1,000 resamples over conversations) for each arm against arm A on recall@20% and AUROC, Bonferroni-corrected across the eight comparisons. Seed-level p-values combined by Fisher's method. 95% bootstrap CIs on the headline metric.

### 4.5 Latency protocol

Two hundred to three hundred real seeker turns are streamed through the service. Synchronous latency is measured as tokenisation plus one forward pass plus the flag decision — everything a counsellor waits on. Asynchronous latency is measured separately inside the worker. Reported as p50/p95/p99, plus the ratio of attribution cost to classifier cost, which is the figure that establishes why attribution cannot sit in the request path.

---

## 5. Results

Run `bash run_all.sh`, then `python -m src.aggregate`. Tables are written to `results/`. Fill the templates below directly from `results/table_main.csv`, `table_significance.csv` and `table_ood.csv`.

### Table 2. Trajectory ablation, in-distribution (mean ± sd over 5 seeds)

| Model | recall@10 | **recall@20** | recall@30 | precision@20 | AUROC | AUPRC | ECE |
|---|---|---|---|---|---|---|---|
| A. Last-turn score (pointwise) | | | | | | | |
| B. Mean score | | | | | | | |
| C. Engineered trajectory features | | | | | | | |
| D1. GRU over score sequence | | | | | | | |
| D2. GRU over scores + embeddings | | | | | | | |
| *Random ordering* | 0.100 | 0.200 | 0.300 | base rate | 0.500 | base rate | — |

### Table 3. Significance against the pointwise baseline

| Comparison | Δ recall@20 | Fisher *p* (Bonferroni-corrected) |
|---|---|---|
| B vs A | | |
| C vs A | | |
| D1 vs A | | |
| D2 vs A | | |

### Table 4. Out-of-distribution (held-out problem types)

| Model | recall@20 | AUROC | ECE | Δ vs in-distribution |
|---|---|---|---|---|

### Table 5. Latency decomposition

| Path | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|
| Synchronous: encode + score + flag | | | |
| Asynchronous: occlusion attribution | | | — |

Flag rate at θ = 0.6: ____ . Attribution-to-classifier cost ratio: ____ ×.

### Figures

- `results/recall_budget_seed*.png` — recall against alert budget for all five arms, with the random-ordering diagonal. This is the headline figure.
- `results/reliability_seed*.png` — calibration curves, pooled out-of-fold.

### 5.1 How to read the outcome

Three outcomes are all publishable, and the write-up should be drafted before the numbers are known so the framing is not chosen post hoc.

1. **C/D1 > B > A, significantly.** Trajectory carries signal beyond distress quantity. The headline claim of the paper. Report the slope and final-third-minus-first-third feature weights from C, since these are the interpretable version of what D1 learns.
2. **C/D1 ≈ B > A.** Aggregation over turns helps but *order* does not. This is a narrower but clean finding, and it is a genuine correction to the assumption in [8] and [9] that the shape of the trajectory matters at conversation granularity.
3. **All arms ≈ A, or all near random ordering.** The likely cause is that the Dreaddit-supervised encoder does not transfer to short support turns, in which case the turn-score sequence is noise and the trajectory layer has nothing to work with. Diagnose before concluding: check the turn-score distribution printed by `score_conversations.py` (if mean ≈ 0.5 with small sd, the encoder is not discriminating on this domain), and check whether the ESConv self-report labels have signal at all by fitting a logistic regression on situation-text TF-IDF. Report the diagnosis; "trajectory did not help *given this encoder*" is a different and weaker claim than "trajectory does not help", and the paper must not conflate them.

---

## 6. Limitations

Stated rather than concealed, per §4 of the review.

- **The corpus is not a helpline.** ESConv is crowdsourced emotional support dialogue, not crisis transcripts. Every reviewed study using genuine crisis conversations [4], [5], [6] depends on institutional access that is unavailable externally; that constraint is the reason for this substitution, not an oversight.
- **The target is a self-report proxy.** Non-improvement in seeker-rated intensity is confounded with supporter quality, demand characteristics of the crowdsourcing task, and the coarseness of a 1–5 scale. It is a proxy for triage priority, not for clinical risk.
- **The turn encoder is supervised out of domain.** Dreaddit labels stress in long retrospective narratives. Chunking narrows the register gap but does not close it.
- **Sample size limits the sequence models.** ~1.3k conversations constrains how much a learned trajectory model can be asked to do, and a negative result for D1/D2 relative to C should be attributed to data scale, not to sequence modelling in general.
- **Grimland et al. [4] show risk constructs vary by gender and age**, implying a single global threshold does not serve all callers equally. ESConv carries no demographic metadata, so this project cannot test threshold fairness. It should be named as required future work rather than silently omitted.
- **No clinical validity is claimed.** The system is positioned as triage *ordering support* for a human reviewer. It makes no risk assessment and is not a substitute for one.

---

## 7. Mapping to the identified gaps

| Gap | Addressed by | Status |
|---|---|---|
| 3.1 Unit of analysis is the post | conversation-level target, turn sequences | modelled |
| 3.2 Trajectory acknowledged, not modelled | the A/B/C/D ablation, §4.2 | the central experiment |
| 3.3 Metrics do not match triage | recall@budget + ECE as primary, §4.3 | modelled |
| 3.4 Explanation post-hoc, offline, unbudgeted | async attribution service + measured latency, §4.5 | modelled |
| 3.5 Single-corpus evaluation overstates performance | held-out problem-type split | evaluation obligation, not a modelling target |
