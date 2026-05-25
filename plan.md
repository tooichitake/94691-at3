# Group Project Plan

**Group members:** Wilson Hsu (S1), Ratticha Ratanawarocha (S2), Szu-Yu Lin (S3), Zhiyuan Zhao (S4), Manal Ydkw (S5)

## Overview

This document records the group's plan for Phases 2 and 3 of Assignment 3. Phase 1 — the shared data-preparation notebook covering caption cleaning, image-level splitting, BPE tokenizer training, and image transforms — is complete, and its outputs (`data/processed/`) are the fixed inputs to every modelling notebook from this point on. Five members will each implement two architectures (Model 1 in Phase 2, Model 2 in Phase 3), giving ten models in total under a single shared data pipeline. We agreed on the design matrix below so that each member's two models isolate exactly one architectural variable, and across the cohort we span four encoders and three decoder families. The group will meet between Phase 2 and Phase 3 to consolidate findings and confirm each Model 2 before training.

The assignment is due on **20 May 2026**; the group aims to submit two days earlier.


## 1. Allocation of architectures

Each member's Model 1 and Model 2 jump is designed to answer a single question, which becomes the story of that member's two subsections in the final report.

S1 (Wilson Hsu). Model 1: ResNet-50 + LSTM. Model 2: ResNet-50 + LSTM + Bahdanau attention. Question: Does soft attention help an LSTM decoder when the encoder is held fixed?

S2 (Ratticha Ratanawarocha). Model 1: Small CNN (from scratch) + GRU. Model 2: ResNet-50 + GRU + Luong attention. Question: How much does encoder pretraining contribute at our training-set size?

S3 (Szu-Yu Lin). Model 1: ViT-B/16 + LSTM. Model 2: ViT-B/16 + Transformer decoder. Question: Does a Transformer decoder beat an LSTM when the encoder is the same?

S4 (Zhiyuan Zhao). Model 1: ResNet-50 + Transformer decoder. Model 2: CLIP ViT-B/16 + Transformer decoder. Question: Does pretraining target (ImageNet classification vs. CLIP image–text contrastive) matter?

S5 (Manal Ydkw). Model 1: ViT-B/16 + GRU + Luong attention. Model 2: ViT-B/16 + GRU + Bahdanau attention. Question: Multiplicative vs. additive attention scoring, everything else identical.

### Shared modelling commitments

We agreed on the following so that all ten models are comparable:

- The SentencePiece BPE vocabulary trained in Phase 1 is reused by every model. No one retrains the tokenizer.
- All decoders are trained from scratch. Pretrained language models (BERT, GPT-2, LLaMA, etc.) are not used — both because the assignment specification disallows them and because doing so would defeat the assignment's purpose.
- Visual encoders use their pretrained weights and are partially fine-tuned. S2's Phase-2 Small CNN is the only fully-trainable encoder.
- Each model's best checkpoint is selected by validation **CIDEr**.
- Test-set evaluation reports BLEU-1 to BLEU-4 (as required by the rubric), with CIDEr and ROUGE-L as supplementary metrics.


## 2. Timeline

Approximately two weeks of implementation, followed by one week of report writing.

Week 1. Shared `src/` working end-to-end. Smoke test passes on at least two members' machines, including at least one non-Windows environment.

Week 2. All five Model 1s trained and evaluated. Every member uploads their metrics and example captions to the shared drive.

Mid-Week 2. M3 group meeting (see the M3 meeting section below).

Week 3. All five Model 2s trained and evaluated.

Week 4. Report drafted, reviewed, finalised. Submission no later than 20 May 2026.

If any member anticipates slipping a milestone, they will flag it in the group chat as early as possible so the group can reallocate GPU time or report-writing duties.

## 3. Shared code (`src/`)

The `src/` package is shared and imported by every notebook. 

`dataset.py` — train, eval, and inference datasets.

`models/encoders.py` — ResNet, ViT, Small CNN, CLIP.

`models/decoders.py`, `models/attention.py` — LSTM and GRU families, Transformer, Bahdanau and Luong attention.

`models/captioner.py` — encoder–decoder glue.

`training.py` — Stage-1 cross-entropy training loop.

`inference.py` — greedy and beam decoding.

`evaluation.py` — BLEU, CIDEr, and ROUGE-L wrappers around COCO-caption.

`grpo.py` — Stage-2 reward fine-tuning, only if the group adopts it.

Interfaces are expected to evolve as the code is written, so we will not attempt to fix function signatures on paper. Any change to a function imported by another member's code will be announced in the group chat before being merged.

## 4. The M3 group meeting (between Phase 2 and Phase 3)

The purpose of this meeting is to compare the five Model 1 results together, confirm each member's Model 2, and capture the takeaways that become the group-discussion-summary section of the report. It is the only formal group checkpoint between Phases 2 and 3, so attendance is treated as compulsory. Remote attendance over Zoom is acceptable if necessary, and the meeting will be recorded.

Approximate agenda (about one hour):

- **5 minutes per member** — what I built, the metrics I got, one good and one bad caption example.
- **Together** — fill the five-row comparison table; discuss what stood out.
- **Together** — write down three to four takeaways from the deltas across models.
- **2 minutes per member** — which Model 2 I will build and which takeaway is driving the choice.
- **Wrap-up** — GPU schedule for Week 3 training, confirm the report editor.

Each takeaway should connect to a Model 2 that someone is going to build — that is how the group-discussion-summary section of the report derives its content.


## 5. Notebook structure

Every member's notebook follows the same twelve-section layout so the report editor can pull numbers uniformly and the marker can see the group coordinated.

0. Title, author, two-sentence summary of the two models.
1. Imports, random seed (`2026`), device selection.
2. Load the Phase 1 vocabulary and splits.
3. Build dataloaders.
4. **Justification of the Model 1 design** (markdown — this is the graded justification cell).
5. Model 1 code, parameter count, sanity check.
6. Model 1 training.
7. Model 1 evaluation — BLEU-1 to BLEU-4, CIDEr, and ROUGE-L on test, plus example captions.
8. **What the M3 meeting decided, and why this particular Model 2.**
9. Model 2 code.
10. Model 2 training and evaluation.
11. Direct Model 1 vs. Model 2 comparison.


## 6. Report sections and drafting responsibilities

The final report follows the rubric specified in the assignment brief.

1. Dataset — One member consolidates the Phase-1 group's notes.
2. Data preparation — Same as the dataset section above.
3. Architectures (ten subsections, two per member) — Each member drafts their own pair.
4. Group discussion summary — The M3 meeting facilitator (S3, by current assignment).
5. Metrics and loss functions (including Stage-2 if adopted) — S2 and S5.
6. Results and analysis (ten-model comparison table, training curves, failure cases) — The report editor compiles; each member supplies their own metrics and figures.
7. Limitations and future work — The report editor, with input from all members.
8. Statement of contributions — The report editor.
9. References — Each subsection author supplies their own references; the report editor de-duplicates and formats consistently.
