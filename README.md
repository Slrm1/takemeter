# TakeMeter - r/nfl Discourse Classifier

TakeMeter is a fine-tuned text classifier for public r/nfl comments. It classifies how a comment argues, not whether the football opinion is correct.

The three labels are `analysis`, `hot_take`, and `reaction`.

The fine-tuned model is `distilbert-base-uncased`. On the locked 46-comment test set it reached **52.2% accuracy** and **0.41 macro F1**. That misses the success threshold in `planning.md`. The model predicts `analysis` for 40 of 46 test comments. It caught every true analysis comment and almost none of the hot takes.

The Groq zero-shot baseline was not run. The available Groq key is expired, and the local classifier does not need it. The comparison section below says what is missing.

## Community Choice

r/nfl is the main NFL discussion community on Reddit. The same injury, trade, coaching story, or highlight produces immediate reactions, bold unsupported claims, and comments that actually reason from scheme, personnel, or history. Regulars already talk about that difference: a "cooked" prediction is not the same thing as a film read.

The classifier measures discourse structure. A controversial opinion can be analysis when the evidence would still support the claim after the emotional wording is removed. A popular opinion can still be a hot take when it is only asserted.

## Label Taxonomy

| Label | Definition | Example |
|---|---|---|
| analysis | A football claim supported by specific reasoning, evidence, tactical observation, statistics, personnel usage, or a meaningful historical comparison. | "Yeah, the inside receiver ran outside and the outside receiver ran inside. The two defenders ran into each other trying to cover them. It's not often you see a defense run a pick play on itself." |
| hot_take | A confident evaluation, prediction, ranking, or broad claim with little meaningful support, or with evidence too thin to carry the size of the claim. | "I will almost guarantee Penix is going to come in, play like about the 22nd best QB, we win 4 of our last 6, and pick 8th." |
| reaction | An immediate emotional, humorous, celebratory, frustrated, or event-specific response that mainly expresses a feeling rather than a broader football argument. | "I hope Jackson, Allen, and Mahomes can take care of their bodies, because damn is football fun when they are healthy." |

Second example for each label:

- **analysis:** "Not really. Rough count is around 27 snaps where he was faced one on one pre snap and 6 where he was double covered intentionally."
- **hot_take:** "And if you ignore all the bad teams, they're actually winless since 2024."
- **reaction:** "The day the Empire was born. I don't think anyone would imagine that the greatest player of all time would be a 6th round pick."

### Boundary rule

When a strong claim contains evidence, it is analysis only when that evidence meaningfully supports the conclusion. If the evidence is thin, cherry-picked, or added to make an assertion sound stronger, the comment stays a hot take. An immediate response to a play, score, penalty, injury, or news item with no broader argument is a reaction, even if it names a player.

## Dataset

Labeled file: `data/nfl_takemeter_labeled.csv`

Columns: `text`, `label`, `notes`, `source_type`, `source_url`.

### Source

Public r/nfl comments collected on September 23, 2026 through the Arctic Shift public Reddit mirror, after Reddit's own JSON endpoint returned HTTP 403. The pool was recent comments across news, highlight, injury, and discussion threads. Deleted comments, bots, duplicates, pure links, non-football chatter, pure questions, and factual corrections with no take were excluded.

### Labeling process

Each kept comment was read against the decision order in `planning.md`: reaction first if it is mainly an in-the-moment response, analysis if a broader claim still stands once the emotional wording is removed and the evidence actually supports it, otherwise hot take. A first random sample of 300 comments produced too much analysis relative to the other two labels, so a second sample of shorter unused comments was read to add hot takes and reactions. Labels were not flipped to force the counts.

### Label distribution

| Label | Count | Percentage |
|---|---:|---:|
| analysis | 126 | 41.2% |
| hot_take | 90 | 29.4% |
| reaction | 90 | 29.4% |
| Total | 306 | 100% |

No label is above 70%. Each label is above 20%. The split used for training is stratified 70/15/15 with seed 42: 214 train, 46 validation, 46 test.

### Difficult examples

1. **Herbert has no accolades.** "I have personally thought the Chargers have been one of the most over hyped teams for the last decade. Especially after Herbert came to town. I don't understand how people keep saying he is a top 10 QB. He has no accolades to back that up at all." This could be analysis because it cites a criterion. It is a hot take because "no accolades" does not actually support a top-10 rejection. The evidence is decorative relative to the size of the claim.

2. **Penix threshold.** "If Penix is not at least a top 20 QB immediately it probably a sign we should cut and run, but if he's at least as good as like Tyler Shough or Baker Mayfield it would be dumb to not stick with him." This could be a hot take because it is a strong quarterback verdict. It is analysis because it states a decision rule that connects a performance level to a roster choice.

3. **Eagles streak plus an exact score.** "That streak will last at least one more game. They have the Bears third-stringer coming up next... But, then they have LAR, JAC, CHA and DAL... I'm calling at least one game to be 31-27." The 31-27 call is a hot take sitting on top of a real schedule argument. The comment is analysis because the streak claim would still stand if the exact score were removed.

## Fine-Tuning Approach

Base model: `distilbert-base-uncased`

Training ran locally with Hugging Face Transformers on CPU, because this machine has no CUDA GPU. The course notebook's Colab path was not required to run the same training setup.

- Task: three-class sequence classification
- Split: stratified 70/15/15, seed 42
- Epochs: 3
- Learning rate: 2e-5
- Batch size: 16
- Max length: 256
- Train loss after 3 epochs: 0.996
- Validation loss by epoch: 1.045, 0.982, 0.953

### Hyperparameter decision

The learning rate stayed at 2e-5. The dataset is small and the base model is already pretrained, so a larger rate would be more likely to overwrite useful representations. Max length 256 was chosen because these comments fit inside it, which keeps CPU training shorter than a 512-token window. Epochs and batch size stayed at the planned 3 and 16.

Run it again with:

```bash
python scripts/train_eval.py
```

## Zero-Shot Baseline

The required baseline is Groq `meta-llama/llama-4-scout-17b-16e-instruct`, prompted with the three label definitions and instructed to return only `analysis`, `hot_take`, or `reaction`. The prompt is in `scripts/baseline_groq.py`. It is written to score the same locked test split as the fine-tuned model.

That script did not produce numbers. The Groq key available on this machine returns `expired_api_key`. The local model does not depend on Groq, so training and classification continued without it. There is no baseline accuracy to report, and no claim that fine-tuning beat a zero-shot model.

To run the baseline after a new key is in the environment:

```bash
python scripts/baseline_groq.py
```

## Evaluation Report

Test set: 46 comments. Fine-tuned model only.

### Overall result

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Groq zero-shot baseline | not run | not run |
| Fine-tuned DistilBERT | 0.522 | 0.407 |

### Per-class metrics - fine-tuned model

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| analysis | 0.475 | 1.000 | 0.644 | 19 |
| hot_take | 1.000 | 0.071 | 0.133 | 14 |
| reaction | 0.800 | 0.308 | 0.444 | 13 |

### Fine-tuned confusion matrix

Rows are true labels. Columns are predicted labels.

| True \ Predicted | analysis | hot_take | reaction |
|---|---:|---:|---:|
| analysis | 19 | 0 | 0 |
| hot_take | 12 | 1 | 1 |
| reaction | 9 | 0 | 4 |

A supplementary image is saved as `confusion_matrix.png`.

### Did it meet the success threshold?

No.

- Accuracy 0.522 is below 0.75.
- Macro F1 0.407 is below 0.72.
- `hot_take` F1 is 0.133 and `reaction` F1 is 0.444, both below 0.60.
- The baseline comparison could not be made.
- `hot_take` effectively collapsed into `analysis`.

### Wrong-prediction analysis

#### Error 1

Text: "I think Penix sucks, and anything he shows you only risks a commitment you don't want. You guys talk a lot about how much it sucks to constantly spin your wheels because you go on these late, meaningless season runs that don't get you to playoff contention but knock you out of a great pick. Well, that's the future I envision for Penix, if I'm being real here."

True label: `hot_take`

Predicted label: `analysis` (confidence 0.525)

The model confused a hot take with analysis. The comment does contain a mechanism, late-season wins hurting draft position, so it looks argued. The label is hot take because "Penix sucks" and the envisioned future are larger than that mechanism. This is a labeling-boundary problem the model did not learn, not a random miss. Fixing it would take more long hot takes that mention a reason without that reason actually carrying the claim.

#### Error 2

Text: "And if you ignore all the bad teams, they're actually winless since 2024."

True label: `hot_take`

Predicted label: `analysis` (confidence 0.400)

This is sarcasm doing the work of a hot take. It sounds like a statistical correction, which is exactly the surface pattern analysis comments use. The model treated the fake stat as evidence. The boundary is hard because sarcasm and real statistical argument share the same shape. The label is consistent with the sarcasm rule in `planning.md`. More sarcastic stat-jokes labeled hot take would be the training fix.

#### Error 3

Text: "The day the Empire was born. I don't think anyone would imagine that the greatest player of all time would be a 6th round pick."

True label: `reaction`

Predicted label: `analysis` (confidence 0.471)

The comment is a celebratory reaction to the Brady anniversary. The model treated "greatest player of all time" plus "6th round pick" as a historical argument. The football fact is the feeling, not a developed comparison. Short emotional comments that mention a real football fact are the reaction-versus-analysis failures. Nine of thirteen true reactions were predicted as analysis.

### Error pattern

The systematic pattern is an analysis prior. The model predicted `analysis` 40 times, `reaction` 5 times, and `hot_take` once. Every true analysis comment was correct, and 12 of 14 hot takes plus 9 of 13 reactions were pulled into analysis.

It learned that a comment mentioning players, schemes, or a clause of reasoning should be analysis. It did not learn the intended distinction between evidence that supports a claim and evidence that only decorates one. Confidence stays low while it does this: 44 of 46 predictions are under 0.60, so it is unsure and still defaults to the largest class.

### Sample classifications

| Example | Predicted label | Confidence |
|---|---|---:|
| "ZBB at 100% is a huge loss. ZBB at whatever percent he was trying to play at is still a loss, but not irreplaceable. The question is whether Belton can be that replacement." | analysis | 0.584 |
| "Yeah, the inside receiver ran outside and the outside receiver ran inside. The two defenders ran into each other trying to cover them." | analysis | 0.579 |
| "I will almost guarantee Penix is going to come in, play like about the 22nd best QB, we win 4 of our last 6, and pick 8th." | analysis | 0.517 |
| "It seems like AJ was just making everybody's life miserable in Philadelphia" | reaction | 0.412 |
| "I hope Jackson, Allen, and Mahomes can take care of their bodies, because damn is football fun when they are healthy" | analysis | 0.369 |

The ZBB comment is a reasonable analysis prediction. It separates a full-strength loss from a partial one and names the replacement question, which is the personnel reasoning the label is supposed to capture. The Penix prediction in this table is wrong: that comment is a hot take, and the model called it analysis.

### Confidence calibration

Maximum class probability on the test set, compared with accuracy:

| Confidence bin | Count | Accuracy |
|---|---:|---:|
| 0.00–0.60 | 44 | 0.50 |
| 0.60–0.75 | 2 | 1.00 |
| 0.75–1.00 | 0 | — |

A higher score is slightly more meaningful: the only two predictions above 0.60 were both correct. The model almost never reaches 0.75, and a score around 0.50 is a coin flip. The confidence values are weakly ordered and not calibrated to the probability a user would expect from "90% sure."

## What the Model Learned vs. What I Intended

The intended target was the structure of the argument: specific support versus assertion versus an in-the-moment feeling. What the model captured is closer to "does this comment look like football explanation?" Analysis comments in the training data are longer and denser with scheme, counts, and named players. Hot takes and reactions often name the same players. The model used that overlap as a reason to predict analysis.

It missed sarcasm, exact-sounding predictions with no support, and celebratory comments that happen to include a true football fact. Those are the cases the label rules were written to catch. The low confidence and the one-class predictions mean it did not learn a stable three-way boundary.

## Definition of Success

Pre-registered threshold from `planning.md`:

- accuracy at least 0.75
- macro F1 at least 0.72
- every class F1 at least 0.60
- fine-tuned model ahead of the zero-shot baseline by at least 0.05 accuracy or macro F1
- no class collapsed into another

The fine-tuned model missed every one of those that can be checked. It is not useful as a community tool yet.

## Spec Reflection

The spec helped by requiring the confusion matrix as a table and a pre-registered threshold. Accuracy of 0.52 could be waved off as "better than chance on three classes." The matrix makes the actual behavior obvious: 19 of 19 analysis comments are correct because the model rarely predicts anything else.

The implementation diverged from the spec in two places. Training ran locally on CPU instead of a Colab T4, because the machine already had Transformers and no GPU, and the data and metrics are the same either way. The Groq baseline was not run, because the key on this machine is expired and classification was continued without it. The dataset is also 306 comments rather than the planned 240, with analysis at 41% rather than an even three-way split, because the comments that survived the football-only filter contained more real arguments than hot takes.

## AI Usage

### Instance 1 - annotation

The coding agent was directed to collect public r/nfl comments and apply the label rules in `planning.md`. It retrieved comments from Arctic Shift and assigned every final label by reading the comment against the decision order. Off-topic comments, questions, and pure factual corrections were excluded. A second pass added shorter comments because the first pass was heavy on analysis. Those labels were not produced by a separate human annotator. Borderline rows are marked in the `notes` column.

### Instance 2 - failure analysis

After training, the misclassified test comments were grouped by true label and predicted label. The first pattern that appeared was "the model hates hot takes." Checking the texts changed that conclusion. The model is not specifically failing sarcasm or length. It predicts `analysis` whenever the comment contains football detail, including sarcastic stats, emotional history, and unsupported predictions. That corrected pattern is the one reported above.

No LLM was used to pre-label a batch that was then skimmed. The labels in the CSV are the annotation pass described in instance 1.

## Local interface

`scripts/app.py` loads the saved model and prints a label and confidence for a new comment. Train first if `models/nfl-takemeter-distilbert` is not present. Model weights are local and are not required for the written report.

```bash
python scripts/train_eval.py
python scripts/app.py
```

Paste a comment and press Enter. An empty line quits.

## Demo Video

Demo: [Screen Recording 2026-09-23 194307.mp4](file:///C:/Users/selor/Videos/Screen%20Recordings/Screen%20Recording%202026-09-23%20194307.mp4)

Run `python scripts/app.py` for the live classifications.

## Submission Checklist

- [x] Community chosen: r/nfl
- [x] 3-label taxonomy defined
- [x] Hard edge-case rule defined
- [x] Labeled CSV with 306 comments
- [x] Three difficult annotations documented
- [x] DistilBERT fine-tuned
- [x] `evaluation_results.json` and `confusion_matrix.png` written
- [x] Confusion matrix written as a markdown table
- [x] Three wrong predictions analyzed
- [x] Sample classifications with confidence
- [x] Learned-versus-intended reflection
- [x] Error-pattern and confidence-calibration stretch notes
- [x] Local interface in `scripts/app.py`
- [ ] Groq zero-shot baseline
- [x] 3–5 minute demo recorded and linked
