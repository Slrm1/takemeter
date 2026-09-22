# TakeMeter - r/nfl Discourse Classifier

TakeMeter is a fine-tuned text classifier for r/nfl discourse. It classifies how an NFL comment argues its point, not whether the football opinion is correct.

The three labels are analysis, hot_take, and reaction.

## Project Status

Planning and community design: complete

Data collection and manual annotation: pending

Fine-tuning and baseline evaluation: pending

Final evaluation report: pending

Demo video: pending

I am intentionally not filling evaluation sections with invented metrics. They will be completed from the actual Colab outputs after the r/nfl dataset is collected and labeled.

## Community Choice

I chose r/nfl because its discussions contain a useful range of discourse styles. Live and post-game threads contain immediate reactions, while coaching, quarterback, roster, prediction, and strategy discussions contain both unsupported strong opinions and detailed evidence-based arguments.

The classifier therefore measures discourse structure, not agreement with the comment. A controversial opinion can be analysis when it is genuinely supported; a popular opinion can still be a hot_take when it is simply asserted.

See planning.md for the full pre-collection design and edge-case rules.

## Label Taxonomy

| Label | Definition | Clear synthetic example |
|---|---|---|
| analysis | A football claim supported by specific reasoning, evidence, tactical observation, statistics, personnel usage, or meaningful historical comparison. | The blitz rate is not the main issue; the secondary keeps losing leverage on quick routes, so quarterbacks have an immediate answer. |
| hot_take | A confident evaluation, prediction, ranking, or broad claim with little meaningful support or only decorative evidence. | They are absolutely missing the playoffs. This team is cooked. |
| reaction | An immediate emotional, humorous, celebratory, frustrated, or event-specific response with little broader argument. | HOW DID HE MISS THAT KICK? |

### Key boundary rule

When a strong claim contains evidence, it is analysis only when the evidence meaningfully supports the conclusion. If the evidence is thin, cherry-picked, or added mainly to make an assertion sound stronger, the comment remains a hot_take.

## Dataset

The final dataset will be stored at data/nfl_takemeter_labeled.csv.

Planned target: 240 public r/nfl comments, approximately 80 per label.

| Column | Purpose |
|---|---|
| text | Comment text supplied to the classifier |
| label | analysis, hot_take, or reaction |
| notes | Optional annotation note |
| source_type | Thread type used to monitor source diversity |
| source_url | Public r/nfl thread URL for traceability |

### Data collection source

Comments will come from public r/nfl game threads, post-game threads, news threads, discussion threads, and longer analysis posts. Deleted or removed comments, bots, duplicates, pure link drops, and non-football comments will be excluded.

### Label distribution

To be filled from the completed CSV.

| Label | Count | Percentage |
|---|---:|---:|
| analysis | TBD | TBD |
| hot_take | TBD | TBD |
| reaction | TBD | TBD |
| Total | TBD | 100% |

### Difficult-to-label examples

This section will contain at least three real annotated comments that caused genuine uncertainty.

1. TBD
2. TBD
3. TBD

## Fine-Tuning Approach

Base model: distilbert-base-uncased

Planned setup:
- task: three-class sequence classification
- split: 70/15/15
- epochs: 3
- learning rate: 2e-5
- batch size: 16
- environment: Google Colab T4 GPU
- libraries: Hugging Face Transformers, Datasets, and scikit-learn

### Hyperparameter decision

The initial learning rate is 2e-5 because this project uses a small labeled dataset and a pretrained DistilBERT model. A conservative learning rate reduces the risk of overwriting useful pretrained representations too aggressively.

## Zero-Shot Baseline

Baseline model: Groq meta-llama/llama-4-scout-17b-16e-instruct

The exact same locked test set used for the fine-tuned model will be classified zero-shot using the label definitions from planning.md.

## Evaluation Report

This section will be completed from the real locked test set.

### Overall comparison

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Groq zero-shot baseline | TBD | TBD |
| Fine-tuned DistilBERT | TBD | TBD |

### Per-class metrics - zero-shot baseline

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| analysis | TBD | TBD | TBD |
| hot_take | TBD | TBD | TBD |
| reaction | TBD | TBD | TBD |

### Per-class metrics - fine-tuned model

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| analysis | TBD | TBD | TBD |
| hot_take | TBD | TBD | TBD |
| reaction | TBD | TBD | TBD |

### Fine-tuned confusion matrix

The assignment requires the confusion matrix as text in the README.

| True / Predicted | analysis | hot_take | reaction |
|---|---:|---:|---:|
| analysis | TBD | TBD | TBD |
| hot_take | TBD | TBD | TBD |
| reaction | TBD | TBD | TBD |

A supplementary image will also be committed as confusion_matrix.png.

### Wrong-prediction analysis

At least three specific fine-tuned-model errors will be analyzed here.

#### Error 1
Text: TBD

True label: TBD

Predicted label: TBD

Why it likely failed: TBD

What would help: TBD

#### Error 2
Text: TBD

True label: TBD

Predicted label: TBD

Why it likely failed: TBD

What would help: TBD

#### Error 3
Text: TBD

True label: TBD

Predicted label: TBD

Why it likely failed: TBD

What would help: TBD

### Sample classifications

| Example post | Predicted label | Confidence |
|---|---|---:|
| TBD | TBD | TBD |
| TBD | TBD | TBD |
| TBD | TBD | TBD |
| TBD | TBD | TBD |
| TBD | TBD | TBD |

## Definition of Success

Pre-registered threshold:
- accuracy >= 0.75
- macro F1 >= 0.72
- every class F1 >= 0.60
- fine-tuned model improves on the zero-shot baseline by at least 5 percentage points in accuracy or macro F1
- no class collapses almost entirely into another in the confusion matrix

## What the Model Learned vs. What I Intended

To be written after evaluation.

This section will focus on whether the model learned the intended reasoning structure or instead over-relied on shortcuts such as comment length, statistics, sarcasm, capitalization, profanity, or game-thread language.

## Spec Reflection

### How the spec helped

The pre-collection spec forces the label boundaries to be explicit before annotation. In particular, defining the difference between an evidence-backed hot take and real analysis reduces the temptation to change the meaning of a label halfway through the dataset.

### Where implementation diverged

To be completed after training. Any change to the original data target, labels, cleaning rules, model, or hyperparameters will be documented here with the reason.

## AI Usage

### Instance 1 - label stress-testing

I used ChatGPT to challenge the label taxonomy with borderline NFL examples. The final rules focus on whether evidence meaningfully supports a broader claim rather than on comment length or how controversial the opinion sounds.

### Instance 2 - project and rubric planning

I gave ChatGPT the TakeMeter assignment requirements and directed it to specialize the project for r/nfl. It produced an NFL-specific planning structure, dataset target, metric reasoning, and explicit success thresholds. I kept placeholders rather than allowing invented training results or evaluation numbers.

### Annotation disclosure

The initial dataset is planned as manual final annotation. If an LLM is later used to pre-label rows, that use will be disclosed here and every AI-suggested label will be manually reviewed.

## Demo Video

Demo link: Add the 3-5 minute video URL after recording.

The demo must show 3-5 r/nfl-style comments, predicted label and confidence, one correct prediction, one incorrect prediction, and a brief walkthrough of the evaluation report.

## Submission Checklist

- [x] Community chosen: r/nfl
- [x] 3-label taxonomy defined
- [x] Hard edge-case rule defined
- [x] Data collection and class-balance plan defined
- [x] Evaluation metrics justified
- [x] Concrete success threshold defined
- [x] AI Tool Plan documented in planning.md
- [ ] At least 200 public r/nfl comments collected
- [ ] Complete labeled CSV committed
- [ ] At least 3 real difficult annotations documented
- [ ] Zero-shot Groq baseline run
- [ ] DistilBERT fine-tuned
- [ ] evaluation_results.json committed
- [ ] confusion_matrix.png committed
- [ ] Both models metrics written in README
- [ ] Confusion matrix written as a markdown table
- [ ] 3 wrong predictions analyzed
- [ ] 3-5 sample classifications with confidence added
- [ ] Learned-vs-intended reflection completed
- [ ] 3-5 minute demo recorded and linked
