# Show What You Know: TakeMeter

Time estimate: 9-11 hours total

TakeMeter is a fine-tuned text classifier that evaluates discourse quality in an online community. For this repository, the selected community is r/nfl.

The project requires designing a mutually exclusive 2-4 label taxonomy, collecting and annotating at least 200 public examples, fine-tuning a pretrained text classifier, comparing it against a zero-shot Groq baseline, and writing an honest evaluation of where the model works and fails.

## Grading

29 points total: 25 required + 4 stretch.

## Required Deliverables

- planning.md in the repo root, written before data collection and updated before stretch work
- one complete labeled CSV or JSON dataset in the repo or linked from the README
- README.md as the main graded report
- evaluation_results.json
- confusion_matrix.png as a supplementary image
- confusion matrix also written as a markdown table in the README
- 3-5 minute demo video linked from the README

## Required Features

### Label taxonomy

Define 2-4 labels that are mutually exclusive, apply to at least 90% of the chosen discourse without an other bucket, and reflect distinctions that matter to the community.

Document definitions and examples in both planning.md and README.md.

### Annotated dataset

Collect and label at least 200 public posts or comments. Save one complete labeled CSV; the notebook handles the 70/15/15 train, validation, and test split.

The README must document:
- source
- labeling process
- label distribution
- at least 3 real difficult-to-label examples and final decisions

### Fine-tuning

Fine-tune distilbert-base-uncased or another pretrained model. Document the starting model, training approach, and at least one hyperparameter decision.

### Zero-shot baseline

Use Groq meta-llama/llama-4-scout-17b-16e-instruct with no task-specific fine-tuning. Evaluate it on the same locked test set as the fine-tuned model.

### Evaluation report

The README must include:
- accuracy for both models
- per-class metrics for both models
- fine-tuned confusion matrix as a markdown table
- at least 3 specific wrong predictions with analysis
- 3-5 sample classifications with predicted label and confidence
- reflection on what the model learned versus what the labels intended

## Stretch Features

Update planning.md before implementing any stretch work.

- inter-annotator reliability on at least 30 examples
- confidence calibration
- systematic error-pattern analysis
- deployed interface showing label and confidence for a new post

## Recommended Stack

| Component | Tool |
|---|---|
| Base model | distilbert-base-uncased |
| Fine-tuning | Google Colab T4 GPU |
| Libraries | transformers, datasets, scikit-learn |
| Baseline | Groq meta-llama/llama-4-scout-17b-16e-instruct |

## Milestone 1: Choose Community and Labels

For this repo, the community is r/nfl.

Read real NFL discussion before locking labels. Define 2-4 labels, two clear examples per label, and at least one hard boundary case with an explicit decision rule.

## Milestone 2: Write the Spec

planning.md must address:
1. community and why it fits
2. labels and examples
3. hard edge cases
4. data collection plan
5. evaluation metrics and why they fit
6. concrete definition of success

It must also include an AI Tool Plan covering label stress-testing, annotation assistance, and failure analysis.

## Milestone 3: Collect and Annotate

Collect at least 200 public examples into one file. Required minimum columns are text and label. Keep notes on difficult cases.

No single label should dominate the dataset. Aim for at least 20 percent per class.

## Milestone 4: Run the Baseline

Upload the completed dataset into the starter Colab notebook, allow it to create the 70/15/15 split, lock the test set, and run the Groq zero-shot baseline before fine-tuning.

## Milestone 5: Fine-Tune

Fine-tune DistilBERT on the training data, evaluate on the same test set, and export evaluation_results.json and confusion_matrix.png.

## Milestone 6: Evaluate, Document, and Record

Complete the README with both models metrics, per-class results, text confusion matrix, three analyzed failures, sample classifications, reflection, spec reflection, and AI usage.

Record a 3-5 minute demo showing 3-5 classifications, confidence, one correct case, one incorrect case, and a short evaluation walkthrough.

## Submission Checklist

Submit:
- GitHub repository link
- planning.md
- complete labeled dataset
- complete README.md
- 3-5 minute demo video

The README is the main graded artifact, but planning.md is separately graded on its own completeness.
