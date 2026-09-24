# TakeMeter - planning.md

## Community Choice: r/nfl

TakeMeter will study public comments from r/nfl, the main NFL discussion community on Reddit. It is a strong classification setting because the same game, player, coach, trade, or news event can produce very different kinds of discourse: immediate emotional reactions, bold unsupported claims, and detailed football analysis using statistics, scheme, personnel, or historical comparisons.

The classifier will focus on football-discussion comments from game threads, post-game threads, news/discussion threads, weekly discussion posts, and longer original-analysis posts. Deleted comments, bot comments, pure link drops, and comments with no football-discussion content will not be included.

The goal is not to decide whether an opinion is correct. The goal is to classify how the opinion is argued.

## Label Taxonomy

### analysis

Definition: A comment that makes a football claim and supports it with specific reasoning, evidence, tactical observation, statistical context, personnel usage, or a meaningful historical comparison.

Synthetic examples:
1. The pressure rate looks worse than the sack total because the defense keeps rushing four and the secondary is giving quarterbacks an easy first read.
2. I would keep the safety deep against this offense. Their explosive passes come when linebackers bite on play action, so forcing underneath throws takes away their best advantage.

### hot_take

Definition: A confident evaluation, prediction, ranking, or broad claim stated with little meaningful support, or with evidence that is too thin or decorative to form a real argument.

Synthetic examples:
1. This team is going 5-12. Their window is completely closed.
2. He is already a top-five quarterback and anyone who disagrees is not watching football.

### reaction

Definition: An immediate emotional, humorous, celebratory, frustrated, or event-specific response that mainly expresses a feeling rather than a general football argument.

Synthetic examples:
1. HOW DID HE DROP THAT?
2. That fourth-down call just took ten years off my life.

## Decision Rules and Hard Edge Cases

Primary decision order:

1. If the comment is mainly reacting to a just-seen play, score, penalty, injury, result, or piece of news without developing a broader claim, label it reaction.
2. If it makes a broader claim and provides specific football reasoning or evidence that would still support the claim if the emotional wording were removed, label it analysis.
3. If it makes a broader claim but mainly asserts, predicts, ranks, exaggerates, or uses only thin evidence, label it hot_take.

Hardest anticipated edge case: evidence-backed hot takes.

Synthetic borderline example: This coach is holding the offense back. They have scored on only one opening drive all month and still run the same predictable early-down concepts.

Decision rule: label it analysis when the evidence is specific and the reasoning connects that evidence to the conclusion. Label it hot_take when the evidence is merely decorative, cherry-picked without explanation, or too weak to support the size of the claim.

Additional rules:
- Sarcasm about a just-completed play is normally reaction.
- Sarcasm used to make a broader unsupported player or team evaluation is hot_take.
- A short comment can still be analysis if it contains a clear causal football observation.
- A long comment is not automatically analysis.
- A factual correction with no opinion should be excluded rather than forced into a label.
- Questions that do not express a take should be excluded.
- Quoted parent text should be removed when possible.

## Data Collection Plan

All examples will come from public r/nfl comments.

I will sample across multiple thread types:
- live game and RedZone threads
- post-game threads
- news threads
- weekly discussion threads
- original analysis/discussion posts
- prediction, ranking, coaching, roster, and quarterback discussions

I will collect from multiple teams and multiple weeks rather than centering the dataset on one fanbase or one game.

Target dataset size: 240 comments.

| Label | Target count | Target share |
|---|---:|---:|
| analysis | 80 | 33% |
| hot_take | 80 | 33% |
| reaction | 80 | 33% |
| Total | 240 | 100% |

The final dataset will be saved at data/nfl_takemeter_labeled.csv.

CSV columns:
- text
- label
- notes
- source_type
- source_url

Cleaning rules:
- Collect only public material from r/nfl.
- Do not include usernames in model text.
- Exclude deleted or removed comments, bots, duplicates, pure URLs, and non-football comments.
- Avoid collecting many near-identical replies from one chain.
- Do not search only for phrases such as hot take; sample ordinary discussion too.
- Read and label each selected comment individually.
- Keep at least three difficult real examples with notes for the README.

If a label is underrepresented after the first 200 annotations, I will intentionally sample thread types likely to contain more of that class rather than changing borderline labels just to balance the numbers.

## Evaluation Metrics

I will report overall accuracy, per-class precision, recall, and F1, macro F1, and a confusion matrix.

Accuracy is useful but not sufficient because one label could perform poorly while the overall number still looks acceptable.

Macro F1 will be the main summary metric because it weights analysis, hot_take, and reaction equally even if the final class counts are not perfectly balanced.

The confusion matrix will be used to inspect the hardest boundaries, especially analysis vs hot_take and hot_take vs reaction.

The fine-tuned model and Groq zero-shot baseline will be evaluated on the same locked test set.

## Definition of Success

I will consider the classifier useful enough for a real prototype if it meets all of these thresholds:

- overall accuracy of at least 0.75
- macro F1 of at least 0.72
- no class with F1 below 0.60
- fine-tuned model exceeds the zero-shot baseline by at least 5 percentage points in either accuracy or macro F1
- the confusion matrix does not show one class collapsing almost entirely into another

If the model misses these thresholds, I will report that honestly and analyze whether the main issue is annotation consistency, insufficient data, ambiguous boundaries, or model capacity.

## Model and Training Plan

Base model: distilbert-base-uncased

Initial setup:
- 3 epochs
- learning rate 2e-5
- batch size 16
- 70/15/15 train/validation/test split
- Google Colab T4 GPU

I will not tune hyperparameters against the test set. Any changes will be based on training and validation behavior and documented in the README.

## Zero-Shot Baseline Plan

Baseline model: Groq meta-llama/llama-4-scout-17b-16e-instruct

The prompt will include the exact three label definitions and require the model to return only one of:
analysis
hot_take
reaction

The baseline will run on exactly the same locked test examples as the fine-tuned model. Unparseable responses will be counted and reported rather than silently dropped.

## AI Tool Plan

### Label stress-testing

I will give ChatGPT the label definitions, decision order, and evidence-backed hot-take edge case. I will ask it to generate 5-10 synthetic NFL comments that sit at the boundaries between labels. I will classify them myself and tighten the rules before real annotation if needed.

### Annotation assistance

I will not use an LLM to make the final labels for the initial dataset. I may use AI to discuss a small number of genuinely difficult examples after recording my own first judgment, but the final annotation decision will be mine.

If I later use an LLM to pre-label any batch, I will track those rows, manually review every label, and disclose that process in the README.

### Failure analysis

After evaluation, I will give ChatGPT a table containing true label, predicted label, confidence, and text for misclassified test examples. I will ask it to suggest systematic patterns such as sarcasm, short comments, statistics used without reasoning, absolute predictions, or game-thread language. I will verify every proposed pattern myself before reporting it.

## Planned Stretch Features

No stretch feature will be treated as completed until this document is updated before implementation.

Priority:
1. error pattern analysis
2. deployed interface
3. confidence calibration
4. inter-annotator reliability

### Stretch update before implementation

Written before the stretch work, after the labeled dataset existed and before the fine-tuned test errors were interpreted.

I am doing three stretch features:

1. Error pattern analysis. After the test predictions exist, I will group every fine-tuned mistake by the true/predicted pair and look for a repeating cause, especially analysis versus hot_take on short comments, sarcasm, and one decorative statistic. I will check any pattern against the actual misclassified texts before writing it up.
2. Confidence calibration. I will bin the fine-tuned model's maximum class probability and compare accuracy in the high-confidence bin with accuracy in the lower-confidence bin on the same locked test set.
3. A local interface. A small Python app will take a new comment, load the saved DistilBERT classifier, and print the label and confidence. The README will say how to run it.

I am not doing inter-annotator reliability. I do not have a second person who can independently label 30 or more comments, and I will not invent a second annotator or an agreement rate.

## End-to-End Workflow

1. Finalize and stress-test the label rules.
2. Collect public r/nfl comments into one CSV.
3. Manually label each eligible comment.
4. Audit distribution and difficult cases.
5. Upload the CSV to the course Colab notebook.
6. Lock the train/validation/test split.
7. Run the Groq zero-shot baseline.
8. Fine-tune DistilBERT.
9. Evaluate on the same test set.
10. Export evaluation_results.json and confusion_matrix.png.
11. Analyze at least three wrong predictions and systematic error patterns.
12. Complete the README and record the 3-5 minute demo.
