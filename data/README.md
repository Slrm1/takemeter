# r/nfl TakeMeter Dataset

The final dataset belongs in nfl_takemeter_labeled.csv.

## Required Minimum

At least 200 public r/nfl comments, with one final label per row:
- analysis
- hot_take
- reaction

Target: 240 total, approximately 80 per label.

## Collection Rules

Use public comments from varied r/nfl thread types. Exclude usernames from model text, deleted or removed content, bots, duplicate comments, pure URLs, and non-football discussion.

Do not collect many near-identical replies from a single chain.

The final label must be manually reviewed using the decision rules in the repository-root planning.md.

Do not pre-split the file. The course notebook creates the train, validation, and test split.
