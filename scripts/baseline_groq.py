"""Zero-shot Groq baseline on the locked test split.

Requires GROQ_API_KEY in the environment. The key must not be written into the repo.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
from groq import Groq
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "nfl_takemeter_labeled.csv"
SPLIT = ROOT / "data" / "locked_split.json"
OUT_JSON = ROOT / "evaluation_results.json"
BASELINE_PRED = ROOT / "data" / "baseline_predictions.json"

LABELS = ["analysis", "hot_take", "reaction"]
SEED = 42
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

PROMPT = """You classify a public r/nfl comment into exactly one label: analysis, hot_take, or reaction.

analysis: A comment that makes a football claim and supports it with specific reasoning, evidence, tactical observation, statistical context, personnel usage, or a meaningful historical comparison.
hot_take: A confident evaluation, prediction, ranking, or broad claim stated with little meaningful support, or with evidence that is too thin or decorative to form a real argument.
reaction: An immediate emotional, humorous, celebratory, frustrated, or event-specific response that mainly expresses a feeling rather than a general football argument.

Decision order:
1. If the comment is mainly reacting to a just-seen play, score, penalty, injury, result, or piece of news without developing a broader claim, choose reaction.
2. If it makes a broader claim and provides specific football reasoning or evidence that would still support the claim if the emotional wording were removed, choose analysis.
3. If it makes a broader claim but mainly asserts, predicts, ranks, exaggerates, or uses only thin evidence, choose hot_take.

Reply with only one of these words: analysis, hot_take, reaction.

Comment:
{text}
"""


def lock_split() -> list[dict]:
    if SPLIT.exists():
        payload = json.loads(SPLIT.read_text(encoding="utf-8"))
        return payload["test"]
    df = pd.read_csv(DATA)
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label"])
    test_df = test_df.reset_index(drop=True)
    payload = {
        "seed": SEED,
        "sizes": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "test": test_df.to_dict(orient="records"),
    }
    SPLIT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload["test"]


def parse_label(text: str) -> str | None:
    cleaned = (text or "").strip().lower().replace("`", "").replace(".", "")
    if cleaned in LABELS:
        return cleaned
    found = [label for label in LABELS if label in cleaned.split()]
    if len(found) == 1:
        return found[0]
    return None


def main() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is not set")
    test_rows = lock_split()
    client = Groq()
    predictions = []
    for i, row in enumerate(test_rows):
        label = None
        raw = ""
        for attempt in range(3):
            try:
                completion = client.chat.completions.create(
                    model=MODEL,
                    temperature=0,
                    max_tokens=8,
                    messages=[{"role": "user", "content": PROMPT.format(text=row["text"])}],
                )
                raw = completion.choices[0].message.content or ""
                label = parse_label(raw)
                break
            except Exception as exc:  # noqa: BLE001
                raw = str(exc)
                time.sleep(2 * (attempt + 1))
        predictions.append(
            {
                "text": row["text"],
                "true_label": row["label"],
                "predicted_label": label,
                "raw_response": raw,
                "parsed": label is not None,
            }
        )
        print(f"{i+1}/{len(test_rows)} {label} <- {row['label']}")
        time.sleep(0.15)

    parsed = [row for row in predictions if row["parsed"]]
    y_true = [row["true_label"] for row in parsed]
    y_pred = [row["predicted_label"] for row in parsed]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    per_class = {}
    for i, label in enumerate(LABELS):
        per_class[label] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
    block = {
        "model": MODEL,
        "n_test": len(test_rows),
        "n_parsed": len(parsed),
        "n_unparsed": len(test_rows) - len(parsed),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4) if parsed else None,
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)), 4) if parsed else None,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist() if parsed else [],
        },
    }
    BASELINE_PRED.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    existing = {}
    if OUT_JSON.exists():
        existing = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    existing["baseline"] = block
    OUT_JSON.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(json.dumps(block, indent=2))


if __name__ == "__main__":
    main()
