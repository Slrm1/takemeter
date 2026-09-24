"""Fine-tune DistilBERT and evaluate it on a locked test split.

The Groq baseline is a separate script so the test split can be locked first.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "nfl_takemeter_labeled.csv"
SPLIT = ROOT / "data" / "locked_split.json"
OUT_JSON = ROOT / "evaluation_results.json"
CM_PATH = ROOT / "confusion_matrix.png"
PRED_PATH = ROOT / "data" / "test_predictions.json"

MODEL_NAME = "distilbert-base-uncased"
LABELS = ["analysis", "hot_take", "reaction"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
SEED = 42
EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
MAX_LENGTH = 256


class TextDataset(Dataset):
    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def metrics_block(y_true: list[str], y_pred: list[str]) -> dict:
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
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)), 4),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": matrix,
        },
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, zero_division=0
        ),
    }


def main() -> None:
    set_seed(SEED)
    df = pd.read_csv(DATA)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    unknown = sorted(set(df["label"]) - set(LABELS))
    if unknown:
        raise SystemExit(f"unexpected labels: {unknown}")

    # 70/15/15 stratified. First hold out 30%, then split that half-and-half.
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    split_payload = {
        "seed": SEED,
        "sizes": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "train_texts": train_df["text"].tolist(),
        "validation_texts": val_df["text"].tolist(),
        "test": test_df.to_dict(orient="records"),
    }
    SPLIT.write_text(json.dumps(split_payload, ensure_ascii=False), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def encode(frame: pd.DataFrame) -> TextDataset:
        encodings = tokenizer(
            frame["text"].tolist(),
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )
        labels = [LABEL2ID[label] for label in frame["label"].tolist()]
        return TextDataset(encodings, labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    args = TrainingArguments(
        output_dir=str(ROOT / "models" / "checkpoints"),
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        seed=SEED,
        report_to=[],
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encode(train_df),
        eval_dataset=encode(val_df),
        processing_class=tokenizer,
    )
    trainer.train()
    model_dir = ROOT / "models" / "nfl-takemeter-distilbert"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    pred = trainer.predict(encode(test_df))
    probs = torch.softmax(torch.tensor(pred.predictions), dim=-1).numpy()
    pred_ids = probs.argmax(axis=1)
    y_true = test_df["label"].tolist()
    y_pred = [ID2LABEL[int(i)] for i in pred_ids]
    block = metrics_block(y_true, y_pred)

    rows = []
    for i, record in test_df.iterrows():
        confidence = float(probs[i].max())
        predicted = y_pred[i]
        rows.append(
            {
                "text": record["text"],
                "true_label": record["label"],
                "predicted_label": predicted,
                "confidence": round(confidence, 4),
                "probabilities": {LABELS[j]: round(float(probs[i][j]), 4) for j in range(len(LABELS))},
                "correct": predicted == record["label"],
                "notes": "" if pd.isna(record.get("notes", "")) else str(record.get("notes", "")),
                "source_url": "" if pd.isna(record.get("source_url", "")) else str(record.get("source_url", "")),
            }
        )
    PRED_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    matrix = np.array(block["confusion_matrix"]["matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Fine-tuned DistilBERT confusion matrix")
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            ax.text(c, r, str(matrix[r, c]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=150)
    plt.close(fig)

    results = {
        "model": MODEL_NAME,
        "hyperparameters": {
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "seed": SEED,
            "split": "70/15/15 stratified",
        },
        "split_sizes": split_payload["sizes"],
        "finetuned": {k: v for k, v in block.items() if k != "classification_report"},
        "classification_report": block["classification_report"],
    }
    # Preserve an existing baseline block if the baseline script already ran.
    if OUT_JSON.exists():
        previous = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        if "baseline" in previous:
            results["baseline"] = previous["baseline"]
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(block["classification_report"])
    print("accuracy", block["accuracy"], "macro_f1", block["macro_f1"])
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
