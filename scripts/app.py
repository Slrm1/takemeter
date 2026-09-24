"""Local classifier interface.

Run from the repo root after training:

    python scripts/app.py
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "nfl-takemeter-distilbert"


def load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


def classify(text: str, tokenizer, model) -> tuple[str, float, dict[str, float]]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    labels = [model.config.id2label[i] for i in range(len(probs))]
    scores = {labels[i]: float(probs[i]) for i in range(len(labels))}
    best = max(scores, key=scores.get)
    return best, scores[best], scores


def main() -> None:
    if not MODEL_DIR.exists():
        raise SystemExit("Train the model first: python scripts/train_eval.py")
    tokenizer, model = load()
    print("TakeMeter — r/nfl discourse classifier")
    print("Paste a comment and press Enter. Empty line quits.")
    while True:
        text = input("\nComment: ").strip()
        if not text:
            break
        label, confidence, scores = classify(text, tokenizer, model)
        print(f"Label: {label}")
        print(f"Confidence: {confidence:.1%}")
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        print("Scores: " + ", ".join(f"{name} {score:.1%}" for name, score in ranked))


if __name__ == "__main__":
    main()
