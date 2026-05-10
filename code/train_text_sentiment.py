import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, accuracy_score

# =========================
# CONFIG
# =========================
BALANCED_DIR = "/mnt/optimusmesh/checkingfiles/meld_balanced"
SUBSET_TAG = "20"   # change to "10" if your generated subset files are actually 10%

TRAIN_CSV = f"{BALANCED_DIR}/train_{SUBSET_TAG}_sentiment.csv"
DEV_CSV   = f"{BALANCED_DIR}/dev_{SUBSET_TAG}_sentiment.csv"
TEST_CSV  = f"{BALANCED_DIR}/test_{SUBSET_TAG}_sentiment.csv"

MODEL_NAME = "roberta-base"
MAX_LEN = 128
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = f"/mnt/optimusmesh/checkingfiles/meld_project/outputs/checkpoints/text_outputs_sentiment_{SUBSET_TAG}"
RESULTS_DIR = f"/mnt/optimusmesh/checkingfiles/meld_project/outputs/results"
FEATURE_DIR = f"/mnt/optimusmesh/checkingfiles/meld_project/outputs/features"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)

SENTIMENT2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}
TARGET_NAMES = list(SENTIMENT2ID.keys())
LABEL_IDS = list(range(len(TARGET_NAMES)))


def uid_from_row(row):
    return f"dia{int(row['Dialogue_ID'])}_utt{int(row['Utterance_ID'])}"


def save_features(save_path, ids, features, labels):
    torch.save({
        "ids": ids,
        "features": features.cpu(),
        "labels": labels
    }, save_path)
    print(f"Saved features to: {save_path}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.df = pd.read_csv(csv_path).copy()
        self.df["Utterance"] = self.df["Utterance"].fillna("").astype(str)
        self.df["Sentiment"] = self.df["Sentiment"].astype(str).str.strip().str.lower()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["Utterance"]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        label = SENTIMENT2ID[row["Sentiment"]]

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "uid": uid_from_row(row)
        }


class TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, len(SENTIMENT2ID))

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        feat = out.last_hidden_state[:, 0, :]
        logits = self.classifier(feat)
        return logits, feat


def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            logits, _ = model(ids, mask)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)

            all_labels.extend(y.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels,
        all_preds,
        labels=LABEL_IDS,
        target_names=TARGET_NAMES,
        digits=4,
        zero_division=0
    )
    return avg_loss, acc, report, all_labels, all_preds


def save_all_features(model, loader, split_name):
    ids_all, feats_all, labels_all = [], [], []
    model.eval()

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            _, feats = model(ids, mask)

            ids_all.extend(batch["uid"])
            feats_all.append(feats.cpu())
            labels_all.extend(batch["label"].tolist())

    feats_all = torch.cat(feats_all, dim=0)
    save_features(
        f"{FEATURE_DIR}/text_sentiment_{SUBSET_TAG}_{split_name}.pt",
        ids_all,
        feats_all,
        labels_all
    )


def save_metrics_json(train_loss, train_acc, dev_loss, dev_acc, test_loss, test_acc):
    out_path = f"{RESULTS_DIR}/text_sentiment_{SUBSET_TAG}_metrics.json"
    metrics = {
        "model_name": MODEL_NAME,
        "task": "sentiment",
        "subset_tag": SUBSET_TAG,
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "epochs": EPOCHS,
        "device": DEVICE,
        "train": {"loss": train_loss, "accuracy": train_acc},
        "dev": {"loss": dev_loss, "accuracy": dev_acc},
        "test": {"loss": test_loss, "accuracy": test_acc},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics json to: {out_path}")


def run():
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = TextDataset(TRAIN_CSV, tokenizer)
    dev_ds   = TextDataset(DEV_CSV, tokenizer)
    test_ds  = TextDataset(TEST_CSV, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    dev_loader   = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = TextModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_dev_acc = 0.0
    best_path = f"{OUT_DIR}/best_text_sentiment.pt"

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(ids, mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

        train_loss, train_acc, _, _, _ = evaluate_model(model, train_loader, criterion)
        dev_loss, dev_acc, _, _, _ = evaluate_model(model, dev_loader, criterion)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Dev   Loss: {dev_loss:.4f} | Dev   Acc: {dev_acc:.4f}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    print("\n========== FINAL RESULTS: SENTIMENT ==========\n")

    train_loss, train_acc, train_report, _, _ = evaluate_model(model, train_loader, criterion)
    print("TRAIN")
    print(f"Loss: {train_loss:.4f}")
    print(f"Accuracy: {train_acc:.4f}")
    print(train_report)

    dev_loss, dev_acc, dev_report, _, _ = evaluate_model(model, dev_loader, criterion)
    print("DEV")
    print(f"Loss: {dev_loss:.4f}")
    print(f"Accuracy: {dev_acc:.4f}")
    print(dev_report)

    test_loss, test_acc, test_report, _, _ = evaluate_model(model, test_loader, criterion)
    print("TEST")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(test_report)

    save_metrics_json(train_loss, train_acc, dev_loss, dev_acc, test_loss, test_acc)

    save_all_features(model, train_loader, "train")
    save_all_features(model, dev_loader, "dev")
    save_all_features(model, test_loader, "test")

    print(f"\nAll outputs saved in: {OUT_DIR}")


if __name__ == "__main__":
    run()
