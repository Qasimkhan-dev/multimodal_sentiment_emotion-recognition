import os
import json
import random
import subprocess
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, accuracy_score

# =========================
# CONFIG
# =========================
BALANCED_DIR = "/mnt/optimusmesh/checkingfiles/meld_balanced"
AUDIO_BASE = "/mnt/optimusmesh/checkingfiles/meld_audio_15"
SUBSET_TAG = "20"

TRAIN_CSV = f"{BALANCED_DIR}/train_{SUBSET_TAG}_sentiment.csv"
DEV_CSV   = f"{BALANCED_DIR}/dev_{SUBSET_TAG}_sentiment.csv"
TEST_CSV  = f"{BALANCED_DIR}/test_{SUBSET_TAG}_sentiment.csv"

TRAIN_AUDIO = f"{AUDIO_BASE}/train"
DEV_AUDIO   = f"{AUDIO_BASE}/dev"
TEST_AUDIO  = f"{AUDIO_BASE}/test"

MODEL_NAME = "facebook/wav2vec2-base"
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 5
MAX_AUDIO_SECONDS = 8
TARGET_SR = 16000
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = "/mnt/optimusmesh/checkingfiles/audio_outputs_sentiment_15"
RESULTS_DIR = "/mnt/optimusmesh/checkingfiles/results"
FEATURE_DIR = "/mnt/optimusmesh/checkingfiles/features"
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


def load_wav(path, target_sr=16000, max_seconds=8):
    cmd = [
        "ffmpeg",
        "-i", path,
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(target_sr),
        "-"
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    if result.returncode != 0 or len(result.stdout) == 0:
        raise RuntimeError(f"Failed to decode audio: {path}")

    wav = np.frombuffer(result.stdout, dtype=np.float32)
    wav = torch.tensor(wav, dtype=torch.float32)

    max_len = target_sr * max_seconds
    if wav.shape[0] > max_len:
        wav = wav[:max_len]
    elif wav.shape[0] < max_len:
        pad = max_len - wav.shape[0]
        wav = torch.nn.functional.pad(wav, (0, pad))

    return wav


class AudioDataset(Dataset):
    def __init__(self, csv_path, audio_root):
        self.df = pd.read_csv(csv_path).copy()
        self.audio_root = audio_root

        valid_rows = []
        for _, row in self.df.iterrows():
            uid = uid_from_row(row)
            audio_path = os.path.join(audio_root, f"{uid}.wav")
            label = str(row["Sentiment"]).strip().lower()

            if os.path.exists(audio_path) and label in SENTIMENT2ID:
                valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"{csv_path} -> usable audio samples: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = uid_from_row(row)
        audio_path = os.path.join(self.audio_root, f"{uid}.wav")

        wav = load_wav(audio_path, target_sr=TARGET_SR, max_seconds=MAX_AUDIO_SECONDS)
        label = SENTIMENT2ID[str(row["Sentiment"]).strip().lower()]

        return {
            "waveform": wav,
            "label": torch.tensor(label, dtype=torch.long),
            "uid": uid
        }


class AudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(MODEL_NAME)
        self.encoder.gradient_checkpointing_enable()
        self.classifier = nn.Linear(self.encoder.config.hidden_size, len(SENTIMENT2ID))

    def forward(self, input_values, attention_mask=None):
        out = self.encoder(input_values=input_values, attention_mask=attention_mask)
        feat = out.last_hidden_state.mean(dim=1)
        logits = self.classifier(feat)
        return logits, feat



processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)


def collate_fn(batch):
    waves = [item["waveform"].numpy() for item in batch]

    enc = processor(
        waves,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )

    input_values = enc["input_values"]

    if "attention_mask" in enc:
        attention_mask = enc["attention_mask"]
    else:
        attention_mask = torch.ones_like(input_values, dtype=torch.long)

    labels = torch.stack([item["label"] for item in batch])
    uids = [item["uid"] for item in batch]

    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "label": labels,
        "uid": uids
    }


def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            input_values = batch["input_values"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            logits, _ = model(input_values, attention_mask)
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
            input_values = batch["input_values"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            _, feats = model(input_values, attention_mask)

            ids_all.extend(batch["uid"])
            feats_all.append(feats.cpu())
            labels_all.extend(batch["label"].tolist())

    feats_all = torch.cat(feats_all, dim=0)
    save_features(
        f"{FEATURE_DIR}/audio_sentiment_15_{split_name}.pt",
        ids_all,
        feats_all,
        labels_all
    )


def save_metrics_json(train_loss, train_acc, dev_loss, dev_acc, test_loss, test_acc):
    out_path = f"{RESULTS_DIR}/audio_sentiment_15_metrics.json"
    metrics = {
        "model_name": MODEL_NAME,
        "task": "sentiment",
        "subset_csv_tag": SUBSET_TAG,
        "audio_folder_tag": "15",
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "epochs": EPOCHS,
        "max_audio_seconds": MAX_AUDIO_SECONDS,
        "target_sr": TARGET_SR,
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

    train_ds = AudioDataset(TRAIN_CSV, TRAIN_AUDIO)
    dev_ds   = AudioDataset(DEV_CSV, DEV_AUDIO)
    test_ds  = AudioDataset(TEST_CSV, TEST_AUDIO)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    dev_loader   = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = AudioModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_dev_acc = 0.0
    best_path = f"{OUT_DIR}/best_audio_sentiment.pt"

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            input_values = batch["input_values"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(input_values, attention_mask)
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

    print("\n========== FINAL RESULTS: AUDIO SENTIMENT ==========\n")

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
