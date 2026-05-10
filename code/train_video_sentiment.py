import os
import math
import json
import random
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.metrics import classification_report, accuracy_score

# =========================
# CONFIG
# =========================
BASE_PATH = "/mnt/optimusmesh/checkingfiles/meld_dataset/MELD-RAW/MELD.Raw"
BALANCED_DIR = "/mnt/optimusmesh/checkingfiles/meld_balanced"
SUBSET_TAG = "20"

TRAIN_CSV = f"{BALANCED_DIR}/train_{SUBSET_TAG}_sentiment.csv"
DEV_CSV   = f"{BALANCED_DIR}/dev_{SUBSET_TAG}_sentiment.csv"
TEST_CSV  = f"{BALANCED_DIR}/test_{SUBSET_TAG}_sentiment.csv"

TRAIN_VID = f"{BASE_PATH}/train/train_splits"
DEV_VID   = f"{BASE_PATH}/dev/dev_splits_complete"
TEST_VID  = f"{BASE_PATH}/test/output_repeated_splits_test"

NUM_FRAMES = 16
IMG_SIZE = 224
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = f"/mnt/optimusmesh/checkingfiles/video_outputs_sentiment_{SUBSET_TAG}"
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


def get_video_path(video_root, row):
    uid = uid_from_row(row)
    return os.path.join(video_root, f"{uid}.mp4")


def load_video_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    selected = set(indices.tolist())
    frames = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in selected:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        idx += 1

    cap.release()

    if len(frames) == 0:
        return []

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return frames[:num_frames]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=1280, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class VideoDataset(Dataset):
    def __init__(self, csv_path, video_root):
        self.df = pd.read_csv(csv_path).copy()
        self.video_root = video_root
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        valid_rows = []
        for _, row in self.df.iterrows():
            video_path = get_video_path(video_root, row)
            label = str(row["Sentiment"]).strip().lower()
            if os.path.exists(video_path) and label in SENTIMENT2ID:
                valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"{csv_path} -> usable video samples: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = uid_from_row(row)
        video_path = get_video_path(self.video_root, row)

        frames = load_video_frames(video_path, NUM_FRAMES)
        if len(frames) == 0:
            frames = [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) for _ in range(NUM_FRAMES)]

        frames = [self.transform(frame) for frame in frames]
        video_tensor = torch.stack(frames, dim=0)

        label = SENTIMENT2ID[str(row["Sentiment"]).strip().lower()]
        return {
            "video": video_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "uid": uid
        }


class VideoModel(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.pos_enc = PositionalEncoding(1280)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1280,
            nhead=8,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, len(SENTIMENT2ID))
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        x = self.features(x)
        x = self.pool(x).view(B * T, -1)
        x = x.view(B, T, 1280)

        x = self.pos_enc(x)
        x = self.transformer(x)

        feat = x.mean(dim=1)
        logits = self.classifier(feat)
        return logits, feat


def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            videos = batch["video"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            logits, _ = model(videos)
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
            videos = batch["video"].to(DEVICE)
            _, feats = model(videos)

            ids_all.extend(batch["uid"])
            feats_all.append(feats.cpu())
            labels_all.extend(batch["label"].tolist())

    feats_all = torch.cat(feats_all, dim=0)
    save_features(
        f"{FEATURE_DIR}/video_sentiment_{SUBSET_TAG}_{split_name}.pt",
        ids_all,
        feats_all,
        labels_all
    )


def save_metrics_json(train_loss, train_acc, dev_loss, dev_acc, test_loss, test_acc):
    out_path = f"{RESULTS_DIR}/video_sentiment_{SUBSET_TAG}_metrics.json"
    metrics = {
        "task": "sentiment",
        "subset_tag": SUBSET_TAG,
        "num_frames": NUM_FRAMES,
        "img_size": IMG_SIZE,
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

    train_ds = VideoDataset(TRAIN_CSV, TRAIN_VID)
    dev_ds   = VideoDataset(DEV_CSV, DEV_VID)
    test_ds  = VideoDataset(TEST_CSV, TEST_VID)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    dev_loader   = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = VideoModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_dev_acc = 0.0
    best_path = f"{OUT_DIR}/best_video_sentiment.pt"

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            videos = batch["video"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(videos)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

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

    print("\n========== FINAL RESULTS: VIDEO SENTIMENT ==========\n")

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
