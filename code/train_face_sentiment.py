import os
import json
import random
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from facenet_pytorch import MTCNN, InceptionResnetV1

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

NUM_FRAMES = 8
FACE_SIZE = 160
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = f"/mnt/optimusmesh/checkingfiles/face_outputs_sentiment_{SUBSET_TAG}"
RESULTS_DIR = "/mnt/optimusmesh/checkingfiles/results"
FEATURE_DIR = "/mnt/optimusmesh/checkingfiles/features"
CACHE_DIR = f"/mnt/optimusmesh/checkingfiles/face_cache_sentiment_{SUBSET_TAG}"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

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


def sample_video_frames(video_path, num_frames=8):
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


mtcnn = MTCNN(image_size=FACE_SIZE, margin=10, keep_all=False, device=DEVICE)


def extract_face_sequence(frames):
    face_tensors = []

    for frame in frames:
        face = mtcnn(frame)
        if face is None:
            face = torch.zeros(3, FACE_SIZE, FACE_SIZE)
        face_tensors.append(face.cpu())

    return torch.stack(face_tensors, dim=0)


class FaceDataset(Dataset):
    def __init__(self, csv_path, video_root):
        self.df = pd.read_csv(csv_path).copy()
        self.video_root = video_root

        valid_rows = []
        for _, row in self.df.iterrows():
            video_path = get_video_path(video_root, row)
            label = str(row["Sentiment"]).strip().lower()
            if os.path.exists(video_path) and label in SENTIMENT2ID:
                valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"{csv_path} -> usable face samples: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = uid_from_row(row)
        video_path = get_video_path(self.video_root, row)
        cache_path = os.path.join(CACHE_DIR, f"{uid}.pt")

        if os.path.exists(cache_path):
            face_seq = torch.load(cache_path, map_location="cpu")
        else:
            frames = sample_video_frames(video_path, NUM_FRAMES)
            if len(frames) == 0:
                face_seq = torch.zeros(NUM_FRAMES, 3, FACE_SIZE, FACE_SIZE)
            else:
                face_seq = extract_face_sequence(frames)
            torch.save(face_seq, cache_path)

        label = SENTIMENT2ID[str(row["Sentiment"]).strip().lower()]
        return {
            "faces": face_seq,
            "label": torch.tensor(label, dtype=torch.long),
            "uid": uid
        }


class AttentionPool(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        pooled = (weights * x).sum(dim=1)
        return pooled


class FaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained="vggface2").eval()

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )
        self.attn = AttentionPool(512)
        self.classifier = nn.Linear(512, len(SENTIMENT2ID))

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        with torch.no_grad():
            feat = self.backbone(x)

        feat = feat.view(B, T, 512)
        lstm_out, _ = self.lstm(feat)
        pooled = self.attn(lstm_out)
        logits = self.classifier(pooled)
        return logits, pooled


def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            faces = batch["faces"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            logits, _ = model(faces)
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
            faces = batch["faces"].to(DEVICE)
            _, feats = model(faces)

            ids_all.extend(batch["uid"])
            feats_all.append(feats.cpu())
            labels_all.extend(batch["label"].tolist())

    feats_all = torch.cat(feats_all, dim=0)
    save_features(
        f"{FEATURE_DIR}/face_sentiment_{SUBSET_TAG}_{split_name}.pt",
        ids_all,
        feats_all,
        labels_all
    )


def save_metrics_json(train_loss, train_acc, dev_loss, dev_acc, test_loss, test_acc):
    out_path = f"{RESULTS_DIR}/face_sentiment_{SUBSET_TAG}_metrics.json"
    metrics = {
        "task": "sentiment",
        "subset_tag": SUBSET_TAG,
        "num_frames": NUM_FRAMES,
        "face_size": FACE_SIZE,
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

    train_ds = FaceDataset(TRAIN_CSV, TRAIN_VID)
    dev_ds   = FaceDataset(DEV_CSV, DEV_VID)
    test_ds  = FaceDataset(TEST_CSV, TEST_VID)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader   = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = FaceModel().to(DEVICE)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )
    criterion = nn.CrossEntropyLoss()

    best_dev_acc = 0.0
    best_path = f"{OUT_DIR}/best_face_sentiment.pt"

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        print(f"\nStarting epoch {epoch+1}/{EPOCHS}")
        model.train()

        running_loss = 0.0
        running_examples = 0

        for step, batch in enumerate(train_loader, 1):
            faces = batch["faces"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(faces)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            running_examples += y.size(0)

            if step % 20 == 0 or step == len(train_loader):
                avg_step_loss = running_loss / max(running_examples, 1)
                print(
                    f"Epoch {epoch+1}/{EPOCHS} | "
                    f"Step {step}/{len(train_loader)} | "
                    f"Train Loss {avg_step_loss:.4f}"
                )

        dev_loss, dev_acc, _, _, _ = evaluate_model(model, dev_loader, criterion)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Dev   Loss: {dev_loss:.4f} | Dev   Acc: {dev_acc:.4f}")
        print(f"Epoch Time: {time.time() - epoch_start:.1f}s")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    print("\n========== FINAL RESULTS: FACE SENTIMENT ==========\n")

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
