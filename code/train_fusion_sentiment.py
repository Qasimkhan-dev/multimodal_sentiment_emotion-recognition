import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score

# =========================
# CONFIG
# =========================
TEXT_FEATURE_DIR = "/mnt/optimusmesh/checkingfiles/meld_project/outputs/features"
FEATURE_DIR = "/mnt/optimusmesh/checkingfiles/features"
RESULTS_DIR = "/mnt/optimusmesh/checkingfiles/results"
OUT_DIR = "/mnt/optimusmesh/checkingfiles/fusion_outputs_sentiment"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 5

SUBSET_TAG = "20"
AUDIO_TAG = "15"

SENTIMENT2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}
TARGET_NAMES = list(SENTIMENT2ID.keys())
LABEL_IDS = list(range(len(TARGET_NAMES)))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_feature_file(path):
    return torch.load(path, map_location="cpu")


def to_feature_map(data):
    return {uid: feat for uid, feat in zip(data["ids"], data["features"])}


def to_label_map(data):
    return {uid: label for uid, label in zip(data["ids"], data["labels"])}


def get_common_ids(*feature_maps):
    common = set(feature_maps[0].keys())
    for fmap in feature_maps[1:]:
        common &= set(fmap.keys())
    return sorted(common)


class FusionDataset(Dataset):
    def __init__(self, ids, text_map, audio_map, video_map, face_map, label_map):
        self.ids = ids
        self.text_map = text_map
        self.audio_map = audio_map
        self.video_map = video_map
        self.face_map = face_map
        self.label_map = label_map

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        uid = self.ids[idx]

        text_feat = self.text_map[uid]
        audio_feat = self.audio_map[uid]
        video_feat = self.video_map[uid]
        face_feat = self.face_map[uid]

        fused = torch.cat([text_feat, audio_feat, face_feat, video_feat], dim=0).float()
        label = torch.tensor(self.label_map[uid], dtype=torch.long)

        return fused, label


class FusionMLP(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
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
    return avg_loss, acc, report


def save_metrics_json(train_loss, train_acc, dev_loss, dev_acc, test_loss, test_acc):
    out_path = f"{RESULTS_DIR}/fusion_sentiment_metrics.json"
    metrics = {
        "task": "fusion_sentiment",
        "text_tag": SUBSET_TAG,
        "audio_tag": AUDIO_TAG,
        "video_tag": SUBSET_TAG,
        "face_tag": SUBSET_TAG,
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
    set_seed()

    text_train = load_feature_file(f"{TEXT_FEATURE_DIR}/text_sentiment_{SUBSET_TAG}_train.pt")
    text_dev = load_feature_file(f"{TEXT_FEATURE_DIR}/text_sentiment_{SUBSET_TAG}_dev.pt")
    text_test = load_feature_file(f"{TEXT_FEATURE_DIR}/text_sentiment_{SUBSET_TAG}_test.pt")

    audio_train = load_feature_file(f"{FEATURE_DIR}/audio_sentiment_{AUDIO_TAG}_train.pt")
    audio_dev = load_feature_file(f"{FEATURE_DIR}/audio_sentiment_{AUDIO_TAG}_dev.pt")
    audio_test = load_feature_file(f"{FEATURE_DIR}/audio_sentiment_{AUDIO_TAG}_test.pt")

    video_train = load_feature_file(f"{FEATURE_DIR}/video_sentiment_{SUBSET_TAG}_train.pt")
    video_dev = load_feature_file(f"{FEATURE_DIR}/video_sentiment_{SUBSET_TAG}_dev.pt")
    video_test = load_feature_file(f"{FEATURE_DIR}/video_sentiment_{SUBSET_TAG}_test.pt")

    face_train = load_feature_file(f"{FEATURE_DIR}/face_sentiment_{SUBSET_TAG}_train.pt")
    face_dev = load_feature_file(f"{FEATURE_DIR}/face_sentiment_{SUBSET_TAG}_dev.pt")
    face_test = load_feature_file(f"{FEATURE_DIR}/face_sentiment_{SUBSET_TAG}_test.pt")

    text_train_map = to_feature_map(text_train)
    text_dev_map = to_feature_map(text_dev)
    text_test_map = to_feature_map(text_test)

    audio_train_map = to_feature_map(audio_train)
    audio_dev_map = to_feature_map(audio_dev)
    audio_test_map = to_feature_map(audio_test)

    video_train_map = to_feature_map(video_train)
    video_dev_map = to_feature_map(video_dev)
    video_test_map = to_feature_map(video_test)

    face_train_map = to_feature_map(face_train)
    face_dev_map = to_feature_map(face_dev)
    face_test_map = to_feature_map(face_test)

    train_label_map = to_label_map(text_train)
    dev_label_map = to_label_map(text_dev)
    test_label_map = to_label_map(text_test)

    train_ids = get_common_ids(text_train_map, audio_train_map, video_train_map, face_train_map)
    dev_ids = get_common_ids(text_dev_map, audio_dev_map, video_dev_map, face_dev_map)
    test_ids = get_common_ids(text_test_map, audio_test_map, video_test_map, face_test_map)

    print(f"Common train ids: {len(train_ids)}")
    print(f"Common dev ids  : {len(dev_ids)}")
    print(f"Common test ids : {len(test_ids)}")

    if len(train_ids) == 0 or len(dev_ids) == 0 or len(test_ids) == 0:
        raise ValueError("No common ids found across modalities. Check feature filenames and uid alignment.")

    train_ds = FusionDataset(train_ids, text_train_map, audio_train_map, video_train_map, face_train_map, train_label_map)
    dev_ds = FusionDataset(dev_ids, text_dev_map, audio_dev_map, video_dev_map, face_dev_map, dev_label_map)
    test_ds = FusionDataset(test_ids, text_test_map, audio_test_map, video_test_map, face_test_map, test_label_map)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    sample_x, _ = train_ds[0]
    input_dim = sample_x.shape[0]
    print(f"Fusion input dim: {input_dim}")

    model = FusionMLP(input_dim=input_dim, num_classes=len(SENTIMENT2ID)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_dev_acc = 0.0
    best_path = f"{OUT_DIR}/best_fusion_sentiment.pt"

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        train_loss, train_acc, _ = evaluate_model(model, train_loader, criterion)
        dev_loss, dev_acc, _ = evaluate_model(model, dev_loader, criterion)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Dev   Loss: {dev_loss:.4f} | Dev   Acc: {dev_acc:.4f}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    print("\n========== FINAL RESULTS: FUSION SENTIMENT ==========\n")

    train_loss, train_acc, train_report = evaluate_model(model, train_loader, criterion)
    print("TRAIN")
    print(f"Loss: {train_loss:.4f}")
    print(f"Accuracy: {train_acc:.4f}")
    print(train_report)

    dev_loss, dev_acc, dev_report = evaluate_model(model, dev_loader, criterion)
    print("DEV")
    print(f"Loss: {dev_loss:.4f}")
    print(f"Accuracy: {dev_acc:.4f}")
    print(dev_report)

    test_loss, test_acc, test_report = evaluate_model(model, test_loader, criterion)
    print("TEST")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(test_report)

    save_metrics_json(train_loss, train_acc, dev_loss, dev_acc, test_loss, test_acc)

    print(f"\nAll outputs saved in: {OUT_DIR}")


if __name__ == "__main__":
    run()
