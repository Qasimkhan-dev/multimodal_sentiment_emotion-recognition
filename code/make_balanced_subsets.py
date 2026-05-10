# make_balanced_subsets.py
import os
import math
import pandas as pd

BASE_PATH = "/mnt/optimusmesh/checkingfiles/meld_dataset/MELD-RAW/MELD.Raw"
OUT_DIR = "/mnt/optimusmesh/checkingfiles/meld_balanced"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_CSV = f"{BASE_PATH}/train/train_sent_emo.csv"
DEV_CSV   = f"{BASE_PATH}/dev_sent_emo.csv"
TEST_CSV  = f"{BASE_PATH}/test_sent_emo.csv"

def make_balanced_subset(input_csv, output_csv, label_col, frac=0.1, seed=42):
    df = pd.read_csv(input_csv).copy()
    df[label_col] = df[label_col].astype(str).str.strip()

    sampled_parts = []
    for label, group in df.groupby(label_col):
        n_take = max(1, math.floor(len(group) * frac))
        sampled_parts.append(group.sample(n=n_take, random_state=seed))

    out = pd.concat(sampled_parts, axis=0).sample(frac=1, random_state=seed).reset_index(drop=True)
    out.to_csv(output_csv, index=False)
    print(output_csv, len(out))
    print(out[label_col].value_counts())

# 20% emotion-balanced
make_balanced_subset(TRAIN_CSV, f"{OUT_DIR}/train_20_emotion.csv", "Emotion", frac=0.15)
make_balanced_subset(DEV_CSV,   f"{OUT_DIR}/dev_20_emotion.csv",   "Emotion", frac=0.15)
make_balanced_subset(TEST_CSV,  f"{OUT_DIR}/test_20_emotion.csv",  "Emotion", frac=0.15)

# 20% sentiment-balanced
make_balanced_subset(TRAIN_CSV, f"{OUT_DIR}/train_20_sentiment.csv", "Sentiment", frac=0.15)
make_balanced_subset(DEV_CSV,   f"{OUT_DIR}/dev_20_sentiment.csv",   "Sentiment", frac=0.15)
make_balanced_subset(TEST_CSV,  f"{OUT_DIR}/test_20_sentiment.csv",  "Sentiment", frac=0.15)
