import os
import subprocess
import pandas as pd

BASE_PATH = "/mnt/optimusmesh/checkingfiles/meld_dataset/MELD-RAW/MELD.Raw"
BALANCED_DIR = "/mnt/optimusmesh/checkingfiles/meld_balanced"

TRAIN_CSV = f"{BALANCED_DIR}/train_20_emotion.csv"
DEV_CSV   = f"{BALANCED_DIR}/dev_20_emotion.csv"
TEST_CSV  = f"{BALANCED_DIR}/test_20_emotion.csv"

TRAIN_VID = f"{BASE_PATH}/train/train_splits"
DEV_VID   = f"{BASE_PATH}/dev/dev_splits_complete"
TEST_VID  = f"{BASE_PATH}/test/output_repeated_splits_test"

AUDIO_BASE = "/mnt/optimusmesh/checkingfiles/meld_audio_15"
TRAIN_AUDIO = f"{AUDIO_BASE}/train"
DEV_AUDIO   = f"{AUDIO_BASE}/dev"
TEST_AUDIO  = f"{AUDIO_BASE}/test"

os.makedirs(TRAIN_AUDIO, exist_ok=True)
os.makedirs(DEV_AUDIO, exist_ok=True)
os.makedirs(TEST_AUDIO, exist_ok=True)

def uid_from_row(row):
    return f"dia{int(row['Dialogue_ID'])}_utt{int(row['Utterance_ID'])}"

def extract_split_audio(csv_path, video_root, audio_root):
    df = pd.read_csv(csv_path)
    total = len(df)
    done = 0
    missing = 0
    failed = 0

    for _, row in df.iterrows():
        uid = uid_from_row(row)
        video_path = os.path.join(video_root, f"{uid}.mp4")
        audio_path = os.path.join(audio_root, f"{uid}.wav")

        if not os.path.exists(video_path):
            missing += 1
            continue

        if os.path.exists(audio_path):
            done += 1
            continue

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ac", "1",
            "-ar", "16000",
            audio_path
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if result.returncode == 0 and os.path.exists(audio_path):
            done += 1
        else:
            failed += 1

    print(csv_path)
    print(f"Total rows      : {total}")
    print(f"Audio ready     : {done}")
    print(f"Missing videos  : {missing}")
    print(f"Failed extracts : {failed}")
    print("-" * 50)

extract_split_audio(TRAIN_CSV, TRAIN_VID, TRAIN_AUDIO)
extract_split_audio(DEV_CSV, DEV_VID, DEV_AUDIO)
extract_split_audio(TEST_CSV, TEST_VID, TEST_AUDIO)

print("All audio extraction finished.")
