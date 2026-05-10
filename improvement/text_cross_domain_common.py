import importlib.util
import os
from pathlib import Path
import re

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from assignment3_common import (
    CHECKINGFILES_DIR,
    DEFAULT_EPOCHS,
    DEFAULT_SEED,
    meld_subset_note,
    resolve_meld_subset_paths,
    save_json,
    set_seed,
)


_original_find_spec = importlib.util.find_spec


def _find_spec_without_soundfile(name, package=None):
    if name == "soundfile":
        return None
    return _original_find_spec(name, package)


importlib.util.find_spec = _find_spec_without_soundfile
try:
    from datasets import load_dataset
finally:
    importlib.util.find_spec = _original_find_spec


DAILYDIALOG_EMOTION_MAP = {
    0: "neutral",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "joy",
    5: "sadness",
    6: "surprise",
}
TWEETEVAL_SENTIMENT_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}
DEFAULT_KAGGLEHUB_SENTIMENT_DATASET = "kazanova/sentiment140"
SENTIMENT_TEXT_COLUMN_CANDIDATES = (
    "text",
    "sentence",
    "tweet",
    "content",
    "review",
    "comment",
    "message",
    "headline",
    "utterance",
)
SENTIMENT_LABEL_COLUMN_CANDIDATES = (
    "label",
    "sentiment",
    "Sentiment",
    "target",
    "polarity",
    "class",
    "score",
)
SENTIMENT_SPLIT_COLUMN_CANDIDATES = ("split", "set", "subset", "partition")
SENTIMENT_UID_COLUMN_CANDIDATES = ("uid", "id", "ids", "tweet_id", "review_id")
SENTIMENT_MANIFEST_NAME_HINTS = (
    "sentiment",
    "sent140",
    "twitter",
    "tweet",
    "review",
    "imdb",
    "train",
    "dataset",
)
SENTIMENT_STRING_LABEL_ALIASES = {
    "negative": "negative",
    "neg": "negative",
    "neutral": "neutral",
    "neu": "neutral",
    "positive": "positive",
    "pos": "positive",
}
SENTIMENT_NUMERIC_LABEL_SCHEMES = {
    "auto": {0: "negative", 2: "neutral", 4: "positive"},
    "tweeteval_3class": {0: "negative", 1: "neutral", 2: "positive"},
    "sentiment140": {0: "negative", 2: "neutral", 4: "positive"},
    "binary_neg0_pos1": {0: "negative", 1: "positive"},
    "binary_pos0_neg1": {0: "positive", 1: "negative"},
}
IEMOCAP_SESSION_SPLITS = {
    "session1": "train",
    "session2": "train",
    "session3": "train",
    "session4": "dev",
    "session5": "test",
}
IEMOCAP_TRANSCRIPT_RE = re.compile(r"^(?P<utt_id>\S+)\s+\[[^\]]+\]:\s*(?P<text>.*)$")
IEMOCAP_EVAL_RE = re.compile(r"^\[[^\]]+\]\s+(?P<utt_id>\S+)\s+(?P<label>[A-Za-z]+)\s+\[")
IEMOCAP_KAGGLE_DATASET = "sangayb/iemocap"
IEMOCAP_DEFAULT_ROOT_CANDIDATES = [
    CHECKINGFILES_DIR / "IEMOCAP",
    CHECKINGFILES_DIR / "IEMOCAP_full_release",
    CHECKINGFILES_DIR / "iemocap",
    CHECKINGFILES_DIR / "data" / "IEMOCAP",
    CHECKINGFILES_DIR / "data" / "IEMOCAP_full_release",
    CHECKINGFILES_DIR / "data" / "iemocap",
    CHECKINGFILES_DIR.parent / "IEMOCAP",
    CHECKINGFILES_DIR.parent / "IEMOCAP_full_release",
    CHECKINGFILES_DIR.parent / "iemocap",
    Path("/mnt/optimusmesh/IEMOCAP"),
    Path("/mnt/optimusmesh/IEMOCAP_full_release"),
    Path("/mnt/optimusmesh/iemocap"),
]
IEMOCAP_MANIFEST_SUFFIXES = {".csv", ".tsv", ".txt", ".jsonl", ".json"}
IEMOCAP_MANIFEST_NAME_HINTS = (
    "iemocap",
    "emotion",
    "label",
    "labels",
    "metadata",
    "utterance",
    "transcript",
)
IEMOCAP_LABEL_ALIASES = {
    "ang": "anger",
    "anger": "anger",
    "dis": "disgust",
    "disgust": "disgust",
    "fea": "fear",
    "fear": "fear",
    "fru": "__FRU__",
    "frustration": "__FRU__",
    "hap": "joy",
    "happy": "joy",
    "happiness": "joy",
    "joy": "joy",
    "exc": "joy",
    "excited": "joy",
    "excitement": "joy",
    "neu": "neutral",
    "neutral": "neutral",
    "sad": "sadness",
    "sadness": "sadness",
    "sur": "surprise",
    "surprise": "surprise",
    "oth": None,
    "other": None,
    "xxx": None,
    "unknown": None,
}
IEMOCAP_EMOTION_LABEL_TO_ID = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6,
}
IEMOCAP_EMOTION_TO_SENTIMENT = {
    "anger": "negative",
    "disgust": "negative",
    "fear": "negative",
    "joy": "positive",
    "neutral": "neutral",
    "sadness": "negative",
    "surprise": "positive",
}


def uid_from_meld_row(row):
    return f"dia{int(row['Dialogue_ID'])}_utt{int(row['Utterance_ID'])}"


def load_meld_frames(task_name, label_column, label_to_id):
    dataset_paths = resolve_meld_subset_paths(task_name)
    frames = {}
    for split_name, csv_path in dataset_paths.items():
        df = pd.read_csv(csv_path).copy()
        df["Utterance"] = df["Utterance"].fillna("").astype(str)
        df[label_column] = df[label_column].astype(str).str.strip().str.lower()
        unknown_labels = sorted(set(df[label_column]) - set(label_to_id))
        if unknown_labels:
            raise ValueError(f"{csv_path} contains unsupported labels: {unknown_labels}")
        frames[split_name] = pd.DataFrame(
            {
                "text": df["Utterance"],
                "label_name": df[label_column],
                "label_id": df[label_column].map(label_to_id),
                "uid": df.apply(uid_from_meld_row, axis=1),
                "source": "meld",
                "split": split_name,
            }
        )
    return dataset_paths, frames


def load_dailydialog_emotion_frames(label_to_id, dataset_name="daily_dialog"):
    split_map = {"train": "train", "dev": "validation", "test": "test"}
    frames = {}
    for split_name, source_split in split_map.items():
        dataset = load_dataset(dataset_name, split=source_split)
        rows = []
        for dialog_index, example in enumerate(dataset):
            utterances = example["dialog"]
            emotions = example["emotion"]
            for utterance_index, (utterance, emotion_id) in enumerate(zip(utterances, emotions)):
                mapped_label = DAILYDIALOG_EMOTION_MAP.get(int(emotion_id))
                if mapped_label is None:
                    continue
                rows.append(
                    {
                        "text": str(utterance or ""),
                        "label_name": mapped_label,
                        "label_id": label_to_id[mapped_label],
                        "uid": f"dd_{split_name}_dialog{dialog_index}_utt{utterance_index}",
                        "source": "daily_dialog",
                        "split": split_name,
                    }
                )
        frames[split_name] = pd.DataFrame(rows)
    return frames


def load_tweeteval_sentiment_frames(label_to_id, dataset_name="tweet_eval", dataset_config="sentiment"):
    split_map = {"train": "train", "dev": "validation", "test": "test"}
    frames = {}
    for split_name, source_split in split_map.items():
        dataset = load_dataset(dataset_name, dataset_config, split=source_split)
        rows = []
        for row_index, example in enumerate(dataset):
            mapped_label = TWEETEVAL_SENTIMENT_MAP.get(int(example["label"]))
            if mapped_label is None:
                continue
            rows.append(
                {
                    "text": str(example["text"] or ""),
                    "label_name": mapped_label,
                    "label_id": label_to_id[mapped_label],
                    "uid": f"tweet_{split_name}_{row_index}",
                    "source": "tweet_eval",
                    "split": split_name,
                }
            )
        frames[split_name] = pd.DataFrame(rows)
    return frames


def normalize_iemocap_label(raw_label, map_frustration_to):
    label_key = str(raw_label or "").strip().lower()
    mapped_label = IEMOCAP_LABEL_ALIASES.get(label_key)
    if mapped_label == "__FRU__":
        return None if map_frustration_to == "drop" else map_frustration_to
    if mapped_label is not None:
        return mapped_label
    return label_key if label_key else None


def resolve_iemocap_root(iemocap_root=None):
    def find_session_root(candidate_path):
        if not (candidate_path and candidate_path.exists() and candidate_path.is_dir()):
            return None
        if list(candidate_path.glob("Session*")):
            return candidate_path.resolve()
        for child_path in candidate_path.iterdir():
            if child_path.is_dir() and list(child_path.glob("Session*")):
                return child_path.resolve()
        return None

    candidates = []
    if iemocap_root:
        candidates.append(Path(iemocap_root).expanduser())
    env_root = os.getenv("IEMOCAP_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(IEMOCAP_DEFAULT_ROOT_CANDIDATES)

    for candidate in candidates:
        session_root = find_session_root(candidate)
        if session_root is not None:
            return session_root

    searched = [str(path) for path in candidates]
    raise FileNotFoundError(
        "Unable to locate a local IEMOCAP root. Pass --iemocap-root, set IEMOCAP_ROOT, "
        f"or place the dataset in one of these locations: {searched}"
    )


def download_dataset_with_kagglehub(dataset_handle, dataset_name):
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            f"kagglehub is required to download {dataset_name} automatically. Install it with `pip install kagglehub`."
        ) from exc

    download_path = Path(kagglehub.dataset_download(dataset_handle)).expanduser().resolve()
    print(f"Downloaded {dataset_name} via KaggleHub: {dataset_handle}")
    print(f"KaggleHub dataset path: {download_path}")
    return download_path


def download_iemocap_with_kagglehub(dataset_handle=IEMOCAP_KAGGLE_DATASET):
    return download_dataset_with_kagglehub(dataset_handle, "IEMOCAP")


def read_table_file(table_path):
    table_path = Path(table_path)
    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(table_path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(table_path, sep="\t")
    if suffix == ".jsonl":
        return pd.read_json(table_path, lines=True)
    if suffix == ".json":
        return pd.read_json(table_path)
    raise ValueError(f"Unsupported IEMOCAP manifest format: {table_path}")


def pick_first_existing_column(frame_df, candidate_names, field_name):
    for column_name in candidate_names:
        if column_name in frame_df.columns:
            return column_name
    raise ValueError(f"IEMOCAP manifest is missing a {field_name} column. Expected one of: {candidate_names}")


def normalize_iemocap_split(raw_value):
    split_key = str(raw_value or "").strip().lower()
    if split_key in {"train", "dev", "valid", "validation", "test"}:
        return "dev" if split_key in {"valid", "validation"} else split_key
    if split_key.startswith("session"):
        return IEMOCAP_SESSION_SPLITS.get(split_key)
    if split_key.isdigit():
        return IEMOCAP_SESSION_SPLITS.get(f"session{split_key}")
    return None


def discover_table_file(candidate_root, name_hints):
    candidate_root = Path(candidate_root)
    file_candidates = []
    for file_path in candidate_root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in IEMOCAP_MANIFEST_SUFFIXES:
            continue
        path_name = file_path.name.lower()
        score = sum(1 for hint in name_hints if hint in path_name)
        if score > 0:
            file_candidates.append((score, len(file_path.parts), file_path))

    if not file_candidates:
        return None

    file_candidates.sort(key=lambda item: (-item[0], item[1], str(item[2])))
    return file_candidates[0][2]


def discover_iemocap_manifest(candidate_root):
    return discover_table_file(candidate_root, IEMOCAP_MANIFEST_NAME_HINTS)


def normalize_sentiment_label(raw_label, label_scheme="auto"):
    if pd.isna(raw_label):
        return None
    label_key = str(raw_label).strip().lower()
    if not label_key:
        return None
    if label_key in SENTIMENT_STRING_LABEL_ALIASES:
        return SENTIMENT_STRING_LABEL_ALIASES[label_key]
    if label_scheme not in SENTIMENT_NUMERIC_LABEL_SCHEMES:
        raise ValueError(
            f"Unsupported sentiment label scheme: {label_scheme}. "
            f"Expected one of {sorted(SENTIMENT_NUMERIC_LABEL_SCHEMES)}"
        )
    if re.fullmatch(r"-?\d+(\.0+)?", label_key):
        label_id = int(float(label_key))
        return SENTIMENT_NUMERIC_LABEL_SCHEMES[label_scheme].get(label_id)
    return None


def normalize_generic_split(raw_value):
    split_key = str(raw_value or "").strip().lower()
    if split_key in {"train", "trn"}:
        return "train"
    if split_key in {"dev", "valid", "validation", "val"}:
        return "dev"
    if split_key in {"test", "tst"}:
        return "test"
    return None


def stratified_split_frame(frame_df, seed, train_fraction=0.8, dev_fraction=0.1):
    split_parts = {"train": [], "dev": [], "test": []}
    for label_name, group_df in frame_df.groupby("label_name", sort=False):
        shuffled = group_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        num_rows = len(shuffled)
        if num_rows == 1:
            split_parts["train"].append(shuffled)
            continue
        if num_rows == 2:
            split_parts["train"].append(shuffled.iloc[[0]])
            split_parts["test"].append(shuffled.iloc[[1]])
            continue

        dev_count = max(1, int(round(num_rows * dev_fraction)))
        test_count = max(1, int(round(num_rows * (1.0 - train_fraction - dev_fraction))))
        train_count = num_rows - dev_count - test_count
        while train_count < 1:
            if dev_count >= test_count and dev_count > 1:
                dev_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                break
            train_count = num_rows - dev_count - test_count

        split_parts["train"].append(shuffled.iloc[:train_count])
        split_parts["dev"].append(shuffled.iloc[train_count : train_count + dev_count])
        split_parts["test"].append(shuffled.iloc[train_count + dev_count :])

    return {
        split_name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=frame_df.columns)
        for split_name, parts in split_parts.items()
    }


def maybe_limit_rows(frame_df, max_rows, seed):
    if max_rows is None or len(frame_df) <= max_rows:
        return frame_df.reset_index(drop=True)
    return frame_df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def load_sentiment_manifest_frames(
    label_to_id,
    manifest_path,
    *,
    text_column=None,
    label_column=None,
    split_column=None,
    uid_column=None,
    label_scheme="auto",
    train_fraction=0.8,
    dev_fraction=0.1,
    seed=DEFAULT_SEED,
    max_train_rows=None,
    max_dev_rows=None,
    max_test_rows=None,
):
    frame_df = read_table_file(manifest_path).copy()
    text_column = text_column or pick_first_existing_column(
        frame_df, SENTIMENT_TEXT_COLUMN_CANDIDATES, "text"
    )
    label_column = label_column or pick_first_existing_column(
        frame_df, SENTIMENT_LABEL_COLUMN_CANDIDATES, "label"
    )

    if split_column is None:
        for candidate_name in SENTIMENT_SPLIT_COLUMN_CANDIDATES:
            if candidate_name in frame_df.columns:
                split_column = candidate_name
                break

    if uid_column is None:
        for candidate_name in SENTIMENT_UID_COLUMN_CANDIDATES:
            if candidate_name in frame_df.columns:
                uid_column = candidate_name
                break

    frame_df["text_value"] = frame_df[text_column].fillna("").astype(str).str.strip()
    frame_df["label_name"] = frame_df[label_column].apply(
        lambda raw_label: normalize_sentiment_label(raw_label, label_scheme=label_scheme)
    )
    frame_df = frame_df[
        frame_df["text_value"].ne("")
        & frame_df["label_name"].isin(label_to_id)
    ].copy()
    if frame_df.empty:
        raise ValueError(
            f"No usable sentiment rows were found in {manifest_path}. "
            "Check the label scheme or pass explicit column names."
        )

    if uid_column is None:
        frame_df["uid_value"] = [f"sentiment_manifest_{row_index}" for row_index in range(len(frame_df))]
    else:
        frame_df["uid_value"] = frame_df[uid_column].fillna("").astype(str).str.strip()
        frame_df.loc[frame_df["uid_value"].eq(""), "uid_value"] = [
            f"sentiment_manifest_{row_index}" for row_index in range(len(frame_df))
        ]

    if split_column is not None:
        frame_df["split_name"] = frame_df[split_column].apply(normalize_generic_split)
        frame_df = frame_df[frame_df["split_name"].isin({"train", "dev", "test"})].copy()
        split_frames = {
            split_name: frame_df[frame_df["split_name"] == split_name].copy()
            for split_name in ("train", "dev", "test")
        }
    else:
        split_frames = stratified_split_frame(
            frame_df[["text_value", "label_name", "uid_value"]].copy(),
            seed=seed,
            train_fraction=train_fraction,
            dev_fraction=dev_fraction,
        )

    limits = {"train": max_train_rows, "dev": max_dev_rows, "test": max_test_rows}
    frames = {}
    for split_name in ("train", "dev", "test"):
        split_df = split_frames[split_name].copy()
        split_df = maybe_limit_rows(split_df, limits[split_name], seed)
        frames[split_name] = pd.DataFrame(
            {
                "text": split_df["text_value"],
                "label_name": split_df["label_name"],
                "label_id": split_df["label_name"].map(label_to_id),
                "uid": split_df["uid_value"],
                "source": "kagglehub_sentiment",
                "split": split_name,
            }
        )
    return frames


def load_kagglehub_sentiment_frames(
    label_to_id,
    *,
    dataset_handle=DEFAULT_KAGGLEHUB_SENTIMENT_DATASET,
    manifest_path=None,
    text_column=None,
    label_column=None,
    split_column=None,
    uid_column=None,
    label_scheme="auto",
    train_fraction=0.8,
    dev_fraction=0.1,
    seed=DEFAULT_SEED,
    max_train_rows=None,
    max_dev_rows=None,
    max_test_rows=None,
):
    resolved_manifest = Path(manifest_path).expanduser().resolve() if manifest_path else None
    if resolved_manifest is None:
        download_root = download_dataset_with_kagglehub(dataset_handle, "sentiment dataset")
        resolved_manifest = discover_table_file(download_root, SENTIMENT_MANIFEST_NAME_HINTS)
        if resolved_manifest is None:
            raise FileNotFoundError(
                "No sentiment manifest file could be detected in the downloaded Kaggle dataset. "
                "Pass --source-manifest and explicit column names if needed."
            )
        print(f"Using KaggleHub sentiment manifest: {resolved_manifest}")
    elif not resolved_manifest.exists():
        raise FileNotFoundError(f"Sentiment manifest not found: {resolved_manifest}")

    return load_sentiment_manifest_frames(
        label_to_id=label_to_id,
        manifest_path=resolved_manifest,
        text_column=text_column,
        label_column=label_column,
        split_column=split_column,
        uid_column=uid_column,
        label_scheme=label_scheme,
        train_fraction=train_fraction,
        dev_fraction=dev_fraction,
        seed=seed,
        max_train_rows=max_train_rows,
        max_dev_rows=max_dev_rows,
        max_test_rows=max_test_rows,
    )


def load_iemocap_emotion_manifest_frames(label_to_id, manifest_path, map_frustration_to="anger"):
    frame_df = read_table_file(manifest_path).copy()
    text_column = pick_first_existing_column(frame_df, ["text", "utterance", "transcript", "sentence"], "text")
    label_column = pick_first_existing_column(frame_df, ["label", "emotion", "Emotion"], "label")
    uid_column = None
    for candidate_name in ("uid", "utterance_id", "id", "turn_id"):
        if candidate_name in frame_df.columns:
            uid_column = candidate_name
            break
    split_column = None
    for candidate_name in ("split", "set", "subset"):
        if candidate_name in frame_df.columns:
            split_column = candidate_name
            break
    if split_column is None:
        split_column = pick_first_existing_column(frame_df, ["session", "Session"], "split/session")

    frame_df["split_name"] = frame_df[split_column].apply(normalize_iemocap_split)
    frame_df["label_name"] = frame_df[label_column].apply(
        lambda raw_label: normalize_iemocap_label(raw_label, map_frustration_to)
    )
    frame_df["text_value"] = frame_df[text_column].fillna("").astype(str).str.strip()
    if uid_column is None:
        frame_df["uid_value"] = [f"iemocap_manifest_{row_index}" for row_index in range(len(frame_df))]
    else:
        frame_df["uid_value"] = frame_df[uid_column].fillna("").astype(str).str.strip()
    frame_df = frame_df[
        frame_df["split_name"].isin({"train", "dev", "test"})
        & frame_df["label_name"].isin(label_to_id)
        & frame_df["text_value"].ne("")
        & frame_df["uid_value"].ne("")
    ].copy()

    frames = {}
    for split_name in ("train", "dev", "test"):
        split_df = frame_df[frame_df["split_name"] == split_name].copy()
        frames[split_name] = pd.DataFrame(
            {
                "text": split_df["text_value"],
                "label_name": split_df["label_name"],
                "label_id": split_df["label_name"].map(label_to_id),
                "uid": split_df["uid_value"],
                "source": "iemocap",
                "split": split_name,
            }
        )
    return frames


def parse_iemocap_transcript_file(transcript_path):
    transcript_map = {}
    with open(transcript_path, "r", encoding="utf-8", errors="ignore") as file_obj:
        for line in file_obj:
            match = IEMOCAP_TRANSCRIPT_RE.match(line.strip())
            if not match:
                continue
            transcript_map[match.group("utt_id")] = match.group("text").strip()
    return transcript_map


def resolve_iemocap_dataset_inputs(
    iemocap_root=None,
    iemocap_manifest=None,
    iemocap_source="auto",
    kagglehub_dataset=IEMOCAP_KAGGLE_DATASET,
):
    if iemocap_manifest:
        manifest_path = Path(iemocap_manifest).expanduser().resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"IEMOCAP manifest not found: {manifest_path}")
        return {"mode": "manifest", "path": manifest_path}

    if iemocap_source not in {"auto", "local", "kagglehub"}:
        raise ValueError(f"Unsupported iemocap_source: {iemocap_source}")

    if iemocap_source in {"auto", "local"}:
        try:
            root_dir = resolve_iemocap_root(iemocap_root)
            print(f"Using local IEMOCAP root: {root_dir}")
            return {"mode": "raw_root", "path": root_dir}
        except FileNotFoundError:
            if iemocap_source == "local":
                raise

    download_root = download_iemocap_with_kagglehub(kagglehub_dataset)
    try:
        resolved_root = resolve_iemocap_root(download_root)
        print(f"Using KaggleHub IEMOCAP root: {resolved_root}")
        return {"mode": "raw_root", "path": resolved_root}
    except FileNotFoundError:
        manifest_path = discover_iemocap_manifest(download_root)
        if manifest_path is not None:
            print(f"Using KaggleHub IEMOCAP manifest: {manifest_path}")
            return {"mode": "manifest", "path": manifest_path}
        raise FileNotFoundError(
            "Downloaded IEMOCAP dataset was found, but no raw Session* root or manifest file could be detected. "
            f"Downloaded path: {download_root}"
        )


def load_iemocap_emotion_frames(
    label_to_id,
    iemocap_root=None,
    iemocap_manifest=None,
    map_frustration_to="anger",
    iemocap_source="auto",
    kagglehub_dataset=IEMOCAP_KAGGLE_DATASET,
):
    resolved_source = resolve_iemocap_dataset_inputs(
        iemocap_root=iemocap_root,
        iemocap_manifest=iemocap_manifest,
        iemocap_source=iemocap_source,
        kagglehub_dataset=kagglehub_dataset,
    )
    if resolved_source["mode"] == "manifest":
        return load_iemocap_emotion_manifest_frames(
            label_to_id=label_to_id,
            manifest_path=resolved_source["path"],
            map_frustration_to=map_frustration_to,
        )

    root_dir = resolved_source["path"]
    transcript_cache = {}
    split_rows = {"train": [], "dev": [], "test": []}

    for session_dir in sorted(root_dir.glob("Session*")):
        split_name = IEMOCAP_SESSION_SPLITS.get(session_dir.name.lower())
        if split_name is None:
            continue
        eval_dir = session_dir / "dialog" / "EmoEvaluation"
        transcript_dir = session_dir / "dialog" / "transcriptions"
        if not eval_dir.exists() or not transcript_dir.exists():
            continue

        for eval_path in sorted(eval_dir.glob("*.txt")):
            dialog_id = eval_path.stem
            transcript_path = transcript_dir / f"{dialog_id}.txt"
            if transcript_path not in transcript_cache:
                transcript_cache[transcript_path] = (
                    parse_iemocap_transcript_file(transcript_path) if transcript_path.exists() else {}
                )
            dialog_transcripts = transcript_cache[transcript_path]

            with open(eval_path, "r", encoding="utf-8", errors="ignore") as file_obj:
                for line in file_obj:
                    match = IEMOCAP_EVAL_RE.match(line.strip())
                    if not match:
                        continue
                    utt_id = match.group("utt_id")
                    mapped_label = normalize_iemocap_label(match.group("label"), map_frustration_to)
                    if mapped_label not in label_to_id:
                        continue
                    text_value = dialog_transcripts.get(utt_id, "").strip()
                    if not text_value:
                        continue
                    split_rows[split_name].append(
                        {
                            "text": text_value,
                            "label_name": mapped_label,
                            "label_id": label_to_id[mapped_label],
                            "uid": utt_id,
                            "source": "iemocap",
                            "split": split_name,
                        }
                    )

    frames = {split_name: pd.DataFrame(rows) for split_name, rows in split_rows.items()}
    empty_splits = [split_name for split_name, frame_df in frames.items() if frame_df.empty]
    if empty_splits:
        raise ValueError(
            f"IEMOCAP parsing produced empty splits for {empty_splits}. "
            "Verify the dataset root or provide a manifest via --iemocap-manifest."
        )
    return frames


def load_iemocap_sentiment_frames(
    label_to_id,
    *,
    iemocap_root=None,
    iemocap_manifest=None,
    map_frustration_to="anger",
    iemocap_source="auto",
    kagglehub_dataset=IEMOCAP_KAGGLE_DATASET,
):
    emotion_frames = load_iemocap_emotion_frames(
        label_to_id=IEMOCAP_EMOTION_LABEL_TO_ID,
        iemocap_root=iemocap_root,
        iemocap_manifest=iemocap_manifest,
        map_frustration_to=map_frustration_to,
        iemocap_source=iemocap_source,
        kagglehub_dataset=kagglehub_dataset,
    )
    frames = {}
    for split_name, frame_df in emotion_frames.items():
        sentiment_df = frame_df.copy()
        sentiment_df["label_name"] = sentiment_df["label_name"].map(IEMOCAP_EMOTION_TO_SENTIMENT)
        sentiment_df = sentiment_df[sentiment_df["label_name"].isin(label_to_id)].copy()
        sentiment_df["label_id"] = sentiment_df["label_name"].map(label_to_id)
        sentiment_df["source"] = "iemocap_pseudo_sentiment"
        frames[split_name] = sentiment_df[["text", "label_name", "label_id", "uid", "source", "split"]].reset_index(drop=True)

    empty_splits = [split_name for split_name, frame_df in frames.items() if frame_df.empty]
    if empty_splits:
        raise ValueError(
            f"IEMOCAP pseudo-sentiment mapping produced empty splits for {empty_splits}. "
            "Check the dataset layout or manifest."
        )
    return frames


class FrameDataset(Dataset):
    def __init__(self, frame_df, tokenizer, max_len):
        self.df = frame_df.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(row["label_id"]), dtype=torch.long),
            "uid": row["uid"],
        }


class TextTransferModel(nn.Module):
    def __init__(self, model_name, classifier_hidden_size, head_dropout, num_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.dropout = nn.Dropout(head_dropout)
        self.feature_proj = nn.Linear(self.encoder.config.hidden_size, classifier_hidden_size)
        self.activation = nn.GELU()
        self.classifier = nn.Linear(classifier_hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_feature = out.last_hidden_state[:, 0, :]
        feat = self.feature_proj(self.dropout(cls_feature))
        feat = self.activation(feat)
        feat = self.dropout(feat)
        logits = self.classifier(feat)
        return logits, feat


def build_loaders(frame_splits, tokenizer, batch_size, num_workers, max_len, train_shuffle=True):
    pin_memory = torch.cuda.is_available()
    datasets = {split_name: FrameDataset(frame_df, tokenizer, max_len) for split_name, frame_df in frame_splits.items()}
    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers, pin_memory=pin_memory),
        "dev": DataLoader(datasets["dev"], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        "train_eval": DataLoader(datasets["train"], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
    }


def build_class_weights(frame_df, label_ids):
    counts = frame_df["label_id"].value_counts().sort_index()
    total = len(frame_df)
    num_classes = len(label_ids)
    weights = []
    for label_id in label_ids:
        count = int(counts.get(label_id, 0))
        weights.append(0.0 if count <= 0 else total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float)


def evaluate_model(model, loader, criterion, device, label_ids, target_names):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(len(all_labels), 1)
    report_text = classification_report(
        all_labels,
        all_preds,
        labels=label_ids,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        all_labels,
        all_preds,
        labels=label_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, labels=label_ids, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_labels, all_preds, labels=label_ids, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_preds, labels=label_ids).tolist(),
        "classification_report": report_dict,
    }
    return metrics, report_text


def train_stage(
    *,
    model,
    loaders,
    stage_name,
    output_dir,
    device,
    lr,
    epochs,
    warmup_ratio,
    primary_metric,
    label_ids,
    target_names,
    class_weights=None,
):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = max(len(loaders["train"]) * epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    stage_dir = output_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    best_path = stage_dir / "best.pt"
    best_metric_value = float("-inf")
    history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_examples = 0

        for batch in loaders["train"]:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_examples += labels.size(0)

        train_epoch_loss = running_loss / max(running_examples, 1)
        train_epoch_acc = running_correct / max(running_examples, 1)
        dev_metrics, _ = evaluate_model(model, loaders["dev"], criterion, device, label_ids, target_names)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_epoch_loss,
                "train_accuracy": train_epoch_acc,
                "dev_loss": dev_metrics["loss"],
                "dev_accuracy": dev_metrics["accuracy"],
                "dev_macro_f1": dev_metrics["macro_f1"],
                "dev_weighted_f1": dev_metrics["weighted_f1"],
            }
        )

        print(f"\n[{stage_name}] Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_epoch_acc:.4f}")
        print(
            "Dev   Loss: "
            f"{dev_metrics['loss']:.4f} | Dev Acc: {dev_metrics['accuracy']:.4f} | "
            f"Dev Macro-F1: {dev_metrics['macro_f1']:.4f} | "
            f"Dev Weighted-F1: {dev_metrics['weighted_f1']:.4f}"
        )

        selection_metric = dev_metrics[primary_metric]
        if selection_metric > best_metric_value:
            best_metric_value = selection_metric
            torch.save(model.state_dict(), best_path)
            print(f"Saved best {stage_name} model to {best_path} using dev {primary_metric}={selection_metric:.4f}")

    try:
        state_dict = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(best_path, map_location=device)
    model.load_state_dict(state_dict)
    final_metrics = {}
    final_reports = {}
    for split_name, loader_name in (("train", "train_eval"), ("dev", "dev"), ("test", "test")):
        metrics, report = evaluate_model(model, loaders[loader_name], criterion, device, label_ids, target_names)
        final_metrics[split_name] = metrics
        final_reports[split_name] = report

    return {
        "best_path": best_path,
        "history": history,
        "best_metric_value": best_metric_value,
        "metrics": final_metrics,
        "reports": final_reports,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
    }


def evaluate_on_target_dataset(model, frame_splits, tokenizer, *, batch_size, num_workers, max_len, device, label_ids, target_names):
    loaders = build_loaders(
        frame_splits,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        max_len=max_len,
        train_shuffle=False,
    )
    criterion = nn.CrossEntropyLoss()
    result = {"metrics": {}, "reports": {}}
    print("\n========== ZERO SHOT ON MELD ==========\n")
    for split_name, loader_name in (("train", "train_eval"), ("dev", "dev"), ("test", "test")):
        metrics, report = evaluate_model(model, loaders[loader_name], criterion, device, label_ids, target_names)
        result["metrics"][split_name] = metrics
        result["reports"][split_name] = report
        print(split_name.upper())
        print(
            f"Loss: {metrics['loss']:.4f} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Macro-F1: {metrics['macro_f1']:.4f} | "
            f"Weighted-F1: {metrics['weighted_f1']:.4f}"
        )
        print(report)
    return result


def print_stage_summary(stage_name, result):
    print(f"\n========== {stage_name.upper()} ==========\n")
    for split_name in ("train", "dev", "test"):
        metrics = result["metrics"][split_name]
        print(split_name.upper())
        print(
            f"Loss: {metrics['loss']:.4f} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Macro-F1: {metrics['macro_f1']:.4f} | "
            f"Weighted-F1: {metrics['weighted_f1']:.4f}"
        )
        print(result["reports"][split_name])


def run_cross_domain_experiment(
    *,
    task_name,
    label_column,
    label_to_id,
    model_name,
    source_name,
    source_loader_fn,
    source_stage_name,
    output_dir,
    max_len=128,
    batch_size=16,
    source_batch_size=16,
    lr=2e-5,
    source_lr=2e-5,
    epochs=DEFAULT_EPOCHS,
    source_epochs=DEFAULT_EPOCHS,
    num_workers=2,
    classifier_hidden_size=512,
    head_dropout=0.3,
    primary_metric="macro_f1",
    warmup_ratio=0.1,
    use_class_weights=True,
    seed=DEFAULT_SEED,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_names = list(label_to_id.keys())
    label_ids = list(range(len(target_names)))

    meld_paths, meld_frames = load_meld_frames(task_name, label_column, label_to_id)
    source_frames = source_loader_fn(label_to_id)
    source_split_sizes = {split_name: len(frame_df) for split_name, frame_df in source_frames.items()}
    target_split_sizes = {split_name: len(frame_df) for split_name, frame_df in meld_frames.items()}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TextTransferModel(
        model_name=model_name,
        classifier_hidden_size=classifier_hidden_size,
        head_dropout=head_dropout,
        num_classes=len(label_to_id),
    ).to(device)

    print(f"Running cross-domain {task_name} on device: {device}")
    print(f"Source dataset: {source_name}")
    print(f"Target dataset: MELD subset ({meld_subset_note()})")
    print(f"Target MELD csvs: {meld_paths}")
    print(f"Using text backbone: {model_name}")
    print(f"Target subset sizes: {target_split_sizes}")
    print(f"Source split sizes: {source_split_sizes}")

    result_bundle = {
        "experiment_type": "cross_domain_text_transfer",
        "task": task_name,
        "source_dataset": source_name,
        "target_dataset": "meld_subset",
        "meld_subset_note": meld_subset_note(),
        "meld_paths": {split_name: str(path) for split_name, path in meld_paths.items()},
        "config": {
            "model_name": model_name,
            "max_len": max_len,
            "batch_size": batch_size,
            "source_batch_size": source_batch_size,
            "lr": lr,
            "source_lr": source_lr,
            "epochs": epochs,
            "source_epochs": source_epochs,
            "num_workers": num_workers,
            "classifier_hidden_size": classifier_hidden_size,
            "head_dropout": head_dropout,
            "primary_metric": primary_metric,
            "warmup_ratio": warmup_ratio,
            "use_class_weights": use_class_weights,
            "seed": seed,
        },
        "device": device,
        "source_split_sizes": source_split_sizes,
        "target_split_sizes": target_split_sizes,
    }

    source_loaders = build_loaders(
        source_frames,
        tokenizer=tokenizer,
        batch_size=source_batch_size,
        num_workers=num_workers,
        max_len=max_len,
    )
    source_class_weights = build_class_weights(source_frames["train"], label_ids) if use_class_weights else None
    source_result = train_stage(
        model=model,
        loaders=source_loaders,
        stage_name=source_stage_name,
        output_dir=output_dir,
        device=device,
        lr=source_lr,
        epochs=source_epochs,
        warmup_ratio=warmup_ratio,
        primary_metric=primary_metric,
        label_ids=label_ids,
        target_names=target_names,
        class_weights=source_class_weights,
    )
    print_stage_summary(source_stage_name, source_result)
    result_bundle[source_stage_name] = source_result

    zero_shot_result = evaluate_on_target_dataset(
        model,
        meld_frames,
        tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        max_len=max_len,
        device=device,
        label_ids=label_ids,
        target_names=target_names,
    )
    result_bundle["zero_shot_on_meld"] = zero_shot_result

    meld_loaders = build_loaders(
        meld_frames,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        max_len=max_len,
    )
    meld_class_weights = build_class_weights(meld_frames["train"], label_ids) if use_class_weights else None
    transfer_result = train_stage(
        model=model,
        loaders=meld_loaders,
        stage_name="meld_finetune",
        output_dir=output_dir,
        device=device,
        lr=lr,
        epochs=epochs,
        warmup_ratio=warmup_ratio,
        primary_metric=primary_metric,
        label_ids=label_ids,
        target_names=target_names,
        class_weights=meld_class_weights,
    )
    print_stage_summary("meld_finetune", transfer_result)
    result_bundle["meld_finetune"] = transfer_result

    metrics_path = output_dir / "metrics.json"
    save_json(metrics_path, result_bundle)
    print(f"\nSaved metrics json to: {metrics_path}")
    print(f"All outputs saved in: {output_dir}")
    return result_bundle
