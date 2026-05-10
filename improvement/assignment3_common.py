import json
import os
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


PACKAGE_DIR = Path(__file__).resolve().parent
CHECKINGFILES_DIR = PACKAGE_DIR.parent
OUTPUT_ROOT = PACKAGE_DIR / "outputs"
MELD_BALANCED_DIR = CHECKINGFILES_DIR / "meld_balanced"
MELD_FEATURE_DIR = CHECKINGFILES_DIR / "features"
MELD_TEXT_FEATURE_DIR = CHECKINGFILES_DIR / "meld_project" / "outputs" / "features"

MELD_SUBSET_FILE_TAG = "20"
MELD_SUBSET_PERCENT_LABEL = "15"
AUDIO_FEATURE_TAG = "15"
DEFAULT_EPOCHS = 5
DEFAULT_SEED = 42


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed=DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def to_serializable(value):
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.dim() == 0:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def save_json(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(to_serializable(payload), file_obj, indent=2)


def run_with_log_capture(output_dir, log_prefix, run_fn):
    output_dir = ensure_dir(output_dir)
    log_dir = ensure_dir(output_dir / "logs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{log_prefix}_{timestamp}.log"

    with open(log_path, "w", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            print(f"Saving log to: {log_path}")
            result = run_fn()
            print(f"Completed run. Log saved to: {log_path}")
            return result
        except Exception:
            traceback.print_exc()
            raise
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def meld_subset_note():
    return (
        "This workspace uses the class-balanced MELD subset files tagged '20', "
        f"which actually correspond to the {MELD_SUBSET_PERCENT_LABEL}% subset used for Assignment 3."
    )


def resolve_meld_subset_paths(task_name):
    task_suffix = task_name.lower().strip()
    if task_suffix not in {"emotion", "sentiment"}:
        raise ValueError(f"Unsupported MELD task: {task_name}")
    return {
        "train": MELD_BALANCED_DIR / f"train_{MELD_SUBSET_FILE_TAG}_{task_suffix}.csv",
        "dev": MELD_BALANCED_DIR / f"dev_{MELD_SUBSET_FILE_TAG}_{task_suffix}.csv",
        "test": MELD_BALANCED_DIR / f"test_{MELD_SUBSET_FILE_TAG}_{task_suffix}.csv",
    }
