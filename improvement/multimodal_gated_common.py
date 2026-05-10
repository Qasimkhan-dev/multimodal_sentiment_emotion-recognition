import os

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from assignment3_common import (
    AUDIO_FEATURE_TAG,
    DEFAULT_EPOCHS,
    DEFAULT_SEED,
    MELD_FEATURE_DIR,
    MELD_SUBSET_FILE_TAG,
    MELD_TEXT_FEATURE_DIR,
    meld_subset_note,
    save_json,
    set_seed,
)


def load_feature_file(path):
    return torch.load(path, map_location="cpu")


def feature_path(task_name, modality, split_name):
    if modality == "text":
        return MELD_TEXT_FEATURE_DIR / f"text_{task_name}_{MELD_SUBSET_FILE_TAG}_{split_name}.pt"
    if modality == "audio":
        return MELD_FEATURE_DIR / f"audio_{task_name}_{AUDIO_FEATURE_TAG}_{split_name}.pt"
    return MELD_FEATURE_DIR / f"{modality}_{task_name}_{MELD_SUBSET_FILE_TAG}_{split_name}.pt"


def to_feature_map(data):
    return {uid: feat for uid, feat in zip(data["ids"], data["features"])}


def to_label_map(data):
    return {uid: label for uid, label in zip(data["ids"], data["labels"])}


def get_common_ids(*feature_maps):
    common_ids = set(feature_maps[0].keys())
    for feature_map in feature_maps[1:]:
        common_ids &= set(feature_map.keys())
    return sorted(common_ids)


class FusionFeatureDataset(Dataset):
    def __init__(self, ids, feature_maps, label_map, modalities):
        self.ids = ids
        self.feature_maps = feature_maps
        self.label_map = label_map
        self.modalities = modalities

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        uid = self.ids[idx]
        item = {"uid": uid, "label": torch.tensor(self.label_map[uid], dtype=torch.long)}
        for modality in self.modalities:
            item[modality] = self.feature_maps[modality][uid].float()
        return item


class ModalityProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GatedAttentionFusion(nn.Module):
    def __init__(self, input_dims, modalities, hidden_dim, num_classes, dropout=0.3, modality_dropout=0.15):
        super().__init__()
        self.modalities = list(modalities)
        self.modality_dropout = modality_dropout
        self.projectors = nn.ModuleDict(
            {
                modality: ModalityProjector(input_dims[modality], hidden_dim, dropout)
                for modality in self.modalities
            }
        )
        gate_hidden_dim = max(hidden_dim // 2, 32)
        self.gates = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(hidden_dim, gate_hidden_dim),
                    nn.GELU(),
                    nn.Linear(gate_hidden_dim, 1),
                )
                for modality in self.modalities
            }
        )
        self.modality_embeddings = nn.Parameter(torch.randn(len(self.modalities), hidden_dim) * 0.02)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _sample_modality_mask(self, batch_size, num_modalities, device):
        if (not self.training) or self.modality_dropout <= 0.0:
            return torch.ones(batch_size, num_modalities, dtype=torch.bool, device=device)

        keep_mask = torch.rand(batch_size, num_modalities, device=device) > self.modality_dropout
        empty_rows = torch.where(~keep_mask.any(dim=1))[0]
        for row_index in empty_rows.tolist():
            keep_index = torch.randint(0, num_modalities, (1,), device=device).item()
            keep_mask[row_index, keep_index] = True
        return keep_mask

    def forward(self, batch_features):
        projected = []
        gate_logits = []

        for modality_index, modality in enumerate(self.modalities):
            features = batch_features[modality]
            modality_repr = self.projectors[modality](features) + self.modality_embeddings[modality_index]
            projected.append(modality_repr)
            gate_logits.append(self.gates[modality](modality_repr))

        projected = torch.stack(projected, dim=1)
        gate_logits = torch.cat(gate_logits, dim=1)
        keep_mask = self._sample_modality_mask(
            batch_size=projected.size(0),
            num_modalities=projected.size(1),
            device=projected.device,
        )
        masked_gate_logits = gate_logits.masked_fill(~keep_mask, -1e4)
        weights = torch.softmax(masked_gate_logits, dim=1)

        weighted_sum = (weights.unsqueeze(-1) * projected).sum(dim=1)
        keep_mask_float = keep_mask.unsqueeze(-1).float()
        mean_active = (projected * keep_mask_float).sum(dim=1) / keep_mask_float.sum(dim=1).clamp(min=1.0)
        fused = weighted_sum + mean_active
        logits = self.classifier(fused)
        return logits, fused, weights


def evaluate_model(model, loader, criterion, device, modalities, label_ids, target_names):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_labels = []
    all_preds = []
    modality_weight_sums = torch.zeros(len(modalities), dtype=torch.float64)
    gate_entropy_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            features = {modality: batch[modality].to(device) for modality in modalities}
            labels = batch["label"].to(device)

            logits, _, weights = model(features)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)
            preds = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

            modality_weight_sums += weights.sum(dim=0).double().cpu()
            batch_entropy = -(weights * torch.log(weights.clamp(min=1e-8))).sum(dim=1)
            gate_entropy_sum += batch_entropy.sum().item()

    avg_loss = total_loss / max(total_examples, 1)
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
        "average_gate_entropy": gate_entropy_sum / max(total_examples, 1),
        "average_modality_weights": {
            modality: float(modality_weight_sums[index].item() / max(total_examples, 1))
            for index, modality in enumerate(modalities)
        },
    }
    return metrics, report_text


def run_multimodal_gated_experiment(
    *,
    task_name,
    label_to_id,
    modalities,
    output_dir,
    hidden_dim=256,
    dropout=0.3,
    modality_dropout=0.15,
    batch_size=32,
    lr=1e-3,
    epochs=DEFAULT_EPOCHS,
    weight_decay=1e-4,
    seed=DEFAULT_SEED,
    primary_metric="macro_f1",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_names = list(label_to_id.keys())
    label_ids = list(range(len(target_names)))
    splits = ["train", "dev", "test"]

    raw_data = {
        modality: {split_name: load_feature_file(feature_path(task_name, modality, split_name)) for split_name in splits}
        for modality in modalities
    }
    feature_maps = {
        modality: {split_name: to_feature_map(raw_data[modality][split_name]) for split_name in splits}
        for modality in modalities
    }
    label_maps = {
        split_name: to_label_map(raw_data[modalities[0]][split_name])
        for split_name in splits
    }
    common_ids = {
        split_name: get_common_ids(*[feature_maps[modality][split_name] for modality in modalities])
        for split_name in splits
    }

    print(f"Running multimodal gated {task_name} on device: {device}")
    print(f"Modalities: {modalities}")
    print(meld_subset_note())
    for split_name in splits:
        print(f"Common {split_name} ids: {len(common_ids[split_name])}")

    if any(len(common_ids[split_name]) == 0 for split_name in splits):
        raise ValueError("No common ids found across modalities for at least one split.")

    datasets = {
        split_name: FusionFeatureDataset(
            ids=common_ids[split_name],
            feature_maps={modality: feature_maps[modality][split_name] for modality in modalities},
            label_map=label_maps[split_name],
            modalities=modalities,
        )
        for split_name in splits
    }
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=False)
    dev_loader = DataLoader(datasets["dev"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)

    sample_item = datasets["train"][0]
    input_dims = {modality: int(sample_item[modality].numel()) for modality in modalities}
    print(f"Input dimensions: {input_dims}")

    model = GatedAttentionFusion(
        input_dims=input_dims,
        modalities=modalities,
        hidden_dim=hidden_dim,
        num_classes=len(label_to_id),
        dropout=dropout,
        modality_dropout=modality_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_metric_value = float("-inf")
    best_path = output_dir / f"best_multimodal_gated_{task_name}.pt"
    history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_examples = 0

        for batch in train_loader:
            features = {modality: batch[modality].to(device) for modality in modalities}
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_examples += labels.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()

        train_epoch_loss = running_loss / max(running_examples, 1)
        train_epoch_acc = running_correct / max(running_examples, 1)
        dev_metrics, _ = evaluate_model(model, dev_loader, criterion, device, modalities, label_ids, target_names)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_epoch_loss,
                "train_accuracy": train_epoch_acc,
                "dev_loss": dev_metrics["loss"],
                "dev_accuracy": dev_metrics["accuracy"],
                "dev_macro_f1": dev_metrics["macro_f1"],
                "dev_weighted_f1": dev_metrics["weighted_f1"],
                "dev_average_gate_entropy": dev_metrics["average_gate_entropy"],
            }
        )

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_epoch_acc:.4f}")
        print(
            "Dev   Loss: "
            f"{dev_metrics['loss']:.4f} | Dev Acc: {dev_metrics['accuracy']:.4f} | "
            f"Dev Macro-F1: {dev_metrics['macro_f1']:.4f} | "
            f"Dev Weighted-F1: {dev_metrics['weighted_f1']:.4f}"
        )
        print(f"Dev modality weights: {dev_metrics['average_modality_weights']}")

        selection_metric = dev_metrics[primary_metric]
        if selection_metric > best_metric_value:
            best_metric_value = selection_metric
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path} using dev {primary_metric}={selection_metric:.4f}")

    model.load_state_dict(torch.load(best_path, map_location=device))

    final_metrics = {}
    final_reports = {}
    for split_name, loader in {
        "train": train_eval_loader,
        "dev": dev_loader,
        "test": test_loader,
    }.items():
        metrics, report = evaluate_model(model, loader, criterion, device, modalities, label_ids, target_names)
        final_metrics[split_name] = metrics
        final_reports[split_name] = report

    payload = {
        "experiment_type": "multimodal_gated",
        "task": task_name,
        "modalities": modalities,
        "meld_subset_file_tag": MELD_SUBSET_FILE_TAG,
        "meld_subset_percent_label": "15",
        "audio_feature_tag": AUDIO_FEATURE_TAG,
        "subset_note": meld_subset_note(),
        "device": device,
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "modality_dropout": modality_dropout,
        "weight_decay": weight_decay,
        "seed": seed,
        "primary_metric": primary_metric,
        "input_dims": input_dims,
        "common_id_counts": {split_name: len(common_ids[split_name]) for split_name in splits},
        "best_checkpoint": best_path,
        "best_metric_value": best_metric_value,
        "history": history,
        "metrics": final_metrics,
    }
    save_json(output_dir / "metrics.json", payload)

    print("\n========== FINAL RESULTS ==========\n")
    for split_name in splits:
        metrics = final_metrics[split_name]
        print(split_name.upper())
        print(
            f"Loss: {metrics['loss']:.4f} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Macro-F1: {metrics['macro_f1']:.4f} | "
            f"Weighted-F1: {metrics['weighted_f1']:.4f}"
        )
        print(f"Average modality weights: {metrics['average_modality_weights']}")
        print(final_reports[split_name])

    print(f"\nSaved metrics json to: {output_dir / 'metrics.json'}")
    print(f"All outputs saved in: {output_dir}")
    return payload
