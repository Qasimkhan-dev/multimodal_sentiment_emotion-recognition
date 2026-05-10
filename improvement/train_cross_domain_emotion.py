import argparse

from assignment3_common import OUTPUT_ROOT, run_with_log_capture
from text_cross_domain_common import load_iemocap_emotion_frames, run_cross_domain_experiment


OUTPUT_DIR = OUTPUT_ROOT / "cross_domain_emotion_iemocap"
EMOTION2ID = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-domain emotion transfer from local IEMOCAP to the 15% MELD subset."
    )
    parser.add_argument(
        "--iemocap-source",
        choices=["auto", "local", "kagglehub"],
        default="kagglehub",
        help="Where to load IEMOCAP from. `kagglehub` downloads sangayb/iemocap automatically.",
    )
    parser.add_argument("--iemocap-root", default=None, help="Path to the local raw IEMOCAP root containing Session1-Session5.")
    parser.add_argument(
        "--iemocap-manifest",
        default=None,
        help="Optional CSV/TSV/JSONL manifest with split/text/label/uid columns for IEMOCAP utterances.",
    )
    parser.add_argument(
        "--kagglehub-dataset",
        default="sangayb/iemocap",
        help="KaggleHub dataset handle used when --iemocap-source kagglehub is selected.",
    )
    parser.add_argument(
        "--map-frustration-to",
        choices=["anger", "neutral", "drop"],
        default="anger",
        help="How to handle IEMOCAP frustration labels when mapping to MELD emotion classes.",
    )
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--source-epochs", type=int, default=5)
    parser.add_argument(
        "--primary-metric",
        choices=["accuracy", "macro_f1", "weighted_f1"],
        default="macro_f1",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_with_log_capture(
        output_dir=OUTPUT_DIR,
        log_prefix="cross_domain_emotion_iemocap",
        run_fn=lambda: run_cross_domain_experiment(
            task_name="emotion",
            label_column="Emotion",
            label_to_id=EMOTION2ID,
            model_name=args.model_name,
            source_name="IEMOCAP Emotion",
            source_loader_fn=lambda label_to_id: load_iemocap_emotion_frames(
                label_to_id=label_to_id,
                iemocap_root=args.iemocap_root,
                iemocap_manifest=args.iemocap_manifest,
                map_frustration_to=args.map_frustration_to,
                iemocap_source=args.iemocap_source,
                kagglehub_dataset=args.kagglehub_dataset,
            ),
            source_stage_name="iemocap_pretrain",
            output_dir=OUTPUT_DIR,
            epochs=args.epochs,
            source_epochs=args.source_epochs,
            primary_metric=args.primary_metric,
        ),
    )
