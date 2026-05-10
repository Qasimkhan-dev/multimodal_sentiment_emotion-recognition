import argparse

from assignment3_common import OUTPUT_ROOT, run_with_log_capture
from text_cross_domain_common import load_iemocap_sentiment_frames, run_cross_domain_experiment


OUTPUT_DIR = OUTPUT_ROOT / "cross_domain_sentiment_iemocap"
SENTIMENT2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-domain sentiment transfer from IEMOCAP-derived pseudo-sentiment to the 15% MELD subset."
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
        help="Optional CSV/TSV/JSONL manifest with split/text/emotion/uid columns for IEMOCAP utterances.",
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
        help="How to handle IEMOCAP frustration labels before converting emotions to sentiment.",
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
        log_prefix="cross_domain_sentiment_iemocap",
        run_fn=lambda: run_cross_domain_experiment(
            task_name="sentiment",
            label_column="Sentiment",
            label_to_id=SENTIMENT2ID,
            model_name=args.model_name,
            source_name="IEMOCAP Pseudo-Sentiment",
            source_loader_fn=lambda label_to_id: load_iemocap_sentiment_frames(
                label_to_id=label_to_id,
                iemocap_root=args.iemocap_root,
                iemocap_manifest=args.iemocap_manifest,
                map_frustration_to=args.map_frustration_to,
                iemocap_source=args.iemocap_source,
                kagglehub_dataset=args.kagglehub_dataset,
            ),
            source_stage_name="iemocap_pseudo_sentiment_pretrain",
            output_dir=OUTPUT_DIR,
            epochs=args.epochs,
            source_epochs=args.source_epochs,
            primary_metric=args.primary_metric,
        ),
    )
