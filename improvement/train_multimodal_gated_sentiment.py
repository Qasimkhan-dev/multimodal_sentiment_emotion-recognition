from assignment3_common import OUTPUT_ROOT, run_with_log_capture
from multimodal_gated_common import run_multimodal_gated_experiment


OUTPUT_DIR = OUTPUT_ROOT / "multimodal_gated_sentiment"
MODALITIES = ["text", "audio", "face", "video"]
SENTIMENT2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


if __name__ == "__main__":
    run_with_log_capture(
        output_dir=OUTPUT_DIR,
        log_prefix="multimodal_gated_sentiment",
        run_fn=lambda: run_multimodal_gated_experiment(
            task_name="sentiment",
            label_to_id=SENTIMENT2ID,
            modalities=MODALITIES,
            output_dir=OUTPUT_DIR,
            epochs=5,
        ),
    )
