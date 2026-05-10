from assignment3_common import OUTPUT_ROOT, run_with_log_capture
from multimodal_gated_common import run_multimodal_gated_experiment


OUTPUT_DIR = OUTPUT_ROOT / "multimodal_gated_emotion"
MODALITIES = ["text", "audio", "face", "video"]
EMOTION2ID = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6,
}


if __name__ == "__main__":
    run_with_log_capture(
        output_dir=OUTPUT_DIR,
        log_prefix="multimodal_gated_emotion",
        run_fn=lambda: run_multimodal_gated_experiment(
            task_name="emotion",
            label_to_id=EMOTION2ID,
            modalities=MODALITIES,
            output_dir=OUTPUT_DIR,
            epochs=5,
        ),
    )
