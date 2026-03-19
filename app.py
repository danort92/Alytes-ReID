"""Gradio web interface for Alytes-ReID.

Usage:
    python app.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np

from src.utils.io import load_yaml

logger = logging.getLogger(__name__)

# Global model instances (loaded once at startup)
_detector = None
_segmenter = None
_reid_model = None
_database = None
_configs: dict[str, Any] = {}


def load_models() -> None:
    """Load all models at startup."""
    global _detector, _segmenter, _reid_model, _database, _configs

    _configs["detection"] = load_yaml(Path("config/detection.yaml"))
    _configs["preprocessing"] = load_yaml(Path("config/preprocessing.yaml"))
    _configs["reid"] = load_yaml(Path("config/reid.yaml"))

    # Models will be loaded here once trained
    # from src.detection.predict import load_detector
    # from src.segmentation.segment import ToadSegmenter
    # from src.reid.model import build_model
    # from src.reid.database import EmbeddingDatabase
    #
    # _detector = load_detector(Path("data/models/detection/best.pt"))
    # _segmenter = ToadSegmenter()
    # _reid_model = build_model(_configs["reid"])
    # _database = EmbeddingDatabase(...)

    logger.info("Models loaded (placeholder — train models first)")


def identify_toad(image: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Run full pipeline on an uploaded image.

    Args:
        image: Input image from Gradio (H, W, 3) in RGB.

    Returns:
        Tuple of (annotated_image, results_text).
    """
    if image is None:
        return None, "Please upload an image."

    if _detector is None:
        return image, (
            "Models not yet loaded.\n\n"
            "Train the detection and re-ID models first using:\n"
            "- notebooks/01_setup_and_training.ipynb\n\n"
            "Then place model files in data/models/ and restart the app."
        )

    # TODO: Full pipeline integration
    # 1. Detect toad
    # 2. Segment with SAM2
    # 3. Preprocess
    # 4. Match against database
    # 5. Return annotated image + results

    return image, "Pipeline not yet active. Train models first."


def register_new_individual(
    image: np.ndarray,
    individual_id: str,
    capture_date: str,
) -> str:
    """Register a new individual in the database.

    Args:
        image: Preprocessed toad image.
        individual_id: Unique ID for the new individual.
        capture_date: Date of capture.

    Returns:
        Status message.
    """
    if not individual_id.strip():
        return "Please enter an individual ID."

    if _reid_model is None or _database is None:
        return "Models not loaded. Train models first."

    # TODO: Extract embedding and add to database
    return f"Registered {individual_id} (capture date: {capture_date})"


def get_database_info() -> str:
    """Get summary of the current database.

    Returns:
        Formatted string with database statistics.
    """
    if _database is None:
        return "Database not loaded. Train models first."

    individuals = _database.get_individuals()
    lines = [f"Database: {_database.size} embeddings, {len(individuals)} individuals\n"]
    for ind_id in individuals:
        count = sum(1 for m in _database.metadata if m["individual_id"] == ind_id)
        lines.append(f"  {ind_id}: {count} sighting(s)")
    return "\n".join(lines)


def build_app() -> gr.Blocks:
    """Build the Gradio application.

    Returns:
        Gradio Blocks app.
    """
    with gr.Blocks(
        title="Alytes-ReID: Toad Identification",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# Alytes-ReID: Midwife Toad Identification\n"
            "Upload a dorsal photo of an *Alytes obstetricans* to identify the individual."
        )

        with gr.Tab("Identify"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        label="Upload toad photo",
                        type="numpy",
                    )
                    identify_btn = gr.Button("Identify", variant="primary")
                with gr.Column():
                    output_image = gr.Image(label="Detection & Segmentation")
                    output_text = gr.Textbox(
                        label="Results",
                        lines=10,
                        interactive=False,
                    )

            identify_btn.click(
                fn=identify_toad,
                inputs=[input_image],
                outputs=[output_image, output_text],
            )

        with gr.Tab("Register New Individual"):
            gr.Markdown("Register a new toad after confirming it's not in the database.")
            with gr.Row():
                reg_image = gr.Image(label="Toad photo", type="numpy")
                with gr.Column():
                    reg_id = gr.Textbox(label="Individual ID", placeholder="e.g., TOAD_042")
                    reg_date = gr.Textbox(label="Capture date", placeholder="YYYY-MM-DD")
                    reg_btn = gr.Button("Register", variant="primary")
                    reg_result = gr.Textbox(label="Status", interactive=False)

            reg_btn.click(
                fn=register_new_individual,
                inputs=[reg_image, reg_id, reg_date],
                outputs=[reg_result],
            )

        with gr.Tab("Database"):
            gr.Markdown("Overview of known individuals in the database.")
            db_info = gr.Textbox(label="Database Summary", lines=15, interactive=False)
            refresh_btn = gr.Button("Refresh")
            refresh_btn.click(fn=get_database_info, outputs=[db_info])

        with gr.Tab("Batch Processing"):
            gr.Markdown("Upload multiple images for batch identification.")
            batch_files = gr.File(
                label="Upload images",
                file_count="multiple",
                file_types=["image"],
            )
            batch_btn = gr.Button("Process Batch", variant="primary")
            batch_output = gr.Textbox(label="Results", lines=15, interactive=False)
            batch_csv = gr.File(label="Download CSV")

            # TODO: Implement batch processing
            batch_btn.click(
                fn=lambda x: ("Batch processing not yet implemented.", None),
                inputs=[batch_files],
                outputs=[batch_output, batch_csv],
            )

    return app


def main() -> None:
    """Launch the Gradio app."""
    logging.basicConfig(level=logging.INFO)
    load_models()

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
