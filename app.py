import sys
import io
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))


from config import MRI_CLASSES, RETINA_CLASSES
from prediction.predict_mri import predict_mri
from prediction.predict_retina import predict_retina
from prediction.predict_fracture import predict_fracture

ASSETS_DIR = PROJECT_ROOT / "assets"

IMAGE_TYPES = {
    "Brain Tumor MRI": {
        "predict_fn": predict_mri,
        "sample": ASSETS_DIR / "brain_mri.jpg",
        "palette": ["#2E86AB", "#A23B72"],
        "upload_label": "Upload Brain MRI Image",
        "help_text": (
            "Upload an axial brain MRI slice. The model classifies it as "
            "Glioma, Meningioma, No Tumor, or Pituitary."
        ),
    },
    "Diabetic Retina Screening": {
        "predict_fn": predict_retina,
        "sample": ASSETS_DIR / "retina.jpeg",
        "palette": ["#2E7D32", "#8D6E63"],
        "upload_label": "Upload Retina (Fundus) Image",
        "help_text": (
            "Upload a retina fundus photo. The model classifies it as "
            "Cataract, Diabetic Retinopathy, Glaucoma, or Normal."
        ),
    },
    "Bone Fracture Detection": {
        "predict_fn": predict_fracture,
        "sample": ASSETS_DIR / "fracturexray.png",
        "palette": ["#C0392B", "#34495E"],
        "upload_label": "Upload Bone X-Ray Image",
        "help_text": (
            "Upload a bone X-ray image. The model classifies it as "
            "Fractured or Not Fractured."
        ),
    },
}

DEFAULT_TYPE = "Brain Tumor MRI"


def _make_prob_bar(probabilities: dict, highlight_class: str, palette: list) -> Image.Image:
    classes = list(probabilities.keys())
    values = [probabilities[c] for c in classes]
    colours = [palette[0] if c == highlight_class else palette[1] for c in classes]

    fig, ax = plt.subplots(figsize=(5, max(2, len(classes) * 0.6)))
    bars = ax.barh(classes, values, color=colours, edgecolor="white", height=0.55)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left", fontsize=10, color="#DDDDDD",
        )

    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", fontsize=9, color="#BBBBBB")
    ax.tick_params(axis="y", labelsize=10, colors="#DDDDDD")
    ax.tick_params(axis="x", labelsize=8, colors="#BBBBBB")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.set_facecolor("#1A1D29")
    fig.patch.set_facecolor("#1A1D29")
    plt.tight_layout(pad=0.8)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def on_type_change(image_type: str):
    info = IMAGE_TYPES[image_type]
    return str(info["sample"]), info["help_text"]


def run_prediction(image_type: str, uploaded_image) -> Tuple:
    if uploaded_image is None:
        return "No image provided.", "", "Please upload an image first.", None

    info = IMAGE_TYPES[image_type]

    try:
        if isinstance(uploaded_image, np.ndarray):
            pil_img = Image.fromarray(uploaded_image.astype("uint8"), "RGB")
        else:
            pil_img = uploaded_image.convert("RGB")

        result = info["predict_fn"](pil_img)

        label = f"{result['predicted_class']}"
        confidence = f"{result['confidence']:.2f}%"
        description = result["description"]
        chart = _make_prob_bar(
            result["probabilities"],
            result["predicted_class"],
            info["palette"],
        )
        return label, confidence, description, chart

    except FileNotFoundError as exc:
        return "Model not found.", "", str(exc), None
    except Exception as exc:
        import traceback
        print(f"\n[app.py] {image_type} prediction error:")
        traceback.print_exc()
        return "Error during prediction.", "", f"An unexpected error occurred: {exc}", None

def build_interface():
    import gradio as gr

    custom_css = """
    body, .gradio-container {
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    }

    .app-title {
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        padding: 0.75rem 0 0.25rem 0;
        color: #111827;
    }
    .dark .app-title {
        color: #F3F4F6;
    }

    .quadrant {
        border-radius: 12px;
        padding: 1.45rem;
        height: 100%;
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
    }
    .dark .quadrant {
        background-color: #151925;
        border: 1px solid #262B3A;
    }

    .quadrant-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #1D4ED8;
    }
    .dark .quadrant-title {
        color: #93C5FD;
    }

    .help-text p {
        font-size: 0.85rem;
        line-height: 1.5;
        margin: 0.25rem 0 0.75rem 0;
        color: #4B5563;
    }
    .dark .help-text p {
        color: #B5B9C6;
    }

    .prediction-label textarea {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        border-left: 4px solid #6366F1 !important;
        border-radius: 6px !important;
        color: #111827 !important;
        background: #FFFFFF !important;
    }
    .dark .prediction-label textarea {
        color: #F3F4F6 !important;
        background: #1A1D29 !important;
    }

    .confidence-label textarea {
        font-size: 1.0rem !important;
        font-weight: 600 !important;
        color: #4B5563 !important;
        background: #FFFFFF !important;
    }
    .dark .confidence-label textarea {
        color: #B5B9C6 !important;
        background: #1A1D29 !important;
    }

    .description-box textarea {
        font-size: 0.88rem;
        line-height: 1.6;
        border-radius: 8px;
        color: #1F2937;
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
    }
    .dark .description-box textarea {
        color: #D1D5DB;
        background: #1A1D29 !important;
        border: 1px solid #262B3A !important;
    }

    .gr-button-primary {
        background-color: #6366F1 !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }

    .disclaimer {
        font-size: 0.78rem;
        text-align: center;
        padding: 1rem;
        margin-top: 1rem;
        color: #6B7280;
        border-top: 1px solid #E5E7EB;
    }
    .dark .disclaimer {
        border-top: 1px solid #262B3A;
    }
    """

    with gr.Blocks(
        theme=gr.themes.Base(
            primary_hue="indigo",
            neutral_hue="slate",
        ),
        css=custom_css,
        title="Medical Image Analyzer",
    ) as demo:

        gr.HTML('<div class="app-title">Medical Image Analyzer</div>')

        # ---- Top row: Type/Upload/Analyze (left)  |  Sample image (right) ----
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["quadrant"]):
                gr.HTML('<div class="quadrant-title">Type &amp; Upload</div>')
                image_type = gr.Dropdown(
                    choices=list(IMAGE_TYPES.keys()),
                    value=DEFAULT_TYPE,
                    label="Image Type",
                )
                help_text = gr.Markdown(
                    IMAGE_TYPES[DEFAULT_TYPE]["help_text"],
                    elem_classes=["help-text"],
                )
                uploaded_input = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    height=220,
                )
                check_btn = gr.Button("Analyze Image", variant="primary")

            with gr.Column(scale=1, elem_classes=["quadrant"]):
                gr.HTML('<div class="quadrant-title">Sample Image</div>')
                sample_display = gr.Image(
                    value=str(IMAGE_TYPES[DEFAULT_TYPE]["sample"]),
                    show_label=False,
                    interactive=False,
                    height=340,
                )

        # ---- Bottom row: Prediction/Confidence/Probabilities (left) | Description (right) ----
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["quadrant"]):
                gr.HTML('<div class="quadrant-title">Prediction &amp; Confidence</div>')
                with gr.Row():
                    prediction_out = gr.Textbox(
                        label="Prediction",
                        interactive=False,
                        elem_classes=["prediction-label"],
                    )
                    confidence_out = gr.Textbox(
                        label="Confidence",
                        interactive=False,
                        elem_classes=["confidence-label"],
                    )
                chart_out = gr.Image(
                    label="Probability Distribution",
                    height=240,
                )

            with gr.Column(scale=1, elem_classes=["quadrant"]):
                gr.HTML('<div class="quadrant-title">Clinical Description</div>')
                description_out = gr.Textbox(
                    show_label=False,
                    lines=16,
                    interactive=False,
                    elem_classes=["description-box"],
                )

        image_type.change(
            fn=on_type_change,
            inputs=image_type,
            outputs=[sample_display, help_text],
        )

        check_btn.click(
            fn=run_prediction,
            inputs=[image_type, uploaded_input],
            outputs=[prediction_out, confidence_out, description_out, chart_out],
            show_progress="full",
        )
    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        allowed_paths=[str(ASSETS_DIR)],
    )