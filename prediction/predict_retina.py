import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from tensorflow.keras.applications.efficientnet import preprocess_input
from config import RETINA_ACTIVE_MODEL_PATH as RETINA_MODEL_PATH, RETINA_CLASSES
from utils.preprocessing import load_image, preprocess_fracture_image as preprocess_retina_image


RETINA_STATIC_CONTEXT = {
    "Cataract": (
        "A cataract is a clouding of the eye's natural lens that affects vision, "
        "most often related to ageing but also linked to diabetes, prolonged "
        "steroid use, or trauma. Diabetic patients develop cataracts earlier and "
        "more frequently than the general population due to sorbitol accumulation "
        "in the lens. Symptoms include blurred or hazy vision, glare sensitivity, "
        "faded colour perception, and difficulty seeing at night. Diagnosis is "
        "confirmed by slit-lamp examination. Treatment is surgical lens replacement "
        "(phacoemulsification with intraocular lens implantation) once the cataract "
        "significantly impairs daily function."
    ),
    "Diabetic Retinopathy": (
        "Diabetic retinopathy is damage to the retinal blood vessels caused by "
        "chronically elevated blood glucose, and is a leading cause of blindness "
        "in working-age adults. It progresses from mild non-proliferative changes "
        "(microaneurysms, dot-and-blot haemorrhages) to proliferative disease with "
        "abnormal new vessel growth that can bleed or cause retinal detachment. "
        "Diabetic macular oedema may occur at any stage and threaten central "
        "vision. Diagnosis is confirmed by dilated fundus examination or retinal "
        "photography. Management includes tight glycaemic and blood-pressure "
        "control, anti-VEGF injections, laser photocoagulation, or vitrectomy "
        "depending on severity."
    ),
    "Glaucoma": (
        "Glaucoma is a group of eye conditions that damage the optic nerve, "
        "typically associated with elevated intraocular pressure, and is a "
        "leading cause of irreversible blindness worldwide. It is often "
        "asymptomatic in early stages, with gradual peripheral vision loss that "
        "can go unnoticed until advanced. Diabetic patients carry a higher risk, "
        "including neovascular glaucoma from proliferative diabetic retinopathy. "
        "Diagnosis involves intraocular pressure measurement, optic nerve "
        "examination, and visual field testing. Treatment includes pressure-"
        "lowering eye drops, laser therapy, or surgery, with the goal of "
        "preventing further nerve damage rather than reversing existing loss."
    ),
    "Normal": (
        "No signs of cataract, diabetic retinopathy, or glaucoma were identified "
        "in this fundus image. The retina, optic disc, and visible vasculature "
        "appear within normal limits based on AI analysis. Although the model did "
        "not detect disease, clinical correlation and regular screening remain "
        "important for diabetic patients, since early retinal changes can be "
        "subtle and progress without symptoms. Routine dilated eye examinations "
        "are recommended per standard diabetic care guidelines even with a normal "
        "result."
    ),
}


_retina_model = None


def _load_model():
    global _retina_model
    if _retina_model is None:
        from tensorflow.keras.models import load_model
        if not Path(RETINA_MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Saved model not found at '{RETINA_MODEL_PATH}'. "
                "Please train the model first: python training/train_retina.py"
            )
        print(f"[Retina] Loading model from {RETINA_MODEL_PATH} ...")
        _retina_model = load_model(
            RETINA_MODEL_PATH,
            custom_objects={"preprocess_input": preprocess_input},
            safe_mode=False,
        )
        print("[Retina] Model loaded.")
    return _retina_model


def _get_description(predicted_class: str) -> str:
    """Return the static clinical description for the predicted class."""
    return RETINA_STATIC_CONTEXT.get(predicted_class, "")


def predict_retina(image: Image.Image) -> Dict:

    model = _load_model()

    tensor = preprocess_retina_image(image)
    probs = model.predict(tensor, verbose=0)[0]   # shape (4,)

    class_idx = int(np.argmax(probs))
    predicted_class = RETINA_CLASSES[class_idx]
    confidence = float(probs[class_idx]) * 100.0

    probabilities = {
        cls: round(float(p) * 100, 2)
        for cls, p in zip(RETINA_CLASSES, probs)
    }

    description = _get_description(predicted_class)

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": probabilities,
        "description": description,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict diabetic retina disease class from a fundus image.")
    parser.add_argument("--image", required=True, help="Path to the fundus image file.")
    args = parser.parse_args()

    from utils.preprocessing import load_image as _load_img
    img = _load_img(args.image)
    result = predict_retina(img)

    print("\n" + "=" * 50)
    print(f"Prediction  : {result['predicted_class']}")
    print(f"Confidence  : {result['confidence']:.2f}%")
    print("\nClass probabilities:")
    for cls, prob in result["probabilities"].items():
        print(f"  {cls:<24} {prob:.2f}%")
    print("\nClinical Description:")
    print(result["description"])
    print("=" * 50)


if __name__ == "__main__":
    main()