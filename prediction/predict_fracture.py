from pathlib import Path
import sys
from xml.parsers.expat import model

import numpy as np
from PIL import Image
import keras
from keras.applications.efficientnet import preprocess_input

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import IMAGE_SIZE, FRACTURE_CLASSES, FRACTURE_ACTIVE_MODEL_PATH

_model = None

DESCRIPTIONS = {
    "Fractured": (
        "The model detected patterns consistent with a bone fracture in the "
        "X-ray image. This is not a diagnosis — please consult a radiologist "
        "or orthopedic specialist to confirm."
    ),
    "Not Fractured": (
        "The model did not detect patterns consistent with a fracture in the "
        "X-ray image. This is not a diagnosis — if you have ongoing pain or "
        "concerns, consult a doctor."
    ),
}


def _load_model():
    global _model
    if _model is None:
        model_path = Path(FRACTURE_ACTIVE_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"Fracture model not found at {model_path}")
        _model = keras.models.load_model(
            model_path,
            custom_objects={"preprocess_input": preprocess_input},
            safe_mode=False,
        )
    return _model


def predict_fracture(pil_image: Image.Image) -> dict:
    model = _load_model()

    img = pil_image.resize(IMAGE_SIZE)
    arr = np.array(img).astype("float32") / 255.0  # model's own Lambda undoes this (*255) then preprocesses
    arr = np.expand_dims(arr, axis=0)

    prob_not_fractured = float(model.predict(arr, verbose=0)[0][0])
    prob_fractured = 1.0 - prob_not_fractured

    predicted_class = "Fractured" if prob_fractured >= 0.5 else "Not Fractured"
    confidence = max(prob_fractured, prob_not_fractured) * 100

    probabilities = {
         "Fractured": prob_fractured * 100,
         "Not Fractured": prob_not_fractured * 100,
}

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "description": DESCRIPTIONS[predicted_class],
    }