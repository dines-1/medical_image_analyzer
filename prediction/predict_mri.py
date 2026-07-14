import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import MRI_ACTIVE_MODEL_PATH as MRI_MODEL_PATH, MRI_CLASSES
from utils.preprocessing import load_image, preprocess_mri_image


MRI_STATIC_CONTEXT = {
    "Glioma": (
        "Gliomas are brain tumours that originate in the glial cells of the brain or "
        "spine. They are the most common type of primary brain tumour in adults and are "
        "classified by grade (I-IV) according to the WHO classification. High-grade "
        "gliomas (grade III-IV, including glioblastoma multiforme) are aggressive and "
        "carry a poor prognosis, while low-grade gliomas (grade I-II) tend to grow more "
        "slowly. Symptoms depend on tumour location and may include headaches, seizures, "
        "cognitive changes, and focal neurological deficits. Diagnosis is confirmed by "
        "MRI with contrast and tissue biopsy. Treatment typically involves surgical "
        "resection, radiotherapy, and chemotherapy (e.g. temozolomide for GBM)."
    ),
    "Meningioma": (
        "Meningiomas are tumours that arise from the meninges – the membranes surrounding "
        "the brain and spinal cord. They are the most common benign intracranial tumour, "
        "accounting for approximately 30% of all primary brain tumours. Most meningiomas "
        "are slow-growing and asymptomatic for years. Symptoms, when present, include "
        "headaches, vision problems, hearing loss, or neurological deficits depending on "
        "location. The majority are benign (WHO grade I), though atypical (grade II) and "
        "anaplastic (grade III) variants exist. Treatment options include watchful waiting, "
        "stereotactic radiosurgery, or surgical resection, guided by tumour size, location, "
        "and symptom burden."
    ),
    "No Tumor": (
        "No tumour has been identified in this brain MRI scan. The parenchyma, ventricles, "
        "and surrounding structures appear within normal limits based on AI analysis. "
        "Normal brain MRI findings include symmetric grey and white matter signal, patent "
        "ventricles of appropriate size, and no mass effect or midline shift. "
        "Although the AI model did not detect a tumour, clinical and radiological "
        "correlation by a qualified neuroradiologist is always recommended for a definitive "
        "assessment. If symptoms persist, follow-up imaging or further clinical evaluation "
        "may be warranted."
    ),
    "Pituitary": (
        "Pituitary tumours (pituitary adenomas) are benign growths of the pituitary gland "
        "located at the base of the brain. They are classified as microadenomas (< 10 mm) "
        "or macroadenomas (>= 10 mm) and may be functional (hormone-secreting) or "
        "non-functional. Functional adenomas can cause hormonal syndromes such as "
        "acromegaly (excess GH), Cushing's disease (excess ACTH), or hyperprolactinaemia. "
        "Macroadenomas may compress the optic chiasm causing visual field defects. "
        "Diagnosis is confirmed by MRI of the pituitary with gadolinium contrast and "
        "endocrine laboratory tests. Treatment options include dopamine agonists, "
        "somatostatin analogues, surgery (trans-sphenoidal), and radiotherapy."
    ),
}


_mri_model = None


def _load_model():
    """Lazy-load the MRI model once and cache it."""
    global _mri_model
    if _mri_model is None:
        from tensorflow.keras.models import load_model
        if not Path(MRI_MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Saved model not found at '{MRI_MODEL_PATH}'. "
                "Please train the model first: python training/train_mri.py"
            )
        print(f"[MRI] Loading model from {MRI_MODEL_PATH} ...")
        _mri_model = load_model(MRI_MODEL_PATH, safe_mode=False)
        print("[MRI] Model loaded.")
    return _mri_model



def _get_description(predicted_class: str) -> str:
    """Return the static clinical description for the predicted class."""
    return MRI_STATIC_CONTEXT.get(predicted_class, "")


def predict_mri(image: Image.Image) -> Dict:

    model = _load_model()

    tensor = preprocess_mri_image(image)
    probs = model.predict(tensor, verbose=0)[0]   # shape (4,)

    class_idx = int(np.argmax(probs))
    predicted_class = MRI_CLASSES[class_idx]
    confidence = float(probs[class_idx]) * 100.0

    probabilities = {
        cls: round(float(p) * 100, 2)
        for cls, p in zip(MRI_CLASSES, probs)
    }

    description = _get_description(predicted_class)
    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": probabilities,
        "description": description,
    }



def main():
    parser = argparse.ArgumentParser(description="Predict brain tumour type from MRI.")
    parser.add_argument("--image", required=True, help="Path to the MRI image file.")
    args = parser.parse_args()

    from utils.preprocessing import load_image as _load_img
    img = _load_img(args.image)
    result = predict_mri(img)

    print("\n" + "=" * 50)
    print(f"Prediction  : {result['predicted_class']}")
    print(f"Confidence  : {result['confidence']:.2f}%")
    print("\nClass probabilities:")
    for cls, prob in result["probabilities"].items():
        print(f"  {cls:<20} {prob:.2f}%")
    print("\nClinical Description:")
    print(result["description"])
    print("=" * 50)


if __name__ == "__main__":
    main()
