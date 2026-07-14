
import os
import random
import numpy as np
from pathlib import Path


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


BASE_DIR = Path(__file__).resolve().parent
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset roots (adjust to your local paths if needed)
FRACTURE_DATASET_DIR = BASE_DIR / "Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification"
MRI_DATASET_DIR = BASE_DIR / "MRI"

IMAGE_SIZE = (224,224)          # (height, width)
IMAGE_CHANNELS = 3               # RGB


BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# EarlyStopping
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MONITOR = "val_loss"

# ReduceLROnPlateau
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.3
REDUCE_LR_MIN_LR = 1e-6

FRACTURE_MODEL_PATH = str(SAVED_MODELS_DIR / "fracture_model.keras")
FRACTURE_WEIGHTS_PATH = str(SAVED_MODELS_DIR / "fracture_weights.weights.h5")
FRACTURE_CHECKPOINT_PATH = str(SAVED_MODELS_DIR / "fracture_best.keras")
FRACTURE_CLASSES = ["Fractured", "Not Fractured"]

# MRI model

MRI_MODEL_PATH = str(SAVED_MODELS_DIR / "mri_model.keras")
MRI_WEIGHTS_PATH = str(SAVED_MODELS_DIR / "mri_weights.weights.h5")
MRI_CHECKPOINT_PATH = str(SAVED_MODELS_DIR / "mri_best.keras")
MRI_CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


 
# --- Transfer-learning fracture model (trained side-by-side with the scratch CNN) ---


 
# --- Transfer-learning MRI model ---
MRI_MODEL_TL_PATH      = SAVED_MODELS_DIR / "mri_model_tl.keras"
MRI_WEIGHTS_TL_PATH    = SAVED_MODELS_DIR / "mri_weights_tl.weights.h5"
MRI_CHECKPOINT_TL_PATH = SAVED_MODELS_DIR / "mri_checkpoint_tl.keras"




# --- Transfer-learning MRI model ---

MRI_MODEL_TL_PATH      = SAVED_MODELS_DIR / "mri_model_tl.keras"
MRI_WEIGHTS_TL_PATH    = SAVED_MODELS_DIR / "mri_weights_tl.weights.h5"
MRI_CHECKPOINT_TL_PATH = SAVED_MODELS_DIR / "mri_checkpoint_tl.keras"

# --- Diabetic Retina Screening (4-class: Cataract, Diabetic Retinopathy,
#     Glaucoma, Normal) ---
# Point this at wherever your "diabetic_retina" folder actually lives.
RETINA_DATASET_DIR = Path(r"D:\Medical _Image_analyzer\diebetics_retina")
RETINA_CLASSES      = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]


RETINA_MODEL_TL_PATH      = SAVED_MODELS_DIR / "retina_model_tl.keras"
RETINA_WEIGHTS_TL_PATH    = SAVED_MODELS_DIR / "retina_weights_tl.weights.h5"
RETINA_CHECKPOINT_TL_PATH = SAVED_MODELS_DIR / "retina_checkpoint_tl.keras"


FRACTURE_ACTIVE_MODEL_PATH = FRACTURE_MODEL_PATH      # or FRACTURE_MODEL_TL_PATH
MRI_ACTIVE_MODEL_PATH      = MRI_MODEL_PATH           # or MRI_MODEL_TL_PATH
RETINA_ACTIVE_MODEL_PATH   = RETINA_MODEL_TL_PATH        # or RETINA_MODEL_TL_PATH
def log_device_info() -> None:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[Config] GPU detected: {[g.name for g in gpus]}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[Config] No GPU detected – running on CPU.")