from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import IMAGE_SIZE


# Generic loader

def load_image(image_path: str) -> Image.Image:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        img = Image.open(path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot open image '{image_path}': {exc}") from exc
    return img


# Fracture preprocessing

def preprocess_fracture_image(
    image: Image.Image,
    target_size: Tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    img = image.resize((target_size[1], target_size[0]))  # PIL: (W, H)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# MRI preprocessing

def preprocess_mri_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (128,128),
) -> np.ndarray:
    img = image.resize((target_size[1], target_size[0]))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)