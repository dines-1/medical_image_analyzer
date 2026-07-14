# Medical_image_analyzer
Medical imaging plays a central role in diagnosis across radiology, ophthalmology, and orthopedics. It helps to identifies the Brain tumor classes with brain MRI scan , fracture from x rays, Retina fundus screening for diabetics classification 
Medical Image Analyzer is a deep learning project for classifying three types of medical images:

- Brain MRI scans for tumour classification
- Retina fundus images for diabetic eye disease screening
- Bone X-ray images for fracture detection

The project includes training scripts, saved model artifacts, prediction modules, and a Gradio application for uploading images and viewing predictions with confidence scores and static clinical descriptions.

## Dataset

The datasets used in this project are taken from Kaggle and stored locally in the project directory. Each dataset is organized by class labels so that Keras image generators can load images directly from folders.

### Brain MRI Dataset

The MRI dataset is stored in `MRI/` with separate `Training/` and `Testing/` folders. It contains four classes:

| Split | Glioma | Meningioma | No Tumor | Pituitary | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 1,400 | 1,400 | 1,400 | 1,400 | 5,600 |
| Testing | 400 | 400 | 400 | 400 | 1,600 |
| Total | 1,800 | 1,800 | 1,800 | 1,800 | 7,200 |

This dataset is used to classify brain MRI images into glioma, meningioma, no tumour, and pituitary tumour categories.

### Diabetic Retina Dataset

The retina dataset is stored in `diebetics_retina/` and contains four folders:

| Class | Images |
| --- | ---: |
| Cataract | 1,038 |
| Diabetic Retinopathy | 1,098 |
| Glaucoma | 1,007 |
| Normal | 1,074 |
| Total | 4,217 |

This dataset is used for fundus image classification across cataract, diabetic retinopathy, glaucoma, and normal retina classes. The training script creates train, validation, and test splits from this directory.

### Bone Fracture Dataset

The fracture dataset is stored in `Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/` with predefined `train`, `val`, and `test` folders.

| Split | Fractured | Not Fractured | Total |
| --- | ---: | ---: | ---: |
| Train | 4,606 | 4,640 | 9,246 |
| Validation | 337 | 492 | 829 |
| Test | 238 | 268 | 506 |
| Total | 5,181 | 5,400 | 10,581 |

This dataset is used for binary X-ray classification: fractured or not fractured.

## Model Structure

All three models use transfer learning with EfficientNetB0 as the feature extractor. The ImageNet pretrained backbone is loaded without its original classification head, and the last 40 layers are fine-tuned while earlier layers remain frozen.

Common model pipeline:

1. Input image resized to `224 x 224 x 3`
2. Rescaling is undone inside the model before EfficientNet preprocessing
3. EfficientNetB0 backbone extracts image features
4. Global average pooling converts feature maps into a compact vector
5. Batch normalization stabilizes training
6. Dropout reduces overfitting
7. Dense classification head predicts the final class

Model-specific heads:

| Model | Output Classes | Final Layer | Loss Function |
| --- | ---: | --- | --- |
| Brain MRI | 4 | Dense softmax | Categorical crossentropy |
| Diabetic Retina | 4 | Dense softmax | Categorical crossentropy |
| Bone Fracture | 2 | Dense sigmoid | Binary crossentropy |

The MRI and retina models use a 256-unit dense layer before output. The fracture model uses a 128-unit dense layer before its binary output.

## Saved Model Accuracy

The saved model folder contains trained model files, weights, training-history plots, and confusion-matrix plots. The accuracy values below are read from the saved training-history plots in `saved_models/`.

| Task | Saved Model | Training Accuracy | Validation Accuracy | Source Artifact |
| --- | --- | ---: | ---: | --- |
| Brain MRI Classification | `mri_model_tl.keras` | About 97% | About 97% | `saved_models/mri_training_history_tl.png` |
| Diabetic Retina Screening | `retina_model_tl.keras` | About 92% | About 80% | `saved_models/retina_training_history_tl.png` |
| Bone Fracture Detection | `fracture_model.keras` | About 99.7% | About 99% | `saved_models/fracture_training_history.png` |

Additional saved evaluation artifacts:

| Task | Confusion Matrix |
| --- | --- |
| Brain MRI Classification | `saved_models/mri_confusion_matrix_tl.png` |
| Diabetic Retina Screening | `saved_models/retina_confusion_matrix_tl.png` |
| Bone Fracture Detection | `saved_models/fracture_confusion_matrix.png` |

## Project Structure

```text
Medical _Image_analyzer/
|-- app.py
|-- config.py
|-- README.md
|-- prediction/
|   |-- predict_mri.py
|   |-- predict_retina.py
|   `-- predict_fracture.py
|-- training/
|   |-- train_mri.py
|   |-- train_retina.py
|   `-- train_fracture.py
|-- utils/
|   |-- preprocessing.py
|   `-- visual.py
|-- saved_models/
|   |-- mri_model_tl.keras
|   |-- retina_model_tl.keras
|   |-- fracture_model.keras
|   |-- *_weights.weights.h5
|   |-- *_training_history*.png
|   `-- *_confusion_matrix*.png
|-- MRI/
|-- diebetics_retina/
`-- Bone_Fracture_Binary_Classification/
```

## Main Files

`app.py` runs the Gradio interface. It lets the user choose an image type, upload a medical image, and view the predicted class, confidence score, probability chart, and clinical description.

`config.py` stores shared paths, class labels, image size, training constants, and active model paths.

`training/train_mri.py`, `training/train_retina.py`, and `training/train_fracture.py` train the individual models and save model files, weights, training-history plots, and confusion matrices.

`prediction/predict_mri.py`, `prediction/predict_retina.py`, and `prediction/predict_fracture.py` load saved models and return prediction results. Each prediction result includes the predicted class, confidence, class probabilities, and a static clinical description.

## Running the Application

Install the required Python packages, then run:

```bash
python app.py
```

The application loads the active models configured in `config.py`:

| Task | Active Model Path |
| --- | --- |
| Brain MRI | `MRI_ACTIVE_MODEL_PATH` |
| Diabetic Retina | `RETINA_ACTIVE_MODEL_PATH` |
| Bone Fracture | `FRACTURE_ACTIVE_MODEL_PATH` |

## Training Models

Each model can be trained independently:

```bash
python training/train_mri.py
python training/train_retina.py
python training/train_fracture.py
```

Training outputs are saved in `saved_models/`.

## Prediction Output

Each prediction module returns a dictionary with this structure:

```python
{
    "predicted_class": "...",
    "confidence": 0.0,
    "probabilities": {"class_name": 0.0},
    "description": "Clinical description..."
}
```

The descriptions are static text stored in the prediction files. No external text-generation API is required for prediction.
