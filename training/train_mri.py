from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers, regularizers

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import (
    log_device_info,
    RANDOM_SEED,
    MRI_DATASET_DIR,
    IMAGE_SIZE,
    IMAGE_CHANNELS,
    BATCH_SIZE,
    LEARNING_RATE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    REDUCE_LR_MIN_LR,
    MRI_MODEL_TL_PATH,
    MRI_WEIGHTS_TL_PATH,
    MRI_CHECKPOINT_TL_PATH,
    MRI_CLASSES,
    SAVED_MODELS_DIR,
)

def build_mri_model_tl(
    input_shape: tuple = (*IMAGE_SIZE, IMAGE_CHANNELS),
    num_classes: int = len(MRI_CLASSES),
    learning_rate: float = LEARNING_RATE,
    fine_tune_last_n: int = 40,
) -> keras.Model:
    base_model = keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = True
    for layer in base_model.layers[:-fine_tune_last_n]:
        layer.trainable = False

    inputs = keras.Input(shape=input_shape, name="mri_input")
    x = layers.Lambda(lambda t: t * 255.0, name="undo_rescale")(inputs)
    x = layers.Lambda(keras.applications.efficientnet.preprocess_input, name="efficientnet_preprocess")(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(0.4, name="drop_fc1")(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="fc1")(x)
    x = layers.Dropout(0.3, name="drop_fc2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="BrainMRIClassifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate * 0.1),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
from utils.visual import plot_training_history, plot_confusion_matrix, print_classification_report, Timer

tf.random.set_seed(RANDOM_SEED)
EPOCHS             = 20     # number of training epochs
TRAIN_SAMPLE_LIMIT = None   # None = use the full dataset every epoch (was 3000)


# Data loading

def build_data_generators():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.10,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.15,
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_dir = str(MRI_DATASET_DIR / "Training")
    test_dir  = str(MRI_DATASET_DIR / "Testing")

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        seed=RANDOM_SEED,
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=RANDOM_SEED,
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


def build_callbacks():
    from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE,
        min_lr=REDUCE_LR_MIN_LR,
        verbose=1,
    )

    checkpoint = ModelCheckpoint(
        filepath=MRI_CHECKPOINT_TL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    return [reduce_lr, checkpoint]

# Evaluation

def evaluate_on_test(model, test_gen):

    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    idx_to_class = {v: k for k, v in test_gen.class_indices.items()}
    class_names  = [idx_to_class[i] for i in sorted(idx_to_class)]

    print_classification_report(y_true, y_pred, class_names)

    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        title="Brain MRI Classification - Confusion Matrix",
        save_path=str(SAVED_MODELS_DIR / "mri_confusion_matrix_tl.png"),
    )

def main():
    print("\n" + "=" * 60)
    print("  Brain MRI Tumour Classification - Training ")
    print("=" * 60)

    log_device_info()

    print("\n[Train] Loading datasets...")
    train_gen, val_gen, test_gen = build_data_generators()
    print(f"  Total train images : {train_gen.samples}")
    print(f"  Val   samples      : {val_gen.samples}")
    print(f"  Test  samples      : {test_gen.samples}")
    print(f"  Classes            : {train_gen.class_indices}")

    # Compute steps_per_epoch based on TRAIN_SAMPLE_LIMIT
    if TRAIN_SAMPLE_LIMIT is not None:
        steps_per_epoch = TRAIN_SAMPLE_LIMIT // BATCH_SIZE
        print(f"\n[Train] Sample limit  : {TRAIN_SAMPLE_LIMIT} images")
        print(f"[Train] Steps / epoch : {steps_per_epoch}  "
              f"(= {TRAIN_SAMPLE_LIMIT} / batch size {BATCH_SIZE})")
    else:
        steps_per_epoch = None
        print(f"\n[Train] Using full dataset ({train_gen.samples} images)")

    print(f"[Train] Epochs        : {EPOCHS}")

    model = build_mri_model_tl(learning_rate=LEARNING_RATE)
    model.summary()

    # Train — runs all EPOCHS, no early stopping
    print("\n[Train] Starting training...")
    with Timer():
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=build_callbacks(),
            verbose=1,
        )

    print("\n[Train] Saving model and weights...")
    model.save(MRI_MODEL_TL_PATH)
    model.save_weights(MRI_WEIGHTS_TL_PATH)
    print(f"  Model   -> {MRI_MODEL_TL_PATH}")
    print(f"  Weights -> {MRI_WEIGHTS_TL_PATH}")

    plot_training_history(
        history,
        title="Brain MRI Classification (Transfer Learning) - Training History",
        save_path=str(SAVED_MODELS_DIR / "mri_training_history_tl.png"),
    )

    print("\n[Train] Evaluating on test set...")
    evaluate_on_test(model, test_gen)

    print("\n[Train] Done.")


if __name__ == "__main__":
    main()