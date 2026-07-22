
import tensorflow as tf
from tensorflow.keras import layers, models



def residual_block(x, filters, stride=1):
    """Basic ResNet-style block: conv-bn-relu -> conv-bn, + skip, then relu."""
    shortcut = x

    y = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)

    y = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    out = layers.Add()([y, shortcut])
    out = layers.ReLU()(out)
    return out


def dense_block(x, num_layers, growth_rate):
    for _ in range(num_layers):
        y = layers.BatchNormalization()(x)
        y = layers.ReLU()(y)
        y = layers.Conv2D(4 * growth_rate, 1, padding="same", use_bias=False)(y)  # bottleneck
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        y = layers.Conv2D(growth_rate, 3, padding="same", use_bias=False)(y)
        x = layers.Concatenate()([x, y])
    return x


def transition_layer(x, compression=0.5):
    """DenseNet transition: 1x1 conv (channel compression) + 2x2 avg pool."""
    filters = int(x.shape[-1] * compression)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    return x



def build_fracture_model(input_shape=(320, 320, 1), num_classes=2, dropout=0.4):
    inputs = layers.Input(shape=input_shape)

    # Stem
    x = layers.Conv2D(32, 7, strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    stage_filters = [32, 64, 128, 256]
    for i, filters in enumerate(stage_filters):
        stride = 1 if i == 0 else 2  # downsample at the start of stages 2-4
        x = residual_block(x, filters, stride=stride)
        x = residual_block(x, filters, stride=1)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="fracture_resnet18_narrow")
    return model

def build_brain_tumor_model(input_shape=(224, 224, 3), num_classes=4, dropout=0.5):
    inputs = layers.Input(shape=input_shape)

    # Stem
    x = layers.Conv2D(16, 3, strides=1, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    stage_filters = [16, 32, 64]
    for i, filters in enumerate(stage_filters):
        stride = 1 if i == 0 else 2
        x = residual_block(x, filters, stride=stride)
        x = residual_block(x, filters, stride=1)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="brain_tumor_resnet_narrow")
    return model


def build_retina_model(input_shape=(448, 448, 3), num_classes=4, growth_rate=12, dropout=0.5):
    inputs = layers.Input(shape=input_shape)

    # Stem
    x = layers.Conv2D(24, 3, strides=1, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    block_layers = [4, 4, 4]  # layers per dense block
    for i, num_layers in enumerate(block_layers):
        x = dense_block(x, num_layers=num_layers, growth_rate=growth_rate)
        if i != len(block_layers) - 1: 
            x = transition_layer(x, compression=0.5)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="retina_densenet_narrow")
    return model

if __name__ == "__main__":
    fracture_model = build_fracture_model()
    fracture_model.summary()

    brain_model = build_brain_tumor_model()
    brain_model.summary()

    retina_model = build_retina_model()
    retina_model.summary()