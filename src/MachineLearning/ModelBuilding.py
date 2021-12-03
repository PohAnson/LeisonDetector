import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D, Dense, Flatten, UpSampling2D
from tensorflow.keras.utils import plot_model

random.seed(123)
np.random.seed(123)

# Disable all GPUS
tf.config.set_visible_devices([], "GPU")
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != "GPU"


def create_model(base=True):
    """Create model

    Args:
        base (bool, optional): Whether to use random plain model or \
            load pretrained VGG16 weight. \
        Defaults to True.

    Returns:
        tensorflow.Sequential: the model created.
    """
    print(f"Creating model ({'base' if base else 'transferred'})")
    model = Sequential(name="MyModel")
    model.add(tf.keras.layers.Input(shape=(512, 512, 3)))
    if base:
        # Manually create entire model.
        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(
            layers.MaxPool2D(
                pool_size=(2, 2), strides=(2, 2), data_format="channels_last"
            )
        )
        model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(
            layers.MaxPool2D(
                pool_size=(2, 2), strides=(2, 2), data_format="channels_last"
            )
        )
        model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
        model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
        model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
        model.add(
            layers.MaxPool2D(
                pool_size=(2, 2), strides=(2, 2), data_format="channels_last"
            )
        )
        model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
        model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    else:
        vgg = VGG16(include_top=False, input_shape=(512, 512, 3))
        for layer in vgg.layers[1:]:
            if layer.name in ["block4_pool", "block5_pool"]:
                continue
            elif layer.name == "block1_conv1":
                initial_weights = layer.get_weights()
                weights = np.mean(initial_weights[0], axis=2)
                weights.resize((3, 3, 1, initial_weights[0].shape[-1]))
                weights = np.repeat(weights, 3, axis=2)
                new = Conv2D.from_config(layer.get_config())
                model.add(new)
                model.layers[-1].set_weights([weights, initial_weights[1]])
            elif (
                layer.name in ["block1_conv2", "block2_conv1", "block2_conv2"]
                or "conv" in layer.name
            ):
                initial_weights = layer.get_weights()
                new = Conv2D.from_config(layer.get_config())
                model.add(new)
                model.layers[-1].set_weights(initial_weights)
            elif "pool" in layer.name:
                new = layers.MaxPooling2D.from_config(layer.get_config())
                model.add(new)

    classificationHead = Conv2D(512, (3, 3), strides=1, padding="same")(
        model.output
    )
    classificationHead = Conv2D(512, (5, 5), strides=1, padding="same")(
        classificationHead
    )
    classificationHead = Flatten()(classificationHead)
    classificationHead = Dense(512, activation="relu")(classificationHead)
    classificationHead = Dense(256, activation="relu")(classificationHead)
    classificationHead = Dense(32, activation="relu")(classificationHead)
    classificationHead = Dense(8, activation="sigmoid", name="class_label")(
        classificationHead
    )

    bboxHead = Conv2D(512, (3, 3), strides=1, padding="same")(model.output)
    bboxHead = UpSampling2D(size=(2, 2))(bboxHead)
    bboxHead = Dense(256, activation="relu")(bboxHead)
    bboxHead = UpSampling2D(size=(2, 2))(bboxHead)
    bboxHead = Dense(128, activation="relu")(bboxHead)
    bboxHead = UpSampling2D(size=(2, 2))(bboxHead)
    bboxHead = Dense(64, activation="relu")(bboxHead)
    bboxHead = Dense(32, activation="relu")(bboxHead)
    bboxHead = Dense(1, activation="sigmoid", name="bounding_box")(bboxHead)

    model = tf.keras.models.Model(
        inputs=model.input, outputs=[classificationHead, bboxHead]
    )
    plot_model(model, show_shapes=True, to_file="model.png")
    print(model.summary())
    return model


if __name__ == "__main__":
    new = create_model()
    print("Finished Creation")
