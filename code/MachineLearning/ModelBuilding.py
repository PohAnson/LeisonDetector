import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, UpSampling2D
from tensorflow.keras.utils import plot_model
from tensorflow.python.eager.context import device
from tensorflow.keras.applications.vgg16 import VGG16

random.seed(123)
np.random.seed(123)

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], "GPU")
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != "GPU"
except:
    pass


def createModel():
    print("Creating model")
    model = Sequential(name="MyModel")
    model.add(tf.keras.layers.Input(shape=(512, 512, 3)))
    for i in VGG16(include_top=False, input_shape=(512, 512, 3)).layers[1:]:
        if i.name in ["block4_pool", "block5_pool"]:
            continue
        elif i.name == "block1_conv1":
            initial_weights = i.get_weights()
            weights = np.mean(initial_weights[0], axis=2)
            weights.resize((3, 3, 1, initial_weights[0].shape[-1]))
            weights = np.repeat(weights, 3, axis=2)
            new = Conv2D.from_config(i.get_config())
            model.add(new)
            model.layers[-1].set_weights([weights, initial_weights[1]])
        elif i.name in ["block1_conv2", "block2_conv1", "block2_conv2"]:
            initial_weights = i.get_weights()
            new = Conv2D.from_config(i.get_config())
            model.add(new)
            model.layers[-1].set_weights(initial_weights)
        elif "conv" in i.name:
            initial_weights = i.get_weights()
            new = Conv2D.from_config(i.get_config())
            model.add(new)
            model.layers[-1].set_weights(initial_weights)
        elif "pool" in i.name:
            new = layers.MaxPooling2D.from_config(i.get_config())
            model.add(new)

    classificationHead = Conv2D(512, (3, 3), strides=1, padding="same")(model.output)
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
    new = createModel()

    print("Finished Creation")
