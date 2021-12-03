import datetime
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

import config
from CustomLossFunction import dice_coef
from Generator import DataGenerator
from ModelBuilding import create_model

seed = 39562
random.seed(seed)
np.random.seed(seed)

weights_used = False
weight_path = r"output/models/bests/weight.02-1.60.h5"


def get_subset_names(df: pd.DataFrame, lesion_type: int, num: int):
    """Get the filenames of the lesion.

    Args:
        df (pd.DataFrame): Data frame containing with the fields \
             'File_name' and 'Coarse_lesion_type'
        lesion_type (int): the type to get
        num (int): Number of file to return

    Returns:
        list: list of filenames.
    """
    fns = list(df[df["Coarse_lesion_type"] == lesion_type]["File_name"])
    random.shuffle(fns)
    return fns[:num]


if __name__ == "__main__":
    # Disable all GPUS
    tf.config.set_visible_devices([], "GPU")
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != "GPU"
    print(config.ANNOTS_PATH)
    df = pd.read_csv(config.ANNOTS_PATH)
    df = df[df["Coarse_lesion_type"] != -1]  # exclude all the unlabelled data

    fns_list = []
    for i in range(1, 9):
        fns_list.extend(get_subset_names(df, i, 1))

    datagen = DataGenerator(
        fns_list,
        config.IMAGES_PATH,
        config.ANNOTS_PATH,
        batch_size=config.BATCH_SIZE,
        targets=["Coarse_lesion_type", "Bounding_boxes"],
    )

    if weights_used:
        model = create_model()
        print("Loading weights")
        model.load_weights(weight_path)

        LR = config.INIT_LR
        opt = tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=1)

        losses = {
            "class_label": tf.keras.losses.categorical_crossentropy,
            "bounding_box": dice_coef,
        }
        model.compile(optimizer=opt, loss=losses, metrics=["acc"])

    else:
        model = create_model()
        model = keras.models.load_model(
            os.path.sep.join(
                [config.BASE_DIR, "output", "models", "20201220", "latest_model"]
            ),
            custom_objects={"dice_coef": dice_coef},
            compile=True,
        )

    cbs = []
    if config.SAVE_ON_EPOCH:
        cbs.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath="./output/models/bests/weight-"
                         "{epoch:02d}-{loss:.2f}.h5",
                save_weights_only=True,
                monitor="loss",
            )
        )
    if config.TENSORBOARD_LOGGING:
        cbs.append(
            tf.keras.callbacks.TensorBoard(
                histogram_freq=1,
                profile_batch=(2, 5),
                update_freq="batch",
                log_dir=os.path.sep.join(
                    [
                        config.config.LOG_DIR,
                        "fit",
                        datetime.datetime.now().strftime("%Y%m%d-%H%M"),
                    ],
                ),
            )
        )

    print(model.summary())
    H = model.fit(x=datagen, epochs=config.NUM_EPOCHS,
                  workers=16, max_queue_size=4)

    print("[INFO] saving object detector model...")
    model.save(config.MODEL_PATH)
    # plot the model training history
    N = config.NUM_EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.title("Bounding Box Regression Loss on Training Set")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    print("end")
