import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
import tensorflow.keras as keras
# 38024, 94261
seed = 94261
random.seed(seed)
np.random.seed(seed)

import config
from CustomLossFunction import DiceLoss, dice_coef
from Generator import DataGenerator
from ModelBuilding import CreateModel

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


df = pd.read_csv(config.ANNOTS_PATH)
df = df[df['Coarse_lesion_type'] != -1]

def subset(df, lesion_type, num):
    fns = list(df[df['Coarse_lesion_type'] == lesion_type]['File_name'])
    random.shuffle(fns)
    return fns[:num]

fns_list = []
for i in range(1,9):
    fns_list.extend(subset(df, i, 5))

filenames1 = fns_list[:]

datagen = DataGenerator(filenames1, config.IMAGES_PATH, config.ANNOTS_PATH, batch_size=config.BATCH_SIZE, targets=['Coarse_lesion_type', 'Bounding_boxes'])

# model = keras.models.load_model(os.path.sep.join([config.BASE_MODEL_PATH, ""]) ,compile=False, custom_objects={'DiceLoss':DiceLoss})

model = CreateModel()
print('Loading weights')
# model.load_weights(r'output/models/bests/model.04-7.95.h5')

losses = {
    "class_label": tf.keras.losses.categorical_crossentropy,
    "bounding_box": dice_coef,
}

print('Compiling')
LR = config.INIT_LR
opt = tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=1)
opt = tf.keras.optimizers.SGD(learning_rate=LR)
model.compile(optimizer=opt, loss=losses, metrics=['acc'])

cbs =[
    tf.keras.callbacks.ModelCheckpoint(filepath='./output/models/bests/model.{epoch:02d}-{loss:.2f}.h5', save_weights_only=True, monitor='loss'),
    tf.keras.callbacks.TensorBoard(histogram_freq=1, profile_batch=(2,10), update_freq='batch', log_dir=f'./logs/fit/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'),
]

print(model.summary())
H = model.fit(x=datagen, epochs=config.NUM_EPOCHS, workers=16,  max_queue_size=4, callbacks=cbs)

print("[INFO] saving object detector model...")
model.save(config.MODEL_PATH)
# plot the model training history
N = config.NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)
