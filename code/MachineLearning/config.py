# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 5e-6
NUM_EPOCHS = 1
BATCH_SIZE = 4

# Saving with each epoch?
SAVE_ON_EPOCH = False

# Logging performance using tensorboard?
TENSORBOARD_LOGGING = False

# import the necessary packages
import os
import datetime
from decouple import config

# define the base path to the input dataset and then use it to derive
# the path to the images directory and annotation CSV file

# Check which platform is it running on to get to the folder
BASE_DATA_PATH = config("BASE_DATA_PATH")
BASE_DIR = config("BASE_DIR")

IMAGES_PATH = os.path.sep.join([BASE_DATA_PATH, "HU_min"])
ANNOTS_PATH = os.path.sep.join([BASE_DATA_PATH, "DL_info.csv"])

# Base output dir
BASE_OUTPUT = os.path.sep.join([BASE_DIR, "output"])
if not os.path.exists(BASE_OUTPUT):
    os.makedirs(BASE_OUTPUT)


# Model outputs
BASE_MODEL_PATH = os.path.sep.join(
    [BASE_OUTPUT, "models", f"{datetime.datetime.now().strftime('%Y%m%d')}"]
)
MODEL_PATH = os.path.sep.join([BASE_MODEL_PATH, f"latest_model"])
MODEL_ARCHITECTURE = os.path.sep.join(
    [
        BASE_MODEL_PATH,
        "..",
        f"architecture{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.h5",
    ]
)
MODEL_WEIGHTS = os.path.sep.join(
    [
        BASE_MODEL_PATH,
        "models",
        f"weights{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    ]
)

LOG_DIR = os.path.sep.join([BASE_DIR, "logs"])
