# import the necessary packages
import os
import datetime

# define the base path to the input dataset and then use it to derive
# the path to the images directory and annotation CSV file

# LINUX
BASE_DATA_PATH = "/shared/WorkAttachment/Datas"
BASE_DIR = '/shared/WorkAttachment/codes/code6'

#WINDOWS
BASE_DATA_PATH = "D:\WorkAttachment\Datas"
BASE_DIR = 'D:\WorkAttachment\codes\code6'

BASE_DATA_PATH = os.path.sep.join(['..', '..', 'Datas'])
BASE_DIR = '.'

# IMAGES_PATH = os.path.sep.join([BASE_DATA_PATH, "categorised_images_HU"])
IMAGES_PATH = os.path.sep.join([BASE_DATA_PATH, "HU"])
ANNOTS_PATH = os.path.sep.join([BASE_DATA_PATH, "DL_info.csv"])

# define the path to the base output directory
BASE_OUTPUT = os.path.sep.join([BASE_DIR, "output"])
if not os.path.exists(BASE_OUTPUT):
    os.makedirs(BASE_OUTPUT)

# os.path.sep.join([BASE_OUTPUT, "detector.h5"])
# define the path to the output serialized model, model training plot,
# and testing image filenames
BASE_MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "models", f"{datetime.datetime.now().strftime('%Y%m%d')}"])
MODEL_PATH = os.path.sep.join([BASE_MODEL_PATH, f"detector{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"])
MODEL_ARCHITECTURE = os.path.sep.join([BASE_MODEL_PATH, f"architecture{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.h5"])
MODEL_WEIGHTS = os.path.sep.join([BASE_MODEL_PATH,'models', f"weights{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"])

PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
LOG_DIR = os.path.sep.join([BASE_DIR, "logs"])
DEBUG_DIR = os.path.sep.join([LOG_DIR, "debug"])

# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 2
BATCH_SIZE = 8


# Targets are mapped starting from 0
NUM_TYPE_MAPPING = {'-1': 'not_val_test','1': 'bone', '2': 'abdomen', '3': 'mediastinum', '4': 'liver',
                    '5': 'lung', '6': 'kidney', '7': 'soft tissue', '8': 'pelvis'}
