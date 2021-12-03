#!/usr/bin/python
"""
To be run in directory of all the image files.
Convert HU image to correct format.
"""

import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd

# from ..MachineLearning import config

original_img_dir = "/run/media/ant/Backup Plus/Anson/WorkAttachment/unzipped_images_png/"  # recursively walk the dir
assert original_img_dir != "", "No img dir"


def load(img_path):
    image = cv2.imread(img_path, -1)
    return image.astype("int32")


def convert(image):
    image -= 32768
    initial_shape = image.shape

    min_val, max_val = min(image.flatten()), max(image.flatten())
    new = map(lambda x: (x - min_val) / (max_val - min_val) * 255, image.flatten())

    new = list(new)
    new = np.array(new).reshape(initial_shape)
    return new


def view(img_arr):
    image = Image.fromarray(img_arr)
    image.show()


def save(img_arr, file_path):
    image = Image.fromarray(img_arr)
    if image.mode == "F":
        image = image.convert("RGB")
    image.save(file_path, format="png")
    print(f"saved {file_path}")


def main():
    # os.chdir(config.IMAGES_PATH)

    for dirpath, dirnames, filenames in os.walk(original_img_dir):
        # continue if no files
        if filenames == []:
            continue
        print(dirpath, dirnames[:3], filenames[:3])
        for fn in filenames:
            img = load(os.path.sep.join([dirpath, fn]))
            new_img = convert(img)
            view(img)
            # save(new_img, fn)
            view(new_img)
            print(fn)
            break

        break


if __name__ == "__main__":
    # fns = os.listdir(original_img_dir)
    # img = convert(load(os.path.sep.join([original_img_dir, fns[0]])))
    # view(img)
    main()
