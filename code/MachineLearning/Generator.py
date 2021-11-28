from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
import numpy as np
from tensorflow.keras.utils import Sequence
import pandas as pd
import random
import matplotlib.pyplot as plt

random.seed(123)
np.random.seed(123)


class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(
        self,
        filenames,
        image_path,
        csvpath=None,
        targets=None,
        dim=(512, 512),
        batch_size=32,
        to_fit=True,
        shuffle=True,
    ):
        """Initialisation

        Args:
            filenames (list): name of the file relative to image_path
            image_path (str): base image paths
            csvpath (str): path of the csv file
            targets (list): list of targets/labels required
            dim (tuple, optional): image dimension. Defaults to (512,512).
            batch_size (int, optional):  Defaults to 32.
            to_fit (bool, optional): if target is generated. Defaults to True.
            shuffle (bool, optional): if it is shuffled after each epoch. Defaults to True.
        """
        # Ensure that a csv path is given.
        assert csvpath is not None

        self.filenames = filenames
        self.image_path = image_path
        self.csvpath = csvpath
        self.targets = (
            ["Coarse_lesion_type", "Bounding_boxes"] if targets is None else targets
        )
        self.dim = dim
        self.batch_size = batch_size
        self.to_fit = to_fit
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch

        :return: number of batches per epoch
        """
        return int(np.floor(len(self.filenames) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data

        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        filenames_temp = [self.filenames[k] for k in indexes]

        # Generate data
        X = self._generate_X(filenames_temp)

        if self.to_fit:
            y = self._generate_y(filenames_temp)
            return X, y
        else:
            return X

    def _generate_X(self, filenames_temp):
        """Create numpy array from images.
        All images channel are rescaled to range [0, 1]

        Args:
            filenames_temp (list): list of filenames
        """
        data = []
        for fn in filenames_temp:
            imgpath = os.path.sep.join([self.image_path, fn])
            img = imagePreprocessing(imgpath, self.dim)
            data.append(img)
        return np.array(data)

    def _generate_y(self, filenames_temp):
        """Create targets

        Args:
            filenames_temp (list): list of filenames
        """
        targets_list = []
        targets = {"class_label": [], "bounding_box": []}
        df = pd.read_csv(self.csvpath)

        for fn in filenames_temp:
            row = df[df["File_name"] == fn]
            if "Coarse_lesion_type" in self.targets:
                leison_types = row["Coarse_lesion_type"]
                temp_lt_list = np.zeros(8, dtype="int")
                for i in leison_types:
                    temp_lt_list[i - 1] = 1
                targets["class_label"].append(temp_lt_list)

            if "Bounding_boxes" in self.targets:
                bboxes = row["Bounding_boxes"].iloc[:]
                original_dim = row["Image_size"].iloc[:]
                img_mask = None
                for i in range(len(row)):
                    cur_dim = original_dim.iloc[i]  # if len(row) > 1 else original_dim
                    cur_coords = [float(j.strip()) for j in bboxes.iloc[i].split(",")]
                    img_mask = layer_mask(self.dim, cur_dim, cur_coords, mask=img_mask)
                targets["bounding_box"].append(img_mask)

            targets_list.append(targets)
        targets["class_label"] = np.array(targets["class_label"])
        targets["bounding_box"] = np.array(targets["bounding_box"])
        return targets

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def image_mask(self, bounding_box):
        """One in box, zero is out of the box"""

        return np.zeros_like(shape=self.dim)


def layer_mask(resultant_dim, original_dim, coords, mask=None):
    """Generate the layer mask for where the lesion is.

    Args:
        resultant_dim (tuple): Shape of the resultant output. Same for all.
        original_dim (str): Dimension of the image dimension.
        coords (list): the (sx,sy, ex,ey)
        mask (np.ndarray, optional): the image mask to modify, create new one if none. Defaults to None.

    Returns:
        [type]: [description]
    """
    if mask is None:
        mask = np.zeros(shape=resultant_dim, dtype=np.float)
    if original_dim != resultant_dim:
        original_dim = [int(j.strip()) for j in original_dim.split(",")]
        sx, ex = [int(x * resultant_dim[0] // original_dim[0]) for x in coords[::2]]
        sy, ey = [int(y * resultant_dim[1] // original_dim[1]) for y in coords[1::2]]
    else:
        sx, sy, ex, ey = [int(j) for j in coords]
    for row_offset in range(sy, ey):
        mask.ravel()[
            sx + row_offset * resultant_dim[0] : ex + row_offset * resultant_dim[0]
        ] = 1
    return mask


def imagePreprocessing(imgpath, dim):
    """Resize, normalise the image.

    Args:
        imgpath (str): path to find the image
        dim (tuple): Resulting dimension of the image.

    Returns:
        np.ndarray: the processed result of the image.
    """
    img = img_to_array(load_img(imgpath).resize(dim), dtype="float32")
    img = np.delete(img, np.s_[1:], axis=-1)
    img = np.repeat(img, 3, axis=2)
    img /= 255
    return img


# def genBboxCoord(csvpath, filenames):
#     """Get the list of

#     Args:
#         csvpath (str): path to the csv file.
#         filenames (list): list of filenames

#     Returns:
#         list: list of coordinate
#     """
#     df = pd.read_csv(csvpath)
#     coords_list = []
#     for fn in filenames:
#         temp_coord_list = []
#         row = df[df["File_name"] == fn]
#         bboxes = row["Bounding_boxes"].iloc[:]
#         for i in range(len(row)):
#             cur_coords = [float(j.strip()) for j in bboxes.iloc[i].split(",")]
#             temp_coord_list.append(cur_coords)
#         coords_list.append(temp_coord_list)
#     return coords_list
