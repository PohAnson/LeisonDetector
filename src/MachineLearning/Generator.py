import os
import random

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import Sequence

random.seed(123)
np.random.seed(123)


class DataGenerator(Sequence):
    """Generates data for Keras.
    Sequence based data generator. \
    Suitable for building data generator for training and prediction.
    """

    def __init__(
        self,
        filenames: list,
        image_path: str,
        csvpath: str = None,
        targets: list = None,
        dim: tuple[int, int] = (512, 512),
        batch_size: int = 32,
        to_fit: bool = True,
        shuffle: bool = True,
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
            shuffle (bool, optional): if it is shuffled after each epoch. \
                Defaults to True.
        """
        # Ensure that a csv path is given.
        assert csvpath is not None

        self.filenames = filenames
        self.image_path = image_path
        self.csvpath = csvpath
        self.targets = (
            ["Coarse_lesion_type", "Bounding_boxes"]
            if targets is None else targets
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
        return int(len(self.filenames) // self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data

        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        batch_first = index * self.batch_size
        indexes = self.indexes[batch_first:batch_first + self.batch_size]

        # Find list of IDs
        filenames_temp = [self.filenames[k] for k in indexes]

        # Generate data
        X = self._generate_X(filenames_temp)

        if self.to_fit:
            y = self._generate_y(filenames_temp)
            return X, y
        else:
            return X

    def _generate_X(self, filenames_temp: list[str]):
        """Create numpy array from images.
        All images channel are rescaled to range [0, 1]

        Args:
            filenames_temp (list): list of filenames
        """
        data = []
        for fn in filenames_temp:
            imgpath = os.path.sep.join([self.image_path, fn])
            img = image_preprocessing(imgpath, self.dim)
            data.append(img)
        return np.array(data)

    @staticmethod
    def get_lesion_label(row: pd.Series) -> np.ndarray:
        """Returns one hot encoding of the leison type.

        Args:
            row (pd.Series): Have header `Coarse_lesion_type`

        Returns:
            np.ndarray: one hot encoding of the target.
        """
        leison_types = row["Coarse_lesion_type"]
        lt_list = np.zeros(8, dtype=np.uint8)
        for i in leison_types:
            lt_list[i - 1] = 1
        return lt_list

    def get_bounding_box(self, row: pd.Series) -> np.ndarray:
        """Generate the bounding box.

        Args:
            row (pd.Series): Have header `Bounding_boxes` and `Image_size`

        Returns:
            np.ndarray: the resultant image mask
        """
        bboxes = row["Bounding_boxes"].iloc[:]
        original_dim = row["Image_size"].iloc[:]
        img_mask = np.array([])
        for i in range(len(row)):
            cur_dim = original_dim.iloc[i]
            cur_dim = tuple(cur_dim.split(","))
            cur_dim = (int(cur_dim[0]), int(cur_dim[1]))
            cur_coords = [float(j.strip()) for j in bboxes.iloc[i].split(",")]
            img_mask = create_layer_mask(
                self.dim, cur_dim, cur_coords, mask=img_mask)

        return img_mask

    def _generate_y(self, filenames_temp) -> dict[str, np.ndarray]:
        """Create targets

        Args:
            filenames_temp (list): list of filenames
        """
        targets = {"class_label": np.array([]), "bounding_box": np.array([])}
        df = pd.read_csv(self.csvpath)

        # generating the target for each item
        for fn in filenames_temp:
            row = df[df["File_name"] == fn]
            if "Coarse_lesion_type" in self.targets:
                targets["class_label"] = np.append(
                    targets["class_label"], self.get_lesion_label(row)
                )
            if "Bounding_boxes" in self.targets:
                targets["bounding_box"] = np.append(
                    targets["bounding_box"], self.get_bounding_box(row)
                )
        return targets

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.filenames))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def create_layer_mask(
    resultant_dim: tuple[int, int],
    original_dim: tuple[int, int],
    coords: list,
    mask: np.ndarray = np.array([]),
) -> np.ndarray:
    """Generate the layer mask for where the lesion is.

    Args:
        resultant_dim (tuple): Shape of the resultant output. Same for all.
        original_dim (str): Dimension of the image dimension.
        coords (list): the (sx,sy, ex,ey)
        mask (np.ndarray, optional): the image mask to modify, create new one \
        if none. Defaults to None.

    Returns:
        np.ndarray: the mask of the array
    """
    if mask.size == 0:  # mask is empty
        mask = np.zeros(shape=resultant_dim, dtype=np.float)
    original_dim = (int(original_dim[0]), int(original_dim[1]))

    # get the coordinate to create mask.
    # sx: start x, sy: start y; start is top left
    # ex: end x, ey: end y; end is bottom right
    if original_dim != resultant_dim:
        # apply scaling to make coordinate in the resultant dimension
        sx, ex = [int(x * resultant_dim[0] // original_dim[0])
                  for x in coords[::2]]
        sy, ey = [int(y * resultant_dim[1] // original_dim[1])
                  for y in coords[1::2]]
    else:
        sx, sy, ex, ey = map(int, coords)

    # create the mask based on the coordinates.
    x_dim = resultant_dim[0]
    for row_offset in range(sy, ey):
        mask.ravel()[
            sx + row_offset * x_dim: ex + row_offset * x_dim] = 1
    return mask


def image_preprocessing(imgpath: str, dim: tuple):
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
