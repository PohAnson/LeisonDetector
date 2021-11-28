# import tensorflow.keras as K
from tensorflow.keras import backend as K


def DiceLoss(targets, inputs, smooth=1e-6):
    # flatten label and prediction tensors
    print(inputs, targets)
    input("wait")
    # inputs = K.flatten(inputs)
    # targets = K.flatten(targets)
    input("before dsum")
    dsum = K.dot(targets, inputs)
    print("after dsum")
    input(dsum)

    intersection = K.sum(dsum)
    dice = (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2])
    union = K.sum(y_true, axis=[1, 2]) + K.sum(y_pred, axis=[1, 2])
    dice = K.mean((2.0 * intersection + smooth) / (union + smooth), axis=0)
    return 1 - dice


def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou
