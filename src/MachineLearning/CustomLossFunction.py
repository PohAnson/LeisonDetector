from tensorflow.keras import backend as K


def dice_coef(y_true, y_prediction, smooth=1):
    intersection = K.sum(y_true * y_prediction, axis=[1, 2])
    union = K.sum(y_true, axis=[1, 2]) + K.sum(y_prediction, axis=[1, 2])
    dice = K.mean((2.0 * intersection + smooth) / (union + smooth), axis=0)
    return 1 - dice
