# import tensorflow.keras as K
from tensorflow.keras import backend as K

def DiceLoss(targets, inputs, smooth=1e-6):
    #flatten label and prediction tensors
    # inputs = K.flatten(inputs)
    # targets = K.flatten(targets)
    intersection = K.sum(K.dot(targets, inputs))
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice