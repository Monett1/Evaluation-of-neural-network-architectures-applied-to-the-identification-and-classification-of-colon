import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np



def dice_coef(y_true, y_pred, smooth=1):
    y_true = K.flatten(y_true) #, num_classes = K.int_shape(y_pred)[-1]))
    y_pred = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true * y_pred))
    return (2. * intersection + smooth) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + smooth)

def dice_0(y_true, y_pred, index=0):
    y_true= K.one_hot(K.cast(y_true, 'int32'),num_classes = K.int_shape(y_pred)[-1]) 
    return dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])

def dice_1(y_true, y_pred, index=1):
    y_true= K.one_hot(K.cast(y_true, 'int32'),num_classes = K.int_shape(y_pred)[-1]) 
    return dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])

def dice_2(y_true, y_pred, index=2):
    y_true= K.one_hot(K.cast(y_true, 'int32'),num_classes = K.int_shape(y_pred)[-1]) 
    return dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])

def dice_3(y_true, y_pred, index=3):
    y_true= K.one_hot(K.cast(y_true, 'int32'),num_classes = K.int_shape(y_pred)[-1]) 
    return dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])

def dice_4(y_true, y_pred, index=4):
    y_true= K.one_hot(K.cast(y_true, 'int32'),num_classes = K.int_shape(y_pred)[-1]) 
    return dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])

def dice_5(y_true, y_pred, index=5):
    y_true= K.one_hot(K.cast(y_true, 'int32'),num_classes = K.int_shape(y_pred)[-1]) 
    return dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])

def dice_6(y_true, y_pred, index=6):
    y_true= K.one_hot(K.cast(y_true, 'int32'),num_classes = K.int_shape(y_pred)[-1]) 
    return dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def categorical_dice(y_true, y_pred):
    """Categorical dice loss function.
        Dice = (2*|X & Y|)/ (|X|+ |Y|)= 2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    Args:
      y_true: reference multi-class segmentation volume (batch, h, w, 1)
      y_pred: predicted multi-class segmentation volume (batch, h, w, nClasses) or (batch, h, w)
    Returns:
      categorical_dice: categorical dice coefficient loss.
    """
    y_true = tf.squeeze(tf.keras.backend.one_hot(tf.keras.backend.cast(y_true, 'int32'), num_classes = K.int_shape(y_pred)[-1])) # One-hot encoding of y_true
    intersection = 2 * tf.reduce_sum(y_true * y_pred, (0, 1, 2)) # Intersection per class
    union = tf.reduce_sum((tf.square(y_true) + tf.square(y_pred)), (0, 1, 2)) # Union per class
    dice = intersection / (union + tf.keras.backend.epsilon()) # Per class dice coef
    cat_dice = tf.keras.backend.mean(dice) # Mean dice coef
    return cat_dice

def iou(y_true, y_pred):
    y_true = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes = K.int_shape(y_pred)[-1]))
    y_pred = K.flatten(y_pred)
    intersection = tf.reduce_sum(tf.cast(y_true, tf.float32) * tf.cast(y_pred, tf.float32)) #K.sum(K.abs(y_true * y_pred), axis=-1)
    score = (intersection + 1.) / (tf.reduce_sum(tf.cast(y_true, tf.float32)) + tf.reduce_sum(tf.cast(y_pred, tf.float32)) - intersection + 1.)
    return score

def categorical_dice_loss(y_true, y_pred):
    return 1 - categorical_dice(y_true, y_pred)