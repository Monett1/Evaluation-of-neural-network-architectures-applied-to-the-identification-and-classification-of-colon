from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.activations import softmax
from keras.models import Model

from utils.metrics import dice_0, dice_1, dice_2, dice_3, dice_4, dice_5, dice_6, iou, categorical_dice, categorical_dice_loss

def get_unet():
    inputs = Input((256, 256, 3))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs) #32
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)  #32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #64
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2) #64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #128
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #128
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3) #256
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4) #256
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4) #512
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5) #512

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3) #256
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6) #256
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6) #256

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3) #128
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7) #128
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7) #128

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3) #64
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8) #64
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8) #64

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3) #32
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9) #32
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9) #32

    conv10 = Conv2D(7, 1, activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    loss = categorical_dice_loss



    model.compile(optimizer=Adam(lr=1e-3), loss=loss, metrics=[dice_0, dice_1, dice_2, dice_3, dice_4, dice_5, dice_6, iou, categorical_dice])

    return model


