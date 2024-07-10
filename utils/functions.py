import sys
import logging
import os
import datetime

import cv2
import joblib 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from utils.generate_data import load_train_test




SEED = 5# Random seed for deterministic
n_classes = 7
epochs = 100
batch_size = 8
img_shape = (256, 256)
NUM_TRIALS = 1  # numero de splits generados
TRAIN_SIZE = 0.8
VALID_SIZE = 0.2
DATA_DIR = '/content/drive/MyDrive/TFM/exp_output/local/data' #Directory in which the data are placed
OUT_DIR = '/content/drive/MyDrive/TFM/exp_output/output' #Data output directory


def normalize(input_imgs):
    input_imgs = tf.cast(input_imgs, tf.float16) / 255.0
    return input_imgs

train_imgs,test_imgs,train_labels,test_labels= load_train_test()
train_imgs= normalize(train_imgs)
train_mask_semantic= train_labels[:,:,:,1]
test_mask_semantic= test_labels[:,:,:,1]


def save_plots(history, epochs, model_name):
    folder_save = f"{OUT_DIR}/figures/"
    epochs = range(epochs)

    #Training dice
    plt.plot(history.history['dice_0']) # categorical_dice, dice_coef
    plt.plot(history.history['dice_1']) # val_categorical_dice, val_dice_coef
    plt.plot(history.history['dice_2'])
    plt.plot(history.history['dice_3'])
    plt.plot(history.history['dice_4']) # categorical_dice, dice_coef
    plt.plot(history.history['dice_5']) # val_categorical_dice, val_dice_coef
    plt.plot(history.history['dice_6'])
    plt.title('Training Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice coef')
    plt.legend(['dice_0','dice_1','dice_2','dice_3','dice_4','dice_5','dice_6'], loc = 'lower right')
    plt.savefig(folder_save + f'Dice_{model_name}.png')

    #Validation dice
    plt.figure()
    plt.plot(history.history['val_dice_0']) # categorical_dice, dice_coef
    plt.plot(history.history['val_dice_1']) # val_categorical_dice, val_dice_coef
    plt.plot(history.history['val_dice_2'])
    plt.plot(history.history['val_dice_3'])
    plt.plot(history.history['val_dice_4']) # categorical_dice, dice_coef
    plt.plot(history.history['val_dice_5']) # val_categorical_dice, val_dice_coef
    plt.plot(history.history['val_dice_6'])
    plt.title('Validation Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice coef')
    plt.legend(['dice_0','dice_1','dice_2','dice_3','dice_4','dice_5','dice_6'], loc = 'lower right')
    plt.savefig(folder_save + f'val_Dice_{model_name}.png')

    #Categorical Dice
    plt.figure()
    plt.plot(history.history['categorical_dice']) # val_categorical_dice, val_dice_coef
    plt.plot(history.history['val_categorical_dice'])
    plt.title('Categorical Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Categorical Dice')
    plt.legend([ 'train_cat_dice', 'val_cat_dice'], loc = 'lower right')
    plt.savefig(folder_save + f'categorical_dice_{model_name}.png')

    # History for IoU
    plt.figure()
    plt.plot(history.history['iou']) # categorical_dice, dice_coef
    plt.plot(history.history['val_iou']) # val_categorical_dice, val_dice_coef
    plt.title('Training and Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend(['IoU', 'val_IoU'], loc = 'lower right')
    plt.savefig(folder_save + f'IoU{model_name}.png')

    #History of loss
    plt.figure()
    #plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, history.history["loss"])
    plt.plot(epochs, history.history["val_loss"])
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend(['train','val'], loc='upper left')
    plt.savefig(folder_save + f'loss_{model_name}.png')
    plt.close('all')

def train_segmentator(model,model_name,epochs=100):
    """Train the segmentation network with annotated images."""

    segmentator = model
    start_time = datetime.datetime.now()

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor= "loss",
        factor = 0.1,
        patience = 10,
        verbose = 0,
        mode = "auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=0,
    )

    segmentator_history = segmentator.fit(
        train_imgs, train_mask_semantic,
        batch_size = batch_size,
        epochs = epochs,
        verbose = 1,
        shuffle = True,
        callbacks = [reduce_lr],
        validation_split = VALID_SIZE
    )

    end_time = datetime.datetime.now() - start_time
    print("Time to train model:", end_time)


    folder_save = f'{OUT_DIR}/models/'

    segmentator.save(f"Segmentation_{model_name}.h5")
    save_plots(segmentator_history,epochs,model_name)

def prediction(model,model_name,test_imgs=test_imgs):
    folder_model = f'{OUT_DIR}/models/'
    test_imgs= normalize(test_imgs)
    segmentator = model
    segmentator.load_weights( f"Segmentation_{model_name}.h5")
    segmentator.evaluate(test_imgs, test_mask_semantic)
    pred_test = segmentator.predict(test_imgs, batch_size=8, verbose=1)
    return pred_test

def display_preds(pred,i=1):
    f, axarr = plt.subplots(5,2)
    f.set_figheight(10)
    f.set_figwidth(12)
    axarr[0,0].imshow(test_imgs[i], cmap='gray')
    axarr[0,1].imshow(test_mask_semantic[i], cmap='gray')
    axarr[1,0].imshow(-pred[i,:,:,0], cmap='gray')
    axarr[1,1].imshow(pred[i,:,:,1], cmap='gray')
    axarr[2,0].imshow(pred[i,:,:,2], cmap='gray')
    axarr[2,1].imshow(pred[i,:,:,3], cmap='gray')
    axarr[3,0].imshow(pred[i,:,:,4], cmap='gray')
    axarr[3,1].imshow(pred[i,:,:,5], cmap='gray')
    axarr[4,0].imshow(pred[i,:,:,6], cmap='gray')

    f.delaxes(axarr[4][1])

def save_images(pred,i,model_name):
 
    os.makedirs(f"{OUT_DIR}/images/test_{i}/{model_name}", exist_ok=True)

    plt.imsave(f"{OUT_DIR}/images/test_{i}/{model_name}/fondo{i}_{model_name}.jpg",-pred[i,:,:,0], cmap='gray')
    plt.imsave(f"{OUT_DIR}/images/test_{i}/{model_name}/neutrofilo{i}_{model_name}.jpg",pred[i,:,:,1], cmap='gray')
    plt.imsave(f"{OUT_DIR}/images/test_{i}/{model_name}/epitelial{i}_{model_name}.jpg",pred[i,:,:,2], cmap='gray')
    plt.imsave(f"{OUT_DIR}/images/test_{i}/{model_name}/linfocito{i}_{model_name}.jpg",pred[i,:,:,3], cmap='gray')
    plt.imsave(f"{OUT_DIR}/images/test_{i}/{model_name}/plasma{i}{model_name}.jpg",pred[i,:,:,4], cmap='gray')
    plt.imsave(f"{OUT_DIR}/images/test_{i}/{model_name}/eosinofilo{i}_{model_name}.jpg",pred[i,:,:,5], cmap='gray')
    plt.imsave(f"{OUT_DIR}/images/test_{i}/{model_name}/conectivo{i}_{model_name}.jpg",pred[i,:,:,6], cmap='gray')

   
