import cv2
import joblib #Joblib es un conjunto de herramientas para proporcionar un pipelining en Python.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit 

SEED = 5# Random seed for deterministic
n_classes = 7
epochs = 50
batch_size = 8
img_shape = (256, 256)
NUM_TRIALS = 1  # número de splits generados
TRAIN_SIZE = 0.8
VALID_SIZE = 0.2
DATA_DIR = '/content/drive/MyDrive/TFM/exp_output/local/data' #Directory in which the data are placed
OUT_DIR = '/content/drive/MyDrive/TFM/exp_output/output' #Data output directory

def display():
    img_array = np.load("f{DATA_DIR}/images.npy")
    label_array=np.load("f{DATA_DIR}/labels.npy)

    instance_map= label_array[:,:,:,0]
    classification_map= label_array[:,:,:,1]

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(img_array[1], cmap='gray')
    axarr[0,1].imshow(instance_map[1], cmap='gray')
    axarr[1,0].imshow(img_array[1], cmap='gray')
    axarr[1,1].imshow(classification_map[1], cmap='gray')


def generate_splits():

    info = pd.read_csv(f'{DATA_DIR}/patch_info.csv')
    file_names = np.squeeze(info.to_numpy()).tolist() #Convierte el dataframe en un numpy array, elimina los ejes de longitud 1 y lo pasa a una lista.

    img_sources = [v.split('-')[0] for v in file_names]
    img_sources = np.unique(img_sources) #Eliminamos el guión que hay en los nombres de los archivos y nos quedamos con el nombre del archivo que hay antes del guión (eliminamos la numeración)

    cohort_sources = [v.split('_')[0] for v in img_sources]
    _, cohort_sources = np.unique(cohort_sources, return_inverse=True) #Obtenemos una lista que contiene elementos del 0 al 4 (0:Consep, 1:crag, 2:dpath, 3:glas, 4:pannuke)

    splitter = StratifiedShuffleSplit( #Creamos un splitter con las condiciones especificadas anteriormente
        n_splits=NUM_TRIALS,
        train_size=TRAIN_SIZE,
        test_size=VALID_SIZE,
        random_state=SEED
    )

    splits = []
    split_generator = splitter.split(img_sources, cohort_sources) #Aplicamos el splitter sobre las dos listas obtenidas anteriormente
    for train_indices, test_indices in split_generator:
        train_cohorts = img_sources[train_indices]
        test_cohorts = img_sources[test_indices]
        assert np.intersect1d(train_cohorts, test_cohorts).size == 0
        train_names = [
            file_name
            for file_name in file_names
            for source in train_cohorts
            if source == file_name.split('-')[0]
        ]
        test_names = [
            file_name
            for file_name in file_names
            for source in test_cohorts
            if source == file_name.split('-')[0]
        ]
        train_names = np.unique(train_names) #Obtenemos un array de los train names
        test_names = np.unique(test_names) #Obtenemos un array de los valid names
        print(f'Train: {len(train_names):04d} - Test: {len(test_names):04d}')
        assert np.intersect1d(train_names, test_names).size == 0 #Comprobamos que no hay ningún elemento en ambos arrays
        train_indices = [file_names.index(v) for v in train_names] #Guardamos los índices de los train names
        test_indices = [file_names.index(v) for v in test_names] #Guardamos los índices de los test_names
        splits.append({ #Añadimos a la lista split los índices de train y valid para las 10 particiones de datos
            'train': train_indices,
            'test': test_indices
        })
    joblib.dump(splits, f"{OUT_DIR}/splits.dat")

def save_train_test():

    imgs = np.load(f'{DATA_DIR}/images.npy') #Cargamos las imágenes
    labels = np.load(f'{DATA_DIR}/labels.npy') #Cargamos las etiquetas

    splits = joblib.load(f'{OUT_DIR}/splits.dat') #Cargamos los splits
    train_indices= splits[0]['train'] #Escogemos los índices del split para el train
    test_indices = splits[0]['test'] #Escogemos los índices del split para el test

    train_imgs= imgs[train_indices]
    test_imgs= imgs[test_indices]

    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    np.save(f'{OUT_DIR}/train_imgs.npy', train_imgs)
    np.save(f'{OUT_DIR}/test_imgs.npy', test_imgs)
    np.save(f'{OUT_DIR}/train_true.npy', train_labels)
    np.save(f'{OUT_DIR}/test_true.npy', test_labels)

#save_train_test()

def load_train_test():

    train_imgs= np.load(f'{OUT_DIR}/train_imgs.npy')
    test_imgs= np.load(f'{OUT_DIR}/test_imgs.npy')
    train_labels= np.load(f'{OUT_DIR}/train_true.npy')
    test_labels= np.load(f'{OUT_DIR}/test_true.npy')

    return train_imgs,test_imgs,train_labels,test_labels
