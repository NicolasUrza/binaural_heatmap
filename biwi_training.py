import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import skimage
from skimage import transform
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import keras
from keras.utils import to_categorical,Sequence
from keras import applications
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D,Input,MaxPooling2D,Conv2D,BatchNormalization,Activation
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.applications import resnet50, inception_resnet_v2, vgg16
from keras.applications.inception_resnet_v2 import preprocess_input
from keras import optimizers
from keras.optimizers import Adam,SGD
import os
from imutils import face_utils
import dlib
import cv2
import subprocess
from os import walk
def to_rgb(img):
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
import math
from keras.utils import Sequence


class data_gen_train(Sequence):

  def __init__(self, train_size, batch_size, heatmap_path, pyr_frame):
    self.batch_length = batch_size
    self.size = train_size
    self.path = heatmap_path
    self.pyr = pyr_frame

  # def __getitem__(self, idx):
  #   start = idx * self.batch_length
  #   end = min((idx + 1) * self.batch_length, self.size)
  #   dataset = []
  #   pose = []
  #   for i in range(start, end):
  #     heatmaps = plt.imread(self.path[i])
  #     nose = heatmaps[:, 0 * 96:1 * 96]
  #     Reye = heatmaps[:, 14 * 96:15 * 96]
  #     Leye = heatmaps[:, 15 * 96:16 * 96]
  #     Rear = heatmaps[:, 16 * 96:17 * 96]
  #     Lear = heatmaps[:, 17 * 96:18 * 96]
  #     stacked_heatmaps = np.dstack([nose, Reye, Leye, Rear, Lear])
  #     dataset.append(stacked_heatmaps)
  #     pose.append(np.array(self.pyr.iloc[i, 1:]))
  #   return ((np.array(dataset) / 255) - 0.5) * 2, np.array(pose)
  def __getitem__(self, idx):
    start = idx * self.batch_length
    end = min((idx + 1) * self.batch_length, self.size)
    dataset = []
    pose = []
    for i in range(start, end):
      heatmaps = plt.imread(self.path[i])
      nose = heatmaps[:, 0 * 96:1 * 96]
      Reye = heatmaps[:, 14 * 96:15 * 96]
      Leye = heatmaps[:, 15 * 96:16 * 96]
      Rear = heatmaps[:, 16 * 96:17 * 96]
      Lear = heatmaps[:, 17 * 96:18 * 96]
      stacked_heatmaps = np.dstack([nose, Reye, Leye, Rear, Lear])
      dataset.append(stacked_heatmaps)
      pose.append(np.array(self.pyr.iloc[i, 1:]))
    X_batch = ((np.array(dataset) / 255) - 0.5) * 2
    y_batch = np.array(pose, dtype="float32")
    return X_batch, y_batch


  def __len__(self):
    return math.ceil(self.size / self.batch_length)
  def __str__(self):
    return  "pyr: " + str(self.pyr) + "path: " + str(self.path) + "size: " + str(self.size) + "batch_length: " + str(self.batch_length) + "len: " + str(self.__len__())
def Loading_filepath(PATH):
  jpgs = []
  for (dirpath, dirnames, filenames) in walk(PATH):
    for filename in filenames[:]:
      if (filename[-3:] == 'png'):
        jpgs.append(dirpath + "/" + filename)
  return jpgs

def GETPYR(heatmap_paths):
    pyr = pd.DataFrame(columns=['image', 'pitch', 'yaw', 'roll'])
    for i in range(len(heatmap_paths)):
      with open('annotations' + heatmap_paths[i][22:37] + '_pose.bin', 'rb') as fid:  # opening the ground truth file
        data_array = np.fromfile(fid, np.float32)
      para = data_array[3:]
      pyr = pyr._append(
        {'image': 'annotations' + heatmap_paths[i][22:37] + '_pose.bin', 'pitch': para[0], 'yaw': para[1],
         'roll': para[2]}, ignore_index=True)
      if i % 1000 == 0:
        print(str(i) + " images done")
    print(str(i) + " images done")
    return pyr

#
# for i in range(1,25):
#   if i < 10:
#     carpeta = '0' + str(i)
#   else:
#     carpeta = str(i)
#   print("procesando carpeta: " + carpeta)
#   print("creando carpeta: "+ "output_heatmaps_folder/"+carpeta )
#   os.mkdir('output_heatmaps_folder/'+carpeta)
#   out = subprocess.Popen(
#     ['/home/nicolas/Escritorio/openpose/build/examples/openpose/openpose.bin', '--image_dir', 'face_rectangle_crops/'+carpeta,
#      '--model_pose', 'COCO', '--heatmaps_add_parts', '-heatmaps_add_bkg', '--display', '0', '--render_pose', '0',
#      '--net_resolution', "96x96", '--write_heatmaps', 'output_heatmaps_folder/'+carpeta], stdout=subprocess.PIPE).communicate()
#   print("carpeta procesada")

heatmap_paths = Loading_filepath('output_heatmaps_folder')
pyr = GETPYR(heatmap_paths)
print("heatmap_paths y pyr cargados")


print("creando modelo")

traingen = data_gen_train(12000,128,heatmap_paths[:12000],pyr[:12000])
print("traingen")
traingen.__getitem__(0)
valgen = data_gen_train(2049,128,heatmap_paths[12000:],pyr[12000:])
inpu = Input(shape = (96,96,5))
x = Conv2D(50, (5, 5), activation=None, padding='valid')(inpu)
#x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Dropout(0.2)(x)
x = Conv2D(50, (5, 5), activation=None, padding='valid')(x)
#x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Dropout(0.2)(x)
x = Conv2D(150, (5, 5), activation=None, padding='valid')(x)
#x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Dropout(0.3)(x)
x = Flatten()(x)
x = Dense(300, activation='tanh')(x)
x = Dropout(0.3)(x)
x = Dense(300, activation='tanh')(x)
x = Dropout(0.3)(x)
pred = Dense(3,activation='tanh')(x)
model = Model(inputs = inpu,outputs = pred)
model.summary()

print("compilando modelo")
model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(0.00001), metrics=['mae'])
model_saver = keras.callbacks.ModelCheckpoint('weights.h5', monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=1)
print("entrenando modelo")
model.fit(traingen,validation_data = valgen,epochs = 10,callbacks = [model_saver])
