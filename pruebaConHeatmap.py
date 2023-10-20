
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.layers import Dense, Activation, LeakyReLU, BatchNormalization, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, Input
from keras.models import Model
def create_model():
    inpu = Input(shape=(96, 96, 5))
    x = Conv2D(50, (5, 5), activation=None, padding='valid')(inpu)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(50, (5, 5), activation=None, padding='valid')(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(150, (5, 5), activation=None, padding='valid')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(300, activation='tanh')(x)
    x = Dropout(0.3)(x)
    x = Dense(300, activation='tanh')(x)
    x = Dropout(0.3)(x)
    pred = Dense(3, activation='tanh')(x)
    model = Model(inputs=inpu, outputs=pred)
    return model
model = create_model()
model.load_weights('weights.h5')
image = plt.imread('/home/nicolas/Escritorio/CINTRA/Headpose_Estimation/output_heatmaps_folder/22/frame_00635_rgb_pose_heatmaps.png')
dataset = []
# Process the image
nose = image[:, 0 * 96:1 * 96]
Reye = image[:, 14 * 96:15 * 96]
Leye = image[:, 15 * 96:16 * 96]
Rear = image[:, 16 * 96:17 * 96]
Lear = image[:, 17 * 96:18 * 96]
stacked_heatmaps = np.dstack([nose, Reye, Leye, Rear, Lear])
dataset.append(stacked_heatmaps)
# Normalize the image
normalized_image = (((np.array(dataset) ) / 255) - 0.5) * 2
print("prediction: ")
predictions = model.predict(normalized_image)
print(str(predictions))

print("anotation: ")
annotations = open('/home/nicolas/Escritorio/CINTRA/Headpose_Estimation/annotations/22/frame_00635_pose.bin', 'rb')
data_array = np.fromfile(annotations, np.float32)
para = data_array[3:]
print("anotaciones: ")
print(str(para))
