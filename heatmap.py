import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import cv2
# import skimage
# from skimage import transform
# from sklearn.preprocessing import LabelEncoder
# from sklearn.utils import class_weight
# import keras
# from keras.utils import to_categorical,Sequence
# from keras import applications
# from keras.preprocessing import image
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential, Model
# from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D,Input,MaxPooling2D,Conv2D,BatchNormalization,Activation
# from keras import backend as k
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
# from keras.applications import resnet50, inception_resnet_v2, vgg16
# from keras.applications.inception_resnet_v2 import preprocess_input
# from keras import optimizers
# from keras.optimizers import Adam,SGD
# import os
# from imutils import face_utils
import dlib
# import math
# import tarfile
import subprocess
import sys
# import argparse
try:
    sys.path.append('/usr/local/python')
    from openpose import pyopenpose as op
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. '
        'Did you enable `BUILD_PYTHON` in CMake and have this Python script '
        'in the right folder? the error was: {}'.format(e)
    )
    raise e


def GETPYR(heatmap_paths):
    '''
    File directory generator.
    ---INPUT---
    Hheatmap_path --- List of heatmap paths
    ---OUTPUT---
    pitch yaw roll iin a dataframe
    ** Made for BIWI dataset **
    '''
    pyr = pd.DataFrame(columns=['image', 'pitch', 'yaw', 'roll'])
    for i in range(len(heatmap_paths)):
        with open('annotations' + heatmap_paths[i][31:46] + '_pose.bin', 'rb') as fid:  # opening the ground truth file
            data_array = np.fromfile(fid, np.float32)
        para = data_array[3:]
        pyr = pyr.append(
            {'image': 'annotations' + heatmap_paths[i][31:46] + '_pose.bin', 'pitch': para[0], 'yaw': para[1],
             'roll': para[2]}, ignore_index=True)
        if i % 1000 == 0:
            print(str(i) + " images done")
    print(str(i) + " images done")
    return pyr



detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")


def GETCROPS(image):
    '''
    recorta la cara en la imagen
    '''
    dets = detector(image, 0)  # 0 is for not upsampling the image
    print(dets)
    if len(dets) == 1:  # Making sure that only one face is detected (as the images contain people in the background)
        crop_img = image[dets[0].rect.top() - 15:dets[0].rect.bottom() + 15, dets[0].rect.left() - 15:dets[0].rect.right() + 15]  # cropping the image with 15 extra pixels on all side
        return crop_img
    return None


def preprocesar_video(image):

    img = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)  # Corrección de la normalización
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Corrección de la conversión de color
    img = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    return img


def ProcesarVideo(videoPath):
    print(videoPath)
    cap = cv2.VideoCapture(videoPath)
    pyr = pd.DataFrame(columns=['image', 'pitch', 'yaw', 'roll'])
    cantFrames = 0
    cantDetecciones = 0
    # Starting OpenPose
    params = dict()
    params["model_folder"] = "/home/nicolas/Escritorio/openpose/models/"
    params["heatmaps_add_parts"] = True
    params["heatmaps_add_bkg"] = True
    params["heatmaps_add_PAFs"] = True
    params["heatmaps_scale"] = 2
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    datum = op.Datum()
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        cantFrames += 1
        image = preprocesar_video(image)
        results = GETCROPS(image)

        if results is not None:
            image = results
            # escribir que se encontro un resultado
            # Process Image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            datum.cvInputData = image
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            # Process outputs
            outputImageF = (datum.inputNetData[0].copy())[0, :, :, :] + 0.5
            outputImageF = cv2.merge([outputImageF[0, :, :], outputImageF[1, :, :], outputImageF[2, :, :]])
            outputImageF = (outputImageF * 255.).astype(dtype='uint8')
            heatmaps = datum.poseHeatMaps.copy()
            heatmaps = (heatmaps).astype(dtype='uint8')
            print("heatmaps: " + str(heatmaps))
            # pasar el heatmap a np array
            counter = 0
            cant_heatmaps = heatmaps.shape[0]
            print("la cantidad de heatmaps es: " + str(cant_heatmaps))
            for i in range (0, cant_heatmaps):
                heatmap = heatmaps[counter, :, :].copy()
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                combined = cv2.addWeighted(outputImageF, 0.5, heatmap, 0.5, 0)
                cv2.imshow("Heatmaps Detected", combined)
                counter += 1

            #cv2.putText(image, "Face Detected", (5,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (250, 0, 0), 1)
            cantDetecciones += 1
        else:
            # escribir que no se encontro un resultado
            cv2.putText(image, "No Detection", (5,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 250), 1)
            cv2.imshow('Video', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()

    porc_detecciones = (cantDetecciones/cantFrames)*100
    print("porcentaje de detecciones: " + str(porc_detecciones) + "%")


def ProcesarVideo2(videoPath):
    print(videoPath)
    cap = cv2.VideoCapture(videoPath)
    pyr = pd.DataFrame(columns=['image', 'pitch', 'yaw', 'roll'])
    cantFrames = 0
    cantDetecciones = 0
    # Starting OpenPose

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        cantFrames += 1
        image = preprocesar_video(image)
        results = GETCROPS(image)

        if results is not None:
            image = results
            # escribir que se encontro un resultado
            # Process Image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite("temporal/image.jpg", image)
            print("procesando frame "+ str(cantFrames))
            out = subprocess.Popen(['/home/nicolas/Escritorio/openpose/build/examples/openpose/openpose.bin', '--image_dir', 'temporal', '--model_pose', 'COCO', '--heatmaps_add_parts', '-heatmaps_add_bkg', '--display', '0', '--render_pose', '0', '--net_resolution', "96x96", '--write_heatmaps', 'temporal'], stdout=subprocess.PIPE).communicate()[0]
            print("procesado")
            input("Press Enter to continue...")

            #cv2.putText(image, "Face Detected", (5,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (250, 0, 0), 1)
            cantDetecciones += 1
        else:
            # escribir que no se encontro un resultado
            cv2.putText(image, "No Detection", (5,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 250), 1)
            cv2.imshow('Video', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()

    porc_detecciones = (cantDetecciones/cantFrames)*100
    print("porcentaje de detecciones: " + str(porc_detecciones) + "%")

if __name__ == '__main__':
    ProcesarVideo2('/home/nicolas/Escritorio/CINTRA/Videos/12 meses/P2 12 meses/20090326-101237/video03.avi')
