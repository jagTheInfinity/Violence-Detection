import numpy as np
import cv2
from skimage.transform import resize
import time
from tensorflow.keras import optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
import tensorflow as tf

def mamon_videoFightModel2(tf, wight=r'./mamonbest947oscombo-drive.hdfs'):
    layers = tf.keras.layers
    models = tf.keras.models
    losses = tf.keras.losses
    metrics = tf.keras.metrics
    num_classes = 2
    cnn = models.Sequential()
    # cnn.add(base_model)

    input_shapes = (160, 160, 3)
    np.random.seed(1234)
    vg19 = tf.keras.applications.vgg19.VGG19
    base_model = vg19(include_top=False, weights='imagenet', input_shape=(160, 160, 3))
    # Freeze the layers except the last 4 layers
    # for layer in base_model.layers:
    #    layer.trainable = False

    cnn = models.Sequential()
    cnn.add(base_model)
    cnn.add(layers.Flatten())
    model = models.Sequential()

    model.add(layers.TimeDistributed(cnn, input_shape=(30, 160, 160, 3)))
    model.add(layers.LSTM(30, return_sequences=True))

    model.add(layers.TimeDistributed(layers.Dense(90)))
    model.add(layers.Dropout(0.1))

    model.add(layers.GlobalAveragePooling1D())

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(num_classes, activation="sigmoid"))

    adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.load_weights(wight)
    rms = optimizers.RMSprop()

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])

    return model
    # Your model architecture function code here

def video_mamonreader(cv2, filename):
    frames = np.zeros((30, 160, 160, 3), dtype=np.float)
    i = 0
    print(frames.shape)
    vc = cv2.VideoCapture(filename)
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
        frame = None
    if frame is not None:
        frm = resize(frame, (160, 160, 3))
        frm = np.expand_dims(frm, axis=0)
        if np.max(frm) > 1:
            frm = frm / 255.0
        frames[i][:] = frm
        i += 1
        print("reading video")
        while i < 30:
            rval, frame = vc.read()
            frm = resize(frame, (160, 160, 3))
            frm = np.expand_dims(frm, axis=0)
            if np.max(frm) > 1:
                frm = frm / 255.0
            frames[i][:] = frm
            i += 1
    return frames
    # Your video reader function code here

def pred_fight(model, video, acuracy=0.8):
    pred_test = model.predict(video)
    if pred_test[0][1] >= acuracy:
        return True, pred_test[0][1]
    else:
        return False, pred_test[0][1]

    # Your prediction function code here

def main_fight(vidoss):
    vid = video_mamonreader(cv2, vidoss)
    datav = np.zeros((1, 30, 160, 160, 3), dtype=np.float)
    datav[0][:][:] = vid
    millis = int(round(time.time() * 1000))
    print(millis)
    f, precent = pred_fight(model22, datav, acuracy=0.65)
    millis2 = int(round(time.time() * 1000))
    print(millis2)
    res_mamon = {'fight': f, 'precentegeoffight': str(precent)}
    res_mamon['processing_time'] = str(millis2 - millis)
    return res_mamon
    # Your main function code here

