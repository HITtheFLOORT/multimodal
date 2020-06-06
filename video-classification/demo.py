"""
Given a video path and a saved model (checkpoint), produce classification
predictions.

Note that if using a model that requires features to be extracted, those
features must be extracted first.

Note also that this is a rushed demo script to help a few people who have
requested it and so is quite "rough". :)
"""
import os
from subprocess import call

import cv2
import tensorflow.compat.v1 as tf
import keras.backend.tensorflow_backend
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

from keras.models import load_model
from data import DataSet
import numpy as np

def predict(data_type, seq_length, saved_model, image_shape, video_name, class_limit):
    model = load_model(saved_model)

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(seq_length=seq_length, class_limit=class_limit)
    else:
        data = DataSet(seq_length=seq_length, image_shape=image_shape,
            class_limit=class_limit)
    
    # Extract the sample from the data.
    sample = data.get_frames_by_filename(video_name, data_type)

    # Predict!
    prediction = model.predict(np.expand_dims(sample, axis=0))
    return prediction
    #data.print_class_from_prediction(np.squeeze(prediction, axis=0))

def preds(saved_model,class_limit,video_name):
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d.
    model = 'conv_3d'
    # Must be a weights file.
    #saved_model = 'data/checkpoints/conv_3d-images.006-0.875.hdf5'
    # Sequence length must match the lengh used during training.
    seq_length = 40
    # Limit must match that used during training.
    #class_limit = 3

    # Demo file. Must already be extracted & features generated (if model requires)
    # Do not include the extension.
    # Assumes it's in data/[train|test]/
    # It also must be part of the train/test data.
    # TODO Make this way more useful. It should take in the path to
    # an actual video file, extract frames, generate sequences, etc.
    #video_name = 'v_Archery_g04_c02'
    #video_name = '6e9e26'

    # Chose images or features and image shape based on network.
    if model in ['conv_3d', 'c3d', 'lrcn']:
        data_type = 'images'
        image_shape = (80, 80, 3)
    elif model in ['lstm', 'mlp']:
        data_type = 'features'
        image_shape = None
    else:
        raise ValueError("Invalid model. See train.py for options.")

    r=predict(data_type, seq_length, saved_model, image_shape, video_name, class_limit)
    print(r)
    return r

def main(path):
    pred = preds('data/acgORact.hdf5', 2, path.replace('.mp4', ''))
    lable1 = ['acg', 'act']
    i = pred.argmax(axis=1)[0]
    if i==2 and pred[0][i]< 0.6:
        i=1
    print(lable1[i] + ':' + str(pred[0][i]))
    if i == 0:
        pred = preds('data/catoonORgame.hdf5', 2, path.replace('.mp4', ''))
        lable1 = ['catoon', 'game']
        a = pred.argmax(axis=1)[0]
        print(lable1[a] + ':' + str(pred[0][a]))
    else:
        if i == 1:
            pred = preds('data/animalORitemORpersonORview.hdf5', 4, path.replace('.mp4', ''))
            lable1 = ['animal', 'item', 'person', 'veiw']
            b = pred.argmax(axis=1)[0]
            print(lable1[b] + ':' + str(pred[0][b]))


def testvideopretreat(path):
    cap = cv2.VideoCapture(path)
    frames_num = cap.get(7)
    num = int(frames_num / 40)
    dest = path.replace('.mp4', '') + '-%04d.jpg'
    call(["ffmpeg", "-i", path, '-vf', "select=not(mod(n\,{a}))".format(a=num - 1), '-vsync', '0', dest])

if __name__ == '__main__':
    path = '4.mp4'
    main(path)

    #testvideopretreat('testdata/'+path)
