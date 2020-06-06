"""
Given a video path and a saved model (checkpoint), produce classification
predictions.

Note that if using a model that requires features to be extracted, those
features must be extracted first.

Note also that this is a rushed demo script to help a few people who have
requested it and so is quite "rough". :)
"""
import glob
import os
import re
from subprocess import call

import cv2
import jieba
import librosa
import tensorflow.compat.v1 as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras.backend.tensorflow_backend
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
from ocr import text_predict
from keras.models import load_model
from data import DataSet
import numpy as np
import numpy as np
from subprocess import call
import jieba
import os
from collections import Counter
from PIL import Image
from textrank4zh import TextRank4Keyword
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

def fivelabel(path):
    pred = preds('acgORact.hdf5', 2, path.replace('.mp4', ''))
    lable1 = ['acg', 'act']
    i = pred.argmax(axis=1)[0]
    lable_f=''
    if i==2 and pred[0][i]< 0.6:
        i=1
    print(lable1[i] + ':' + str(pred[0][i]))
    lable_f+=lable1[i]+' '
    if i == 0:
        pred = preds('catoonORgame.hdf5', 2, path.replace('.mp4', ''))
        lable1 = ['catoon', 'game']
        a = pred.argmax(axis=1)[0]
        print(lable1[a] + ':' + str(pred[0][a]))
        lable_f += lable1[a]
    else:
        if i == 1:
            pred = preds('animalORitemORpersonORview.hdf5', 4, path.replace('.mp4', ''))
            lable1 = ['animal', 'item', 'person', 'veiw']
            b = pred.argmax(axis=1)[0]
            print(lable1[b] + ':' + str(pred[0][b]))
            lable_f += lable1[b]
    print('这可能是一个{}类型的视频'.format(lable_f))
def word(data):
    # 不加首行： data = pd.read_csv(txtPath, encoding='utf-8')
    words = jieba.lcut(data)
    data_key = dict(Counter([i for i in words]))
    data_keys_new = [k for k in data_key.keys()]
    w = ''
    for word in data_keys_new:
        w += word
    return w

def testvideopretreat(path):
    cap = cv2.VideoCapture(path)
    frames_num = cap.get(7)
    num = int(frames_num / 40)
    dest = 'frame/'+path.replace('.mp4', '') + '-%04d.jpg'
    dest2=  'ocrdata/'+path.replace('.mp4', '') + '-%04d.jpg'
    call(["ffmpeg", "-i", path, '-vf', "select=not(mod(n\,{a}))".format(a=num - 1), '-vsync', '0', dest])#视频帧
    call(["ffmpeg", "-i", path, '-vf', "select=not(mod(n\,65))", '-vsync', '0', dest2])  # ocr图片
    call(["ffmpeg", "-i", path, '-vn', 'wavdata/'+path.replace('mp4', 'wav')])  # wav音频文件
def wave_ex(wavs):
    n0 = 9000
    n1 = 9100
    x, sr = librosa.load(wavs)
    zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
    print(sum(zero_crossings))
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
    print(sum(spectral_centroids) / len(spectral_centroids))
    return sum(zero_crossings),sum(spectral_centroids) / len(spectral_centroids)
def ocrex(ocrdata):
    images = glob.glob(os.path.join(ocrdata.replace('.mp4', '') + '*.jpg'))
    words = ''
    for img in images:
        img = Image.open(img).convert('RGB')
        # img.show()
        img = np.array(img)
        text = text_predict(img)
        text = list(map(lambda x: x['text'], text))
        box = ''
        for i in text:
            p = re.compile(r"[\u4e00-\u9fa5]+")
            resu = p.findall(i)
            for r in resu:
                box += r
        box=box.replace('抖音','').replace('号','').replace('扫音','').replace('音','')
        if len(box)>2:
            print('['+box+']')
        words += box
    if len(words) < 4:
        words = ''
    words = word(words).replace('北', '').replace('出', '').replace('山', '')
    print(word(words))

def keyword(w1,w2):
    tr4w = TextRank4Keyword()
    tr4w.analyze(w1)
    k1 = ''
    for item in tr4w.get_keywords(num=5, word_min_len=2):  # 提取5个关键词，关键词最少为2个字
        k1 += '#' + item.word
    print(k1)
    tr4w.analyze(w2)
    k2 = ''
    for item in tr4w.get_keywords(num=5, word_min_len=2):  # 提取5个关键词，关键词最少为2个字
        k2 += '#' + item.word
    print(k2)
    return k1,k2
if __name__ == '__main__':
    path = '8.mp4'
    #testvideopretreat(path)
    wave_ex('wavdata/'+path.replace('.mp4','.wav'))
    ocrex('ocrdata/'+path)
    fivelabel(path)
