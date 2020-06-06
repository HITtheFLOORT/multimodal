import csv
import glob
import os
import re

import numpy as np
from subprocess import call
import jieba
import os
from collections import Counter
from PIL import Image
from textrank4zh import TextRank4Keyword

from ocr import text_predict

def testvideopretreat(path):

    dest = path.replace('.mp4', '') + '-%04d.jpg'
    call(["ffmpeg", "-i", path, '-vf', "select=not(mod(n\,65))", '-vsync', '0', dest])

def word(data):
    # 不加首行： data = pd.read_csv(txtPath, encoding='utf-8')
    words = jieba.lcut(data)
    data_key = dict(Counter([i for i in words]))
    data_keys_new = [k for k in data_key.keys()]
    w = ''
    for word in data_keys_new:
        w += word
    return w



folders = ['F:/mutimedia/ICME2019-多模态视频理解/bilibili-smallvideo-master/bili']
for folder in folders:
    class_folders = glob.glob(os.path.join(folder, '*.txt'))
    for files in class_folders:
        images=glob.glob(os.path.join(files.replace('.txt','')+'*.jpg'))
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
            words += box
        if len(words)<4:
            words=''
        words=word(words).replace('北','').replace('出','').replace('山','')
        print(word(words))
        fo=open(files,'a')
        fo.write(word(words)+'\n')
        fo.close

    tr4w = TextRank4Keyword()

    for files in class_folders:
        fo = open(files, 'r')
        lines=fo.readlines()
        fo.close
        if len(lines)>4:
            fu = open(files, 'a')
            tr4w.analyze(lines[0])
            w=''
            for item in tr4w.get_keywords(num=5, word_min_len=2):  # 提取5个关键词，关键词最少为2个字
                w+='#'+item.word
            print(w)
            fu.write(w + '\n')
            tr4w.analyze(lines[4])
            w = ''
            for item in tr4w.get_keywords(num=5, word_min_len=2):  # 提取5个关键词，关键词最少为2个字
                w += '#'+item.word
            print(w)
            fu.write(w + '\n')
            fu.close()






