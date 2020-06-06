import codecs
import csv
import glob
import os

def insert(a):
    b = []
    l = len(a)
    if l < 2:
        return ''
    for n in range(l):
        if n % 2 == 0:
            b.append(a[n:n + 2])
    return('#'+'#'.join(b))


folders = ['F:/mutimedia/ICME2019-多模态视频理解/bilibili-smallvideo-master/bili']
for folder in folders:
    class_folders = glob.glob(os.path.join(folder, '*.txt'))
    for file in class_folders:
        fo = open(file, 'r')
        f = open('train2.csv', 'a', newline='')
        lines = fo.readlines()
        if len(lines) > 6:
            s=file.replace('F:/mutimedia/ICME2019-多模态视频理解/bilibili-smallvideo-master/bili\\','')
            print(s)
            csv_writer = csv.writer(f)
            csv_writer.writerow([s.replace('.txt',''), lines[0], lines[4], lines[5], lines[6], lines[2], lines[3], lines[1]])
        fo.close