'''
提取文件夹视频的音频到当前目录下
'''
import glob
import os
import os.path
from subprocess import call
import librosa
class_folders = glob.glob(os.path.join('bili', '*.mp4'))
for video in class_folders:
    dest = video.replace('.mp4', '') + '-%04d.jpg'
    call(["ffmpeg", "-i", video, '-vf', "select=not(mod(n\,65))", '-vsync', '0', dest])#ocr图片
    call(["ffmpeg", "-i",video,'-vn', video.replace('mp4','wav')])#wav音频文件
    print('success')

'''
每个音频提取特征写入相应文档
'''
n0 = 9000
n1 = 9100
class_folders = glob.glob(os.path.join('bili', '*.wav'))
for wavs in class_folders:
    x, sr = librosa.load(wavs)
    zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
    #print(sum(zero_crossings))
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
    #print(sum(spectral_centroids) / len(spectral_centroids))
    fo=open(wavs.replace('.wav','.txt'),'a')
    fo.write('{}\n'.format(sum(zero_crossings)))
    fo.write('{}\n'.format(sum(spectral_centroids) / len(spectral_centroids)))
    fo.close
    print('wav success')


'''mfccs = librosa.feature.mfcc(x, sr=sr)
hop_length = 512
chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
print(chromagram.shape)'''