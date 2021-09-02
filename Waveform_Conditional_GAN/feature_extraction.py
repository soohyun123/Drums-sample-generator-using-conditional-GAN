import sys
import os
import numpy as np
import librosa
from os import listdir

data_path = "/projects/c_drums/"  # './dataset/'
out_path = './feature_c_drums/'

def extract_audio(dataset='train'):
    f = open(data_path + dataset + '_list.txt', 'r')

    i = 0
    error = 0

    for file_name in f:
        # progress check
        i = i + 1
        if not (i % 10):
            print(i)

        # load audio file
        file_name = file_name.rstrip('\n')
        file_path = data_path + file_name

        # print file_path
        y, sr = librosa.load(file_path, sr=16000, mono = True)
        y = librosa.util.fix_length(y, 16384)
        print(y.shape)

        if(y.size!=16384):
            error+=1

        # save log_D
        file_name = file_name.replace('.wav', '.npy')
        save_file = out_path + file_name

        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        np.save(save_file, y)

    if error != 0:
        print('ERROR')

    f.close();

    return 0


if __name__ == '__main__':
    for name in listdir(data_path):
        if os.path.isdir(data_path+name):
            extract_audio(dataset=name)
