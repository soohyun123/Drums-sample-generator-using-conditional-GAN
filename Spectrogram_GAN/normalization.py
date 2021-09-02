import os
import numpy as np
import librosa

data_path = "/projects/Nsynth/"
out_path = './spec/'
out_path_2 = './mean_std/'

def normalized_spec(mean, std, dataset='train'):
    f = open(data_path + dataset + '_list.txt', 'r')

    i = 0
    for file_name in f:
        # progress check
        i = i + 1
        if not (i % 10):
            print(i)

        # load audio file
        file_name = file_name.rstrip('\n')

        # save melspec as a file
        file_name = file_name.replace('.wav', '.npy')
        load_file = out_path + file_name

        log_D = np.load(load_file)
        log_D_scaled = (log_D - mean.T)/(1.5*std.T + 0.00001)
        log_D_scaled = np.clip(log_D_scaled, -1., 1.)

        save_file = './spec_7_norm/' + file_name
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        np.save(save_file, log_D_scaled)

    f.close();

if __name__ == '__main__':

    std = np.load(mean_std_path + 'std.npy')
    mean = np.load(mean_std_path + 'mean.npy')
    normalized_spec(mean, std, 'train')
    normalized_spec(mean, std, 'valid')
    normalized_spec(mean, std, 'test')