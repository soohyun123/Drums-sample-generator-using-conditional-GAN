import os
import numpy as np
import librosa

_CLIP_NSTD = 3.
_LOG_EPS = 1e-6

data_path = "/projects/Nsynth/"
out_path = './spec_7/'
out_path2 = './mean_std/'

fft_size = 256
hopsamp = 125

def total_mean_std(dataset='train'):
    f = open(data_path + dataset + '_list.txt', 'r')

    i = 0
    n_frame = 0
    sum_spec = np.zeros((1,128))
    square_spec = np.zeros((1,128))

    for file_name in f:
        # progress check
        i = i + 1
        if not (i % 10):
            print(i)

        # load audio file
        file_name = file_name.rstrip('\n')
        file_path = data_path + file_name

        # print file_path
        y, sr = librosa.load(file_path, sr=16000, duration= 1)

        # STFT
        S = librosa.core.stft(y, n_fft=fft_size, hop_length=hopsamp
                            ,win_length=fft_size)

        # power spectrogram
        D = np.abs(S)**2

        #log compression
        log_D = np.log(D + _LOG_EPS)
        log_D = log_D[:128,:128] 

        #save log_D
        file_name = file_name.replace('.wav', '.npy')
        save_file = out_path + file_name
        
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        np.save(save_file, log_D)   

        #Total normalization

        n_frame += log_D.shape[1]
        sum_spec += np.sum(log_D, axis=1)
        square_spec += np.sum(np.power(log_D, 2), axis=1)

    f.close();

    return n_frame, sum_spec, square_spec


if __name__ == '__main__':
    n_frame = 0
    sum_spec = np.zeros((1, 128))
    square_spec = np.zeros((1, 128))

    a, b, c = total_mean_std(dataset='train')
    n_frame += a
    sum_spec += b
    square_spec += c

    a, b, c = total_mean_std(dataset='valid')
    n_frame += a
    sum_spec += b
    square_spec += c

    a, b, c = total_mean_std(dataset='test')
    n_frame += a
    sum_spec += b
    square_spec += c

    total_mean = sum_spec/n_frame

    save_mean = out_path2 + 'mean.npy'
    if not os.path.exists(os.path.dirname(save_mean)):
        os.makedirs(os.path.dirname(save_mean))

    np.save(out_path_2 + 'mean.npy', total_mean)

    total_std = np.sqrt(square_spec/n_frame - np.power(total_mean,2))

    save_std = out_path2 + 'std.npy'
    np.save(save_std, total_std)