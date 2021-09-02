import os
import numpy as np
import librosa
import audio_utilities

sr=16000
fft_size = 256
hopsamp = 125
iterations = 300

test_size = 25

_LOG_EPS = 1e-6

sample_dir = './Audio_WGAN_results/Fixed_results'
out_dir = './Audio_WGAN_audios/Fixed_results'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

mean = np.load('./meanstd_7/mean.npy')
std = np.load('./meanstd_7/std.npy')

for epoch in [134]:
    epoch_dir = '/epoch_{0}'.format(epoch + 1)

    if not os.path.exists(out_dir + epoch_dir):
        os.makedirs(out_dir + epoch_dir)

    for i in range(test_size):
        file_name = '/{0}.npy'.format(i)
        log_D_scaled =np.load(sample_dir + epoch_dir + file_name)
        log_D_scaled = np.squeeze(log_D_scaled)

        log_D = log_D_scaled*(1.5*std.T + 0.00001)+ mean.T
        D = (np.exp(log_D) - _LOG_EPS) / 1000
        S = D ** 0.5
        S = np.pad(S, ((0, 1), (0, 0)), 'constant', constant_values=0)

        x_reconstruct = audio_utilities.reconstruct_signal_griffin_lim(S.T
                                                ,fft_size, hopsamp, iterations)

        # The output signal must be in the range [-1, 1], 
        max_sample = np.max(abs(x_reconstruct))
        if max_sample > 1.0:
            x_reconstruct = x_reconstruct / max_sample

        # Save the reconstructed signal to a WAV file.
        out_name = '/{0}.wav'.format(i)
        librosa.output.write_wav(out_dir + epoch_dir + out_name, x_reconstruct, sr)