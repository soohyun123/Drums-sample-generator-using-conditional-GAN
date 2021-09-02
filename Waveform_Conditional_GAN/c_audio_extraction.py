import sys
import os
import numpy as np
import librosa
#import audio_utilities

sr=16000

test_size = 10

label = 8

sample_dir = "./Wave_C_WGAN_results/drums_1/Fixed_results"
out_dir = "./Wave_C_WGAN_audios/drums_1/Fixed_results"

if not os.path.exists(os.path.dirname(out_dir)):
    os.makedirs(os.path.dirname(out_dir))

for epoch in [1500, 2000]:
    epoch_dir = '/epoch_{0}'.format(epoch)

    if not os.path.exists(out_dir + epoch_dir):
        os.makedirs(out_dir + epoch_dir)

    for j in range(label):
        label_dir = '/label_{0}'.format(j)

        if not os.path.exists(out_dir + epoch_dir+label_dir):
            os.makedirs(out_dir + epoch_dir+label_dir)

        for i in range(test_size):
            file_name = '/{0}.npy'.format(i)
            audio =np.load(sample_dir + epoch_dir + label_dir + file_name)
            audio = np.squeeze(audio)

            # The output signal must be in the range [-1, 1], otherwise we need to clip or normalize.
            max_sample = np.max(abs(audio))
            if max_sample > 1.0:
                audio = audio / max_sample

            # Save the reconstructed signal to a WAV file.
            out_name = '/{0}.wav'.format(i)
            librosa.output.write_wav(out_dir + epoch_dir + label_dir + out_name, audio, sr)
    #        audio_utilities.save_audio_to_file(x_reconstruct, sr, outfile = out_dir + epoch_dir + out_name)

            if(i == 0):
                full_audio = audio
            else:
                full_audio = np.concatenate((full_audio,audio), axis=None)

            out_name = '/full.wav'
            librosa.output.write_wav(out_dir + epoch_dir + label_dir + out_name, full_audio, sr)
            #audio_utilities.save_audio_to_file(full_audio, sr, outfile=out_dir + epoch_dir + out_name)