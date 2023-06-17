import pandas as pd
import os
import torchaudio
import librosa
all_sound = set(map(lambda x:int(x.split('.')[0]), os.listdir("audio_data_20000_end")))
df = pd.read_csv("dataset/train.csv")
df = df[df['audiocap_id'].isin(all_sound)]
print(len(df.index), len(all_sound))

df.to_csv("dataset/train_final.csv", index=False)
# from audioldm.audio import read_wav_file, wav_to_fbank
# print(os.path.exists("9191.wav"))
# waveform, sample_rate = torchaudio.load("music_data\\0.wav")
# print(waveform.shape, sample_rate)
# data, sampleRate = librosa.load("0.wav")
# print(data.shape, sampleRate)
# waveform, sample_rate = librosa.load("music_data\\1.wav", sr=16000)
# waveform = read_wav_file("270.wav", 1024*160)
# print(waveform.shape)

# from scipy.io import wavfile
# samplerate, data = wavfile.read('trumpet.wav')
# print(samplerate, data.shape)

#Remove files that are not wav
# root_dir = "audio_data_20000_end"
# for file in os.listdir(root_dir):
#     #remove with 0 bytes
#     if(os.path.getsize(os.path.join(root_dir, file)) == 0):
#         # os.remove(os.path.join(root_dir, file))
#         print("Removed", file)
#         continue
    # if(file.endswith(".wav")):
    #     continue
    # # os.remove(os.path.join(root_dir, file))
    # print("Removed", file)