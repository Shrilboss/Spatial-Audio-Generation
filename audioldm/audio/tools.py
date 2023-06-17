import torch
import numpy as np
import torchaudio
import librosa
import soundfile as sf

def get_stero_from_mono(waveform,swap=1,angle=0):
    [HRIR, fs] = torchaudio.load(f"HRTFs\MIT\elev0\H0e{angle:03d}a.wav")
    waveform_mono = torch.mean(waveform, axis=0)
    s_L_1 = torchaudio.functional.convolve(waveform_mono, HRIR[0,:])
    s_R_1 = torchaudio.functional.convolve(waveform_mono, HRIR[1,:])
    #Swap =1 means right ear is closer to the source
    if(swap):
        return s_R_1.view(1,-1), s_L_1.view(1,-1)
        # bin_mix = torch.vstack((s_L_1, s_R_1))
    else:
        return s_L_1.view(1,-1), s_R_1.view(1,-1)
        # bin_mix = torch.vstack((s_R_1, s_L_1))

def get_mel_from_wav(audio, _stft):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, log_magnitudes_stft, energy = _stft.mel_spectrogram(audio)
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    log_magnitudes_stft = (
        torch.squeeze(log_magnitudes_stft, 0).numpy().astype(np.float32)
    )
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
    return melspec, log_magnitudes_stft, energy


def _pad_spec(fbank, target_length=1024):
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    if fbank.size(-1) % 2 != 0:
        fbank = fbank[..., :-1]

    return fbank


def pad_wav(waveform, segment_length):
    waveform_length = waveform.shape[-1]
    assert waveform_length > 100, "Waveform is too short, %s" % waveform_length
    if segment_length is None or waveform_length == segment_length:
        return waveform
    elif waveform_length > segment_length:
        return waveform[:segment_length]
    elif waveform_length < segment_length:
        temp_wav = np.zeros((1, segment_length))
        temp_wav[:, :waveform_length] = waveform
    return temp_wav

def normalize_wav(waveform):
    waveform = waveform - np.mean(waveform)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    return waveform * 0.5


def process_mono_wav(waveform, sr, segment_length):
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    waveform = waveform.numpy()[0, ...]
    waveform = normalize_wav(waveform)
    waveform = waveform[None, ...]
    waveform = pad_wav(waveform, segment_length)
    
    waveform = waveform / np.max(np.abs(waveform))
    waveform = 0.5 * waveform
    
    return waveform


def read_wav_file(filename,segment_length,angle=None,swap=None, channel=0):
    # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
    waveform, sr = torchaudio.load(filename)  # Faster!!!

    waveform_left, waveform_right = get_stero_from_mono(waveform, swap, angle)
    if channel == 0:
        return process_mono_wav(waveform_left, sr, segment_length)
    return process_mono_wav(waveform_right, sr, segment_length)

    

def wav_to_fbank_mono(waveform,target_length=1024, fn_STFT=None):
    waveform = waveform[0, ...]
    waveform = torch.FloatTensor(waveform)

    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)

    fbank = torch.FloatTensor(fbank.T)
    log_magnitudes_stft = torch.FloatTensor(log_magnitudes_stft.T)

    fbank, log_magnitudes_stft = _pad_spec(fbank, target_length), _pad_spec(
        log_magnitudes_stft, target_length
    )

    return fbank, log_magnitudes_stft, waveform



def wav_to_fbank(filename,target_length=1024, fn_STFT=None,angle=None,swap=None, channel=0):
    assert fn_STFT is not None

    # mixup
    # channel = 0 returns left channel and channel = 1 returns right channel
    waveform = read_wav_file(filename, target_length * 160,angle,swap, channel=channel)  # hop size is 160
    return wav_to_fbank_mono(waveform, target_length, fn_STFT)


    