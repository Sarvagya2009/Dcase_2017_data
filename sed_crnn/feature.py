import wave
import numpy as np
import utils
import librosa
from IPython import embed
import os
from sklearn import preprocessing


########################################
#Loading data and extracting mbe based on labels from meta data
########################################

def load_audio(filename, mono=True, fs=44100):
    """Load audio file into numpy array
    Supports 24-bit wav-format
    
    Taken from TUT-SED system: https://github.com/TUT-ARG/DCASE2016-baseline-system-python
    
    Parameters
    ----------
    filename:  str
        Path to audio file

    mono : bool
        In case of multi-channel audio, channels are averaged into single channel.
        (Default value=True)

    fs : int > 0 [scalar]
        Target sample rate, if input audio does not fulfil this, audio is resampled.
        (Default value=44100)

    Returns
    -------
    audio_data : numpy.ndarray [shape=(signal_length, channel)]
        Audio

    sample_rate : integer
        Sample rate

    """

    file_base, file_extension = os.path.splitext(filename) #split to file address and extension
    if file_extension == '.wav':
        _audio_file = wave.open(filename)

        # Audio info
        sample_rate = _audio_file.getframerate() 
        sample_width = _audio_file.getsampwidth()
        number_of_channels = _audio_file.getnchannels() #Returns number of audio channels (1 for mono, 2 for stereo).
        number_of_frames = _audio_file.getnframes()

        # Read raw bytes
        data = _audio_file.readframes(number_of_frames) #Reads and returns at most 'number_of_frames' frames of audio, as a bytes object.
        _audio_file.close()

        # Convert bytes based on sample_width
        num_samples, remainder = divmod(len(data), sample_width * number_of_channels)
        if remainder > 0:
            raise ValueError('The length of data is not a multiple of sample size * number of channels.')
        if sample_width > 4:
            raise ValueError('Sample size cannot be bigger than 4 bytes.')

        if sample_width == 3:
            # 24 bit audio  since Numpy doesn’t have a 24-bit integer dtype, a conversion step is needed.
            a = np.empty((num_samples, number_of_channels, 4), dtype=np.uint8) 
            raw_bytes = np.fromstring(data, dtype=np.uint8) #convert to unsigned int array
            a[:, :, :sample_width] = raw_bytes.reshape(-1, number_of_channels, sample_width) #half of a is filled 
            a[:, :, sample_width:] = (a[:, :, sample_width - 1:sample_width] >> 7) * 255
            audio_data = a.view('<i4').reshape(a.shape[:-1]).T #some preprocessing
        else:
            # 8 bit samples are stored as unsigned ints; others as signed ints.
            dt_char = 'u' if sample_width == 1 else 'i' 
            a = np.fromstring(data, dtype='<%s%d' % (dt_char, sample_width))
            audio_data = a.reshape(-1, number_of_channels).T

        if mono:
            # Down-mix audio
            audio_data = np.mean(audio_data, axis=0)

        # Convert int values into float
        audio_data = audio_data / float(2 ** (sample_width * 8 - 1) + 1)

        # Resample
        if fs != sample_rate:
            audio_data = librosa.core.resample(audio_data, sample_rate, fs)
            sample_rate = fs

        return audio_data, sample_rate
    return None, None


def load_desc_file(_desc_file, __class_labels): #returns a dict for each wav file containing
#([start time] [end time] [class label]) for all scenes in the file FROM THE METADATA
#KEY will be the file name
    _desc_dict = dict()
    for line in open(_desc_file):
        words = line.strip().split('\t')
        name = words[0].split('/')[-1]
        if name not in _desc_dict:
            _desc_dict[name] = list()
        _desc_dict[name].append([float(words[2]), float(words[3]), __class_labels[words[-1]]])
    return _desc_dict

#sample frame and mel band
def extract_mbe(_y, _sr, _nfft, _nb_mel):
    spec, n_fft = librosa.core.spectrum._spectrogram(y=_y, n_fft=_nfft, hop_length=_nfft//2, power=1)
    mel_basis = librosa.filters.mel(sr=_sr, n_fft=_nfft, n_mels=_nb_mel)
    return np.log(np.dot(mel_basis, spec)) #log mel band


