# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 18:21:00 2021

@author: Sarvagya Kumar
Create log mel band energies with window length of 0.04 and hop length of 0.02 and 40 filter banks
"""

DATA_PATH = 'E:/Acoustic scene/DCASE2017-baseline-system-master/applications/data/TUT-acoustic-scenes-2017-development/dir'

FBANK_DEFAULT = {
    'type': 'fbank',
    'win_length_seconds': 0.04,
    'hop_length_seconds': 0.02,
    'bands': 40,
    'fmin': 0,  # Minimum frequency when constructing MEL bands
    'fmax': 22050, # Maximum frequency when constructing MEL band
    'n_fft': 2048, # number of FFT components
    'mono': True,
    'window': 'hamming_asymmetric'  # [hann_asymmetric, hamming_asymmetric] 
}

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


class generate_features:
    def __init__(self,folds):
        self.signal=None
        self.sampling_rate=0
        self.folds=folds
        self.params = FBANK_DEFAULT
        self.count=0
    def make_signal(self, recording, filename):
        """ The following 3 functions are for displaying stft, mel filter bank and log energy bands for the first feature file for visualization purposes"""
        def disp_stft(stft):
            fig, ax = plt.subplots()
            img = librosa.display.specshow(stft, y_axis='log', x_axis='time', ax=ax)
            ax.set_title('Power spectrogram')
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
        
        def disp_mel(mel_basis):
            fig, ax = plt.subplots()
            img = librosa.display.specshow(mel_basis, x_axis='linear', ax=ax)
            ax.set(ylabel='Mel filter', title='Mel filter bank')
            fig.colorbar(img, ax=ax)
        
        def disp_logmel(logmel):
            fig, ax = plt.subplots()
            img = librosa.display.specshow(logmel, x_axis='time', y_axis='mel', ax=ax)
            ax.set(title='Log mel band')
            fig.colorbar(img, ax=ax, format="%+2.f dB")
            
        #create an array from the wav file and get sampling rate   
        audio, sr = librosa.core.load(recording, sr=44100, dtype=np.float32, duration=10.0)
        wnd_len = int(self.params['win_length_seconds'] * sr) 
        hop_len = int(self.params['hop_length_seconds'] * sr)
        
        #create short term fourier transform
        stft = np.abs(librosa.stft(audio,
                                   n_fft=self.params['n_fft'],
                                   win_length=wnd_len,
                                   hop_length=hop_len,))
        
        #create mel filter banks (40 in number)
        mel_basis = librosa.filters.mel(sr=sr,
                                        n_fft=self.params['n_fft'],
                                        n_mels=self.params['bands'],
                                        fmin=self.params['fmin'],
                                        fmax=self.params['fmax'])
        #create mel energy band
        mel_spec = np.dot(mel_basis, stft)
        
        #log energy band of dimension 40x501
        logmel = librosa.core.amplitude_to_db(mel_spec)
        
        #add a new dimension to make it 40x501x1
        elogmel = logmel[:, :, np.newaxis]
        
        #for first recording
        if self.count == 0:
            disp_stft(stft)
            disp_mel(mel_basis)
            disp_logmel(logmel)
            self.count=1
            
        #save logmel feature in path 'filename'
        np.save(filename, elogmel.astype('float16'), allow_pickle=False)
        
        

        
        
    
        