# -*- coding: utf-8 -*-
"""
@author: Orthy
"""

#impot requiredpython libraries
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.io import wavfile
from scipy import signal
import librosa

#imported the csv file with audio file name
df = pd.read_csv(r'HT_or_NHT_train.csv')
#df = pd.read_csv(r'HT_or_NHT_test.csv')
#df = pd.read_csv(r'ht_craw_train.csv')
#df = pd.read_csv(r'ht_craw_test.csv')
#df = pd.read_csv(r'three_class_train.csv')
#df = pd.read_csv(r'three_class_test.csv')


#for loop
#Reading the audio wav file from laptop directory using the csv file
#sr=sampling rate or frequency
#Filtered audio file saved into a laptop directory    
for f in (df.fname):
    signals1, rate1 = librosa.load('ht_calls/'+ f, sr=16000)
    #signals1, rate1 = librosa.load('ht_calls/'+ f, sr=16000)
    #signals1, rate1 = librosa.load('ht_cf/'+ f, sr=16000)
    #signals1, rate1 = librosa.load('ht_cf/'+ f, sr=16000)
    #signals1, rate1 = librosa.load('three_class/'+ f, sr=16000)
    #signals1, rate1 = librosa.load('three_class/'+ f, sr=16000)
    nyquist = 0.5 * rate1
    lowcut1=300 #lower frequency
    highcut1= 1600 #higherfrequency
    low1 = 2*lowcut1 / rate1
    high1 = 2*highcut1 / rate1  
    d, c = signal.butter(8, [low1, high1], btype='bandpass')
    M_coupeband1 = signal.filtfilt(d, c, signals1)
    librosa.output.write_wav(r'ht_calls_filtered/'+f, y=M_coupeband1, sr=rate1)
    #librosa.output.write_wav(r'ht_calls_filtered/'+f, y=M_coupeband1, sr=rate1)
    #librosa.output.write_wav(r'ht_cf_filtered/'+f, y=M_coupeband1, sr=rate1)
    #librosa.output.write_wav(r'ht_cf_filtered/'+f, y=M_coupeband1, sr=rate1)
    #librosa.output.write_wav(r'three_class_filtered/'+f, y=M_coupeband1, sr=rate1)
    #librosa.output.write_wav(r'three_class_filtered/'+f, y=M_coupeband1, sr=rate1)