#This code is to predict "toad" or "non-toad" call in real time
#import required python library
import sounddevice
from scipy.io import wavfile
from scipy import signal
import librosa
from python_speech_features import mfcc
from keras.models import model_from_json
from keras import optimizers
import numpy as np

#define sampling rate as fs
fs=16000
#define recording duration
second=6
print("recording")
#record audio and save it in directory as "output.wav"
record_voice=sounddevice.rec(int(second*fs),samplerate=fs, channels=1)
sounddevice.wait()
wavfile.write(r'output.wav', fs, record_voice)
#read the saved audio file "output.wav"
fs1, signal1 = wavfile.read(r'output.wav')
#apply filtering to the saved audio and save it as output1.wav
nyquist = 0.5 * fs1
lowcut1=1500
highcut1= 2600
low1 = 2*lowcut1 / fs1
high1 = 2*highcut1 / fs1  
d, c = signal.butter(10, [low1, high1], btype='bandpass')
M_coupeband1 = signal.filtfilt(d, c, signal1)
wavfile.write(filename=r'output1.wav', rate=fs1, data=M_coupeband1)

X_test=[]
y_pred=[]

_min, _max = float('inf'), -float('inf')
#read the saved audio file "output1.wav"
fs, signal = wavfile.read(r'output1.wav')

X_sample2 = mfcc(signal, samplerate=8000, winlen=0.08, winstep=0.04, numcep=13, nfft=2048, nfilt=26, winfunc=np.hamming)
mfcc_delta = librosa.feature.delta(X_sample2)
mfcc_delta2 = librosa.feature.delta(X_sample2, order=2)
mel=np.concatenate((X_sample2, mfcc_delta,mfcc_delta2 ),axis=1)
X_test.append(mel)
X_test=np.array(X_test)
# load json and create model
json_file1 = open('model_ht_or_nht.json', 'r')
loaded_model_json1 = json_file1.read()
json_file1.close()
loaded_model1 = model_from_json(loaded_model_json1)
# load weights into new model
loaded_model1.load_weights("model_ht_or_nht.h5")

 
# evaluate loaded model on test data
adam=optimizers.Adam(learning_rate=0.0001)
loaded_model1.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
y_pred = loaded_model1.predict_classes(X_test)
print(y_pred)
