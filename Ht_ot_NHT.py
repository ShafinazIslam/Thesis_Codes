#This code is to classify "houston toad" or "non-toad" call (binary classification)
#import required python libraries
from scipy.io import wavfile
from python_speech_features import mfcc
import pandas as pd
import numpy as np
from keras.layers import Flatten, LSTM
from keras.layers import Dropout, Dense
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import librosa
from keras import optimizers
import matplotlib.pyplot as plt
import time
from keras.models import model_from_json
start_time = time.time()

#imported the csv file with audio file name and label for train data
df1 = pd.read_csv('HT_or_NHT_train.csv')
#imported the csv file with audio file name and label for test data
df2 = pd.read_csv('HT_or_NHT_test.csv')

#column name-"fname-audio files name", it is setting the "file name" as index
df1.set_index('fname', inplace=True)
df2.set_index('fname', inplace=True)

#declaring list for train,test data and train,test label
X_train=[]
y_train=[]
X_test=[]
y_test=[]

# for loop for train data
#setup imax and imix for min-max scalar
#fs1-sampling rate, signal1-original data of audio signal
#Reading the audio wav file from laptop directory using the csv file
#extracting 13 mfcc audio sample for 80 millisecond frame size with 40 millisecond overlap ,applying hamming aindow function for each audio sample
#Extracting first and 2nd derivative of 13 MFCCs
#concatenating total 39 features, concatenating by column
#from the column "label" taking the label for each file name
#appending 39 MFCCs to x_train for each sample
#appending label for each sample to y_train
for f in (df1.index):
    imin, imax = float('inf'), -float('inf')
    fs1, signal1 = wavfile.read('ht_calls_filtered/'+f)
    X_sample = mfcc(signal1, winlen=0.08, winstep= 0.04, numcep=13, nfft=2048, nfilt=26)
    mfcc_delta = librosa.feature.delta(X_sample)
    mfcc_delta2 = librosa.feature.delta(X_sample, order=2)
    mel=np.concatenate((X_sample, mfcc_delta,mfcc_delta2),axis=1)
    label = df1.at[f, 'label']   
    X_train.append(mel)
    y_train.append(label) 
    

# for loop for test data
#setup imax and imix for min-max scalar
#fs1-sampling rate, signal1-original data of audio signal
#Reading the audio wav file from laptop directory using the csv file
#extracting 13 mfcc audio sample for 80 millisecond frame size with 40 millisecond overlap ,applying hamming aindow function for each audio sample
#Extracting first and 2nd derivative of 13 MFCCs
#concatenating total 39 features, concatenating by column
#from the column "label" taking the label for each file name
#appending 39 MFCCs to x_test for each sample
#appending label for each sample to y_test 
for f in (df2.index):
    fs2, signal2 = wavfile.read('ht_calls_filtered/'+f)
    X_sample2 = mfcc(signal2, winlen=0.08, winstep=0.04, numcep=13, nfft=2048, nfilt=26, winfunc=np.hamming)
    mfcc_delta = librosa.feature.delta(X_sample2)
    mfcc_delta2 = librosa.feature.delta(X_sample2, order=2)
    mel=np.concatenate((X_sample, mfcc_delta,mfcc_delta2 ),axis=1)
    label = df2.at[f, 'label']
    X_test.append(mel)
    y_test.append(label)
#converting X_train and y_train to numpy arrays 
X_train, y_train=np.array(X_train), np.array(y_train)
#Reshaping the X_train to a fixed sequence
#X_train.shape[0] is  the number of audio samples(800)
# X_train.shape[1] is the number of frames for each audio sample
#X_train.shape[2] is the number of audio features for each frame
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
#converting X_test and y_test to numpy arrays 
X_test, y_test=np.array(X_test), np.array(y_test)
#Declaring input shape as number of frames for each sample by number of audio features for each frame(149 x 39)
input_shape=(X_train.shape[1], X_train.shape[2])
print(input_shape)
#Declaring the model as sequential
#input shape is time steps x number of features
#One LSTM cell layer with 128 output dimension
#adam optimizer is used with 0.0001 learning rate
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
#model.add(GRU(128, return_sequences=True, input_shape=input_shape))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
adam=optimizers.Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])  

history = model.fit(X_train, y_train, epochs=25, batch_size=32, shuffle=True, validation_split=0.2)

#Saving the model
model_json = model.to_json()
with open("model_ht_or_nht.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('model_ht_or_nht.h5')


#determine and print y prediction
y_pred = model.predict_classes(X_test)

#plotting and saving accuracy curve
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'validation_acc'], loc='best')
plt.show()
plt.savefig('accuracy.png')


#plotting and saving loss curve
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'validation_loss'], loc='best')
plt.show()
plt.savefig('loss.png')
