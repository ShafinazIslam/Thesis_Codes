#this code is to classify "Houston toad" or "Crawfish frog" call (binary classification)

#import required python libraries
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Flatten, LSTM, GRU
from keras.layers import Dropout, Dense
from keras.models import Sequential
from tqdm import tqdm
from python_speech_features import mfcc
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from scipy.signal import hamming
import librosa
from keras import optimizers

#build_rand_feat1 function for training data
#build_rand_feat1 function start
def build_rand_feat1():
    X_train = []
    y_train = []
    #setup _max and _mix for min-max scalar
    _min, _max = float('inf'), -float('inf')
    # for loop
    #rate-sampling rate, wav-original data of audio signal
    #reading audio file from laptop directory
    for file in df1.index:
        rate, wav =  wavfile.read(r'ht_cf_filtered/'+ file)
        label = df1.at[file, 'label']
        #for loop
        #taking one second clip of audio file sequencially
        for i in tqdm(range(0, wav.shape[0], rate)):
            sample=wav[i:i+rate]
            #extracting 13 mfcc audio sample for 80 millisecond frame size with 40 millisecond overlap ,applying hamming aindow function for each audio sample
            X_sample = mfcc(sample, rate, winlen=0.08, winstep=0.04, numcep=13, nfft=2048, nfilt=26,winfunc=np.hamming)
            #Extracting first derivative of 13 MFCCs
            mfcc_delta = librosa.feature.delta(X_sample)
            #Extracting 2nd derivative of 13 MFCCs
            mfcc_delta2 = librosa.feature.delta(X_sample, order=2)
            #concatenating total 39 features, concatenating by column
            mel=np.concatenate((X_sample, mfcc_delta,mfcc_delta2 ),axis=1)
            _min= min(np.amin(mel), _min)
            _max= max(np.amax(mel), _max)
            #appending 39 MFCCs for each sample to X_train
            X_train.append(mel)            
            #appending label for each sample to y_train
            y_train.append(label)
    #converting X_train and y_train to numpy arrays    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = (X_train- _min)/(_max- _min)
    #Reshaping the X_train to a fixed sequence
    #X_train.shape[0] is  the number of audio samples(800)
    # X_train.shape[1] is the number of frames for each audio sample
    #X_train.shape[2] is the number of audio features for each frame
    X_train=X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])   
    return X_train, y_train
#build_rand_feat1 function end
    
#model_prediction function for test data
#model_prediction function start
def model_prediction():
    y_pred=[]
    y_true = []
    #setup _max and _mix for min-max scalar
    _min, _max = float('inf'), -float('inf')
    # for loop
    #rate-sampling rate, wav-original data of audio signal
    #reading audio file from laptop directory
    for file in df2.index:
        rate, wav =  wavfile.read(r'ht_cf_filtered/'+file)
        label = df2.at[file, 'label']
        batch = []
        #for loop
        #taking one second clip of audio file sequencially
        for i in range(0, wav.shape[0], rate):
            sample=wav[i:i+rate]
            #extracting 13 mfcc audio sample for 80 millisecond frame size with 40 millisecond overlap ,applying hamming aindow function for each audio sample
            x = mfcc(sample, rate, winlen=0.08, winstep=0.04, numcep=13, nfft=2048, nfilt=26,winfunc=np.hamming)
            #Extracting first derivative of 13 MFCCs
            mfcc_delta = librosa.feature.delta(x)
            #Extracting 2nd derivative of 13 MFCCs
            mfcc_delta2 = librosa.feature.delta(x, order=2)
            #concatenating total 39 features, concatenating by column
            mel=np.concatenate((x, mfcc_delta, mfcc_delta2 ),axis=1)
            _min= min(np.amin(mel), _min)
            _max= max(np.amax(mel), _max)
            #appending 39 MFCCs to batch
            batch.append(mel)
        #converting batch to numpy array and defined it as X_batch
        X_batch=np.array(batch)
        #predict probability of each one second clip for each audio file
        y_test=model.predict(X_batch)
        #taking mean of probabilities for each audio file
        y_mean=np.mean(y_test, axis=0)
        #if mean > 0.5 the predict 1 else predict 0 and append to y_pred
        if y_mean>0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
        #appending label for each sample to y_true    
        y_true.append(label)  
    return y_true,y_pred
#model_prediction function end
 

#get_recurrent_model function for model preparation
#get_recurrent_model function start
def get_recurrent_model():
    #Declaring the model as sequential
    model = Sequential() 
    #One LSTM cell layer with 128 output dimension
    model.add(LSTM(128, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, input_shape=input_shape))
    #model.add(GRU(128, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, input_shape=input_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2)) #20% dropout
    model.add(Flatten()) #flatten the output from dropout layer
    model.add(Dense(1, activation='sigmoid')) # final output as 1 or 0
    model.summary() #print model summary
    #adam optimizer is used with 0.0001 learning rate
    adam=optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
    return model
#get_recurrent_model function end   

 
#imported the csv file with audio file name and label for train data               
df1 = pd.read_csv(r'ht_craw_train.csv')

#imported the csv file with audio file name and label for test data
df2 = pd.read_csv(r'ht_craw_test.csv')

#column name-"fname-audio files name", it is setting the "file name" as index
df1.set_index('fname', inplace=True)
df2.set_index('fname', inplace=True)

#calling  the function build_rand_feat1 for X_train and y_train
X_train, y_train = build_rand_feat1()

#input shape is time steps x number of features
input_shape=(X_train.shape[1], X_train.shape[2])

#calling  the function get_recurrent_model
model= get_recurrent_model()

history = model.fit(X_train, y_train, epochs=2, batch_size=32, shuffle=True, validation_split=0.2)

#Saving the model
model_json = model.to_json()
with open("model_cf_or_ht.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('model_cf_or_ht.h5')

#calling  the function model_prediction for y_true and y_pred
y_true, y_pred= model_prediction()

#plot and save "Confusion matrix"
plt.figure(1)
print('\nConfusion Matrix')
confmat= (confusion_matrix(y_true, y_pred))
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')   
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

#plot and save "Accuracy_loss curve"
plt.figure(2)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model Accuracy & Loss')
plt.ylabel('accuracy')
plt.ylabel('accuracy_loss')
plt.xlabel('epoch')
plt.legend(['train_acc', 'validation_acc','train_loss', 'validation_loss'], loc='best')
plt.show()
plt.savefig('accuracy_loss.png')
