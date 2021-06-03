from __future__ import print_function
import numpy as np
import os
from sklearn import preprocessing
import time
import sys
import matplotlib.pyplot as plot
import keras
from keras.layers import Bidirectional, TimeDistributed, Conv2D, MaxPooling2D, Input, LSTM, Dense, Activation, Dropout, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import keras.backend as K
K.set_image_data_format('channels_last')
sys.setrecursionlimit(10000) #Set the maximum depth of the Python interpreter stack to n.
import metrics
import utils


"""1. Audio data preprocessing and extracting log mel band energies"""
import feature #use custom functions from the feature file

is_mono = False #set mono as true or false
__class_labels = {
    'brakes squeaking': 0,
    'car': 1,
    'children': 2,
    'large vehicle': 3,
    'people speaking': 4,
    'people walking': 5
}

# location of data.
folds_list = [1, 2]
evaluation_setup_folder = r'E:\Acoustic #scene\sed_crnn\sed-crnn-master\TUT-sound-events-2017-development.meta\TUT-sound-events-2017-development\evaluation_set#up'
audio_folder = r'E:\Acoustic #scene\sed_crnn\sed-crnn-master\TUT-sound-events-2017-development.audio.1\TUT-sound-events-2017-development\audio\stree#t'

# Output
feat_folder = r'E:\Acoustic scene\sed_crnn\sed-crnn-master\feat'
utils.create_folder(feat_folder)

# User set parameters
nfft = 2048 
win_len = nfft
hop_len = win_len // 2 #hop length of filter
nb_mel_bands = 40 #total mel filters
sr = 44100 #sampling rate

# -----------------------------------------------------------------------
# Feature extraction and label generation
# -----------------------------------------------------------------------

# Load labels
#set to True if feature extraction is needed
extract= False
if extract:
    train_file = os.path.join(evaluation_setup_folder, 'street_fold{}_train.txt'.format(1))
    evaluate_file = os.path.join(evaluation_setup_folder, 'street_fold{}_evaluate.txt'.format(1))
    desc_dict = feature.load_desc_file(train_file, __class_labels) #make dict
    desc_dict.update(feature.load_desc_file(evaluate_file, __class_labels)) # contains labels for all the audio in the dataset

    #till here labels are stored for only 1st fold

# Extract features for all audio files, and save it along with labels
    for audio_filename in os.listdir(audio_folder):
        audio_file = os.path.join(audio_folder, audio_filename)
        print('Extracting features and label for : {}'.format(audio_file))
        y, sr = feature.load_audio(audio_file, mono=is_mono, fs=sr) #y = audio data [shape=(signal_length, channel)]
        mbe = None
        
        #now we extract mel band energies for mono or binaural audio
        if is_mono:
            mbe = feature.extract_mbe(y, sr, nfft, nb_mel_bands).T 
        else:
            for ch in range(y.shape[0]): #for each channel extract mbe
                mbe_ch = feature.extract_mbe(y[ch, :], sr, nfft, nb_mel_bands).T
                if mbe is None:
                    mbe = mbe_ch
                else:
                    mbe = np.concatenate((mbe, mbe_ch), 1)

        label = np.zeros((mbe.shape[0], len(__class_labels)))
        tmp_data = np.array(desc_dict[audio_filename]) 
        #for all sequences, extract frame start, end and labels
        frame_start = np.floor(tmp_data[:, 0] * sr / hop_len).astype(int) 
        frame_end = np.ceil(tmp_data[:, 1] * sr / hop_len).astype(int)
        se_class = tmp_data[:, 2].astype(int) #label
        for ind, val in enumerate(se_class): #for each class found in the file
            label[frame_start[ind]:frame_end[ind], val] = 1 #value in between the frame start and end for that class is 1
        tmp_feat_file = os.path.join(feat_folder, '{}_{}.npz'.format(audio_filename, 'mon' if is_mono else 'bin'))
        np.savez(tmp_feat_file, mbe, label) #store the mel band and the corresponding label as a file
        #save npz file for each audio file
        
# -----------------------------------------------------------------------
# Feature Normalization
# -----------------------------------------------------------------------

    for fold in folds_list:
        #for each fold, extract frame start, end with labels
        train_file = os.path.join(evaluation_setup_folder, 'street_fold{}_train.txt'.format(fold))
        evaluate_file = os.path.join(evaluation_setup_folder, 'street_fold{}_evaluate.txt'.format(fold))
        train_dict = feature.load_desc_file(train_file, __class_labels)
        test_dict = feature.load_desc_file(evaluate_file, __class_labels)

        X_train, Y_train, X_test, Y_test = None, None, None, None 
        for key in train_dict.keys(): #for each file in the fold
            tmp_feat_file = os.path.join(feat_folder, '{}_{}.npz'.format(key, 'mon' if is_mono else 'bin'))
            dmp = np.load(tmp_feat_file) #load saved features with labels
            tmp_mbe, tmp_label = dmp['arr_0'], dmp['arr_1']
            if X_train is None:
                X_train, Y_train = tmp_mbe, tmp_label
            else:
                X_train, Y_train = np.concatenate((X_train, tmp_mbe), 0), np.concatenate((Y_train, tmp_label), 0)

        for key in test_dict.keys():
            tmp_feat_file = os.path.join(feat_folder, '{}_{}.npz'.format(key, 'mon' if is_mono else 'bin'))
            dmp = np.load(tmp_feat_file)
            tmp_mbe, tmp_label = dmp['arr_0'], dmp['arr_1']
            if X_test is None:
                X_test, Y_test = tmp_mbe, tmp_label
            else:
                X_test, Y_test = np.concatenate((X_test, tmp_mbe), 0), np.concatenate((Y_test, tmp_label), 0)

        # Normalize the training data of all folds combined, and scale the testing data using the training data weights
        scaler = preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train) 
        X_test = scaler.transform(X_test)

        normalized_feat_file = os.path.join(feat_folder, 'mbe_{}_fold{}.npz'.format('mon' if is_mono else 'bin', fold))
        np.savez(normalized_feat_file, X_train, Y_train, X_test, Y_test) #for each fold
        print('normalized_feat_file : {}'.format(normalized_feat_file))

"""2. Functions for model training"""

def load_data(_feat_folder, _mono, _fold=None):
    feat_file_fold = os.path.join(_feat_folder, 'mbe_{}_fold{}.npz'.format('mon' if _mono else 'bin', _fold))
    dmp = np.load(feat_file_fold)
    _X_train, _Y_train, _X_test, _Y_test = dmp['arr_0'],  dmp['arr_1'],  dmp['arr_2'],  dmp['arr_3']
    return _X_train, _Y_train, _X_test, _Y_test


# -----------------------------------------------------------------------
# Create model architecture of Conv2D(128), batchnormalization and activation
# Followed by max pooliing with pooling size 1,5. This is followed by bidirectional LSTM layers and
# Time distributed FC layers. Time distributed layers apply to each temporal sequence seperately. 
# -----------------------------------------------------------------------


def get_model(data_in, data_out, _cnn_nb_filt, _cnn_pool_size):

    spec_start = Input(shape=(data_in.shape[-3], data_in.shape[-2], data_in.shape[-1]))
    spec_x = spec_start
    for _i, _cnt in enumerate(_cnn_pool_size):
        spec_x = Conv2D(filters=_cnn_nb_filt, kernel_size=(3, 3), padding='same')(spec_x)
        spec_x = BatchNormalization(axis=1)(spec_x)
        spec_x = Activation('relu')(spec_x)
        spec_x = MaxPooling2D(pool_size=(1, _cnn_pool_size[_i]))(spec_x)
        spec_x = Dropout(dropout_rate)(spec_x)
    spec_x = Permute((2, 1, 3))(spec_x) #Permutes the dimensions of the input according to a given pattern. Useful e.g. connecting RNNs and convnets.
    spec_x = Reshape((data_in.shape[-2], -1))(spec_x)
    

    spec_x = Bidirectional(LSTM(100, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True),
            merge_mode='concat')(spec_x)
    spec_x = Bidirectional(LSTM(100, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True),
            merge_mode='concat')(spec_x)
        

    spec_x = TimeDistributed(Dense(32))(spec_x)
    spec_x = Dropout(dropout_rate)(spec_x)
    

    spec_x = TimeDistributed(Dense(data_out.shape[-1]))(spec_x)
    out = Activation('sigmoid', name='strong_out')(spec_x)

    _model = Model(inputs=spec_start, outputs=out)
    _model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics= ['accuracy'])
    print(_model.summary())
    return _model

# -----------------------------------------------------------------------
# Plot metrics with epoch 
# -----------------------------------------------------------------------


def plot_functions(_nb_epoch, _tr_loss, _val_loss, _f1, _er, extension=''):
    plot.figure()

    plot.subplot(211)
    plot.plot(range(_nb_epoch), _tr_loss, label='train loss')
    plot.plot(range(_nb_epoch), _val_loss, label='val loss')
    plot.legend()
    plot.grid(True)

    plot.subplot(212)
    plot.plot(range(_nb_epoch), _f1, label='f')
    plot.plot(range(_nb_epoch), _er, label='er')
    plot.legend()
    plot.grid(True)

    plot.savefig(__models_dir + __fig_name + extension)
    plot.close()
    print('figure name : {}'.format(__fig_name))

# -----------------------------------------------------------------------
# Preprocess data for inputing to model
# input format: samples x 1 x sequece length of frame x no of channels
# output format: samples x sequence length x number of labels
# -----------------------------------------------------------------------


def preprocess_data(_X, _Y, _X_test, _Y_test, _seq_len, _nb_ch):
    # split into sequences
    
    _X = utils.split_in_seqs(_X, _seq_len) 
    _Y = utils.split_in_seqs(_Y, _seq_len)

    _X_test = utils.split_in_seqs(_X_test, _seq_len)
    _Y_test = utils.split_in_seqs(_Y_test, _seq_len)

    _X = utils.split_multi_channels(_X, _nb_ch)
    _X_test = utils.split_multi_channels(_X_test, _nb_ch)
    return _X, _Y, _X_test, _Y_test


"""3. Defining the model parameters"""

is_mono = False  # True: mono-channel input, False: binaural input

feat_folder = 'feat'
__fig_name = '{}_{}'.format('mon' if is_mono else 'bin', time.strftime("%Y_%m_%d_%H_%M_%S"))


nb_ch = 1 if is_mono else 2
batch_size = 64   # Decrease this if you want to run on smaller GPU's
seq_len = 256       # Frame sequence length. Input to the CRNN.
nb_epoch = 350  # Training epochs
patience = int(0.25 * nb_epoch)  # Patience for early stopping

# Number of frames in 1 second, required to calculate F and ER for 1 sec segments.
# Make sure the nfft and sr are the same as in feature.py
sr = 44100
nfft = 2048
frames_1_sec = int(sr/(nfft/2.0)) #resolution

print('\n\nUNIQUE ID: {}'.format(__fig_name))
print('TRAINING PARAMETERS: nb_ch: {}, seq_len: {}, batch_size: {}, nb_epoch: {}, frames_1_sec: {}'.format(
    nb_ch, seq_len, batch_size, nb_epoch, frames_1_sec))

# Folder for saving model and training curves
__models_dir = 'models/'
utils.create_folder(__models_dir)

# CRNN model definition
cnn_nb_filt = 128            # CNN filter size
cnn_pool_size = [5, 2, 2]   # Maxpooling across frequency. Length of cnn_pool_size =  number of CNN layers
dropout_rate = 0.5          # Dropout after each layer

"""4. Model training"""

tr_accuracy, val_accuracy = [0] * len(folds_list), [0] * len(folds_list)
tr_loss, val_loss, f1_overall_1sec_list, er_overall_1sec_list = [0] * len(folds_list), [0] * len(folds_list), [0] * len(folds_list), [0] * len(folds_list)

#running for each fold
for fold in folds_list:
    print('\n\n----------------------------------------------')
    print('FOLD: {}'.format(fold))
    print('----------------------------------------------\n')
    # Load feature and labels, pre-process it
    X, Y, X_test, Y_test = load_data(feat_folder, is_mono, fold)
    X, Y, X_test, Y_test = preprocess_data(X, Y, X_test, Y_test, seq_len, nb_ch)

    
    posterior_thresh = 0.5
    # Load model
    model = get_model(X, Y, cnn_nb_filt, cnn_pool_size)
    es= EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1)
    hist = model.fit(
            X, Y,
            batch_size=batch_size,
            validation_data=(X_test, Y_test),
            epochs=nb_epoch,
            verbose=1,
            callbacks=[es], shuffle= True)
    #get loss and accuracy for each fold
    val_loss[fold-1] = hist.history.get('val_loss')[-1]
    tr_loss[fold-1] = hist.history.get('loss')[-1]
    val_accuracy[fold-1] = hist.history.get('val_loss')[-1]
    tr_accuracy[fold-1] = hist.history.get('loss')[-1]

    """5. Predictions and metrics"""

    
    # Calculate the predictions on test data, in order to calculate ER and F scores
    pred = model.predict(X_test)
    pred_thresh = pred > posterior_thresh

# -----------------------------------------------------------------------
# Compute f1 score for that fold using metrics.py
# -----------------------------------------------------------------------

    
    score_list = metrics.compute_scores(pred_thresh, Y_test, frames_in_1_sec=frames_1_sec)
    f1_overall_1sec_list[fold-1] = score_list['f1_overall_1sec']
    er_overall_1sec_list[fold-1] = score_list['er_overall_1sec']

    # Calculate confusion matrix
    #test_pred_cnt = np.sum(pred_thresh, 2)
    #Y_test_cnt = np.sum(Y_test, 2)
    #conf_mat = confusion_matrix(Y_test_cnt.reshape(-1), test_pred_cnt.reshape(-1))
    #conf_mat = conf_mat / (utils.eps + np.sum(conf_mat, 1)[:, None].astype('float'))
    

    print('train loss : {}, val loss : {}, F1_fold : {}, ER_fold : {}'.format(tr_loss[fold-1], val_loss[fold-1], f1_overall_1sec_list[fold-1], er_overall_1sec_list[fold-1]))
    #plot_functions(nb_epoch, tr_loss, val_loss, f1_overall_1sec_list, er_overall_1sec_list, '_fold_{}'.format(fold))
    print("saving model")
    model.save(os.path.join(__models_dir, '{}_fold_{}_model.h5'.format(__fig_name, fold)))
    
    #plot training and test accuracy
    train_acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    plot.figure()
    plot.plot(train_acc, 'b', label='Training accuracy')
    plot.plot(val_acc, 'r', label ='Validation accuracy')
    plot.title('Training and Validation Accuracy for fold_{}'.format(fold))
    plot.xlabel('epoch')
    plot.ylabel('accuracy_value')
    plot.legend()
    plot.savefig('accuracy_{}.png'.format(fold))
    plot.show()

# -----------------------------------------------------------------------
# Metrics over all folds
# -----------------------------------------------------------------------


print(tr_accuracy, "train accuracy over all folds")
print(val_accuracy, "val accuracy over all folds")
print(tr_loss, "train loss over all folds")
print(val_loss, "val loss over all folds")
print(f1_overall_1sec_list, "f1_overall over all folds")
print(er_overall_1sec_list, "er_overall over all folds")

# -----------------------------------------------------------------------
# Average metrics over all folds
# -----------------------------------------------------------------------

print('MODEL AVERAGE OVER FOUR FOLDS: avg_er: {}, avg_f1: {}, avg_acc: {}'.format(np.mean(er_overall_1sec_list), np.mean(f1_overall_1sec_list), np.mean(val_accuracy)))

file = open("overall_metrics.txt", "w+")
file.write("avg er = " + str(np.mean(er_overall_1sec_list)) + "\n" +"avg f1 = "+ str(np.mean(f1_overall_1sec_list)) + "\n"+ "Validation accuracy= "+ str(np.mean(val_accuracy)))
file.close()