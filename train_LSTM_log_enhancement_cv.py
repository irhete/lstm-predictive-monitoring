#!/usr/bin/env python

from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from sys import argv
import os

#data_path = "/storage/hpc_irheta/bpic2013/"
data_path = "/storage/hpc_irheta/traffic_fines/"
folds_dir = os.path.join(data_path, "folds")
fold_nr = int(argv[1])

#checkpoint_filepath = "bpic2013/bpic2013_fold%s_weights.{epoch:02d}-{val_loss:.2f}.hdf5"%fold_nr
checkpoint_filepath = "traffic_fines/traffic_fines_fold%s_weights.{epoch:02d}-{val_loss:.2f}.hdf5"%fold_nr


lstmsize = 48 #24
dropout = 0.5
optim = 'rmsprop'
loss = 'categorical_crossentropy'
nb_epoch = 10
batch_size = 1 #100

cat_cols = ["Activity"]
case_id_col = "Case ID"
timestamp_col = "Complete Timestamp"

# Read the relevant folds
fold_files = os.listdir(folds_dir)
data = pd.DataFrame()
for file_idx in range(len(fold_files)):
    if file_idx != fold_nr:
        tmp = pd.read_csv(os.path.join(folds_dir, fold_files[file_idx]), sep=";")
        data = pd.concat([data, tmp], axis=0)


data[timestamp_col] = pd.to_datetime(data[timestamp_col])
cat_data = pd.get_dummies(data[cat_cols])
dt_final = pd.concat([data[[case_id_col, timestamp_col]], cat_data], axis=1).fillna(0)
dt_final["START"] = 0
dt_final["END"] = 0

### TRAIN MODEL ###
grouped = dt_final.groupby(case_id_col)
max_events = grouped.size().max()
data_dim = dt_final.shape[1] - 2
time_dim = max_events + 1 # because we are adding artificial start and endpoints

# dummy endpoints
start = np.zeros(data_dim, dtype=int)
start[-2] = 1
end = np.zeros(data_dim, dtype=int)
end[-1] = 1

print('Generate data...')
X = np.zeros((len(dt_final)+len(grouped), time_dim, data_dim))
y = np.zeros((len(dt_final)+len(grouped), data_dim))
case_idx = 0
for name, group in grouped:
    group = group.sort_values(timestamp_col).as_matrix()[:,2:]
    group = np.vstack([start, group, end])
    for i in range(1, len(group)):
        X[case_idx] = pad_sequences(group[np.newaxis,:i,:], maxlen=time_dim)
        y[case_idx] = group[i,:]
        case_idx += 1
        
print('Build model...')
model = Sequential()
model.add(LSTM(lstmsize, input_shape=(time_dim, data_dim)))
model.add(Dropout(dropout))
model.add(Dense(data_dim, activation='softmax'))
        
print('Compiling model...')
model.compile(loss=loss, optimizer=optim)

print("Training...")
checkpointer = ModelCheckpoint(filepath=checkpoint_filepath, verbose=1, save_best_only=True, save_weights_only=True)
model.fit(X, y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=2, validation_split=0.2, callbacks=[checkpointer])

    
