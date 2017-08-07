# Based on https://github.com/verenich/ProcessSequencePrediction

import time
from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Input
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
import csv
from sklearn.metrics import mean_absolute_error, accuracy_score
import os
import sys
from dataset_manager import DatasetManager

import pandas as pd
import numpy as np


dataset_name = "bpic2017"
cls_method = "lstm_singletask"

#train_ratio = 2.0 / 3
train_ratio = 0.8

lstmsize = 100
dropout = 0.2
nb_epoch = 500
n_shared_layers = 1
n_specialized_layers = 1

data_split_type = "temporal"
normalize_over = "train"

output_dir = "results"
params = "pd_fixed_trainratio80_outcome_all_data_singletask"
#params = "lstmsize%s_dropout%s_shared%s_specialized%s"%(lstmsize, dropout, n_shared_layers, n_specialized_layers)
checkpoint_prefix = os.path.join(output_dir, "checkpoints/model_%s_%s"%(dataset_name, params))
checkpoint_filepath = "%s.{epoch:02d}-{val_loss:.2f}.hdf5"%checkpoint_prefix


##### MAIN PART ###### 

print('Preparing data...')
start = time.time()

dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()
train, test = dataset_manager.split_data(data, train_ratio, split=data_split_type) # to reproduce results of Tax et al., use 'ordered' instead of 'temporal'

dt_train = dataset_manager.encode_data_with_label_all_data(train)
print(dataset_manager.encoded_cols)

if normalize_over == "train":
    dataset_manager.calculate_divisors(dt_train)
elif normalize_over == "all":
    dt_all = dataset_manager.extract_timestamp_features(data)
    dt_all = dataset_manager.extract_duration_features(dt_all)
    dataset_manager.calculate_divisors(dt_all)
else:
    print("unknown normalization mode")

dt_train = dataset_manager.normalize_data(dt_train)

max_len = dataset_manager.get_max_case_length(dt_train)
activity_cols = [col for col in dt_train.columns if col.startswith("act")]
n_activities = len(activity_cols)
data_dim = dt_train.shape[1] - 3

X, y_a, y_t, y_o = dataset_manager.generate_3d_data_with_label_all_data(dt_train, max_len)
print(X.shape, y_a.shape, y_t.shape, y_o.shape)
sys.stdout.flush()

print("Done: %s"%(time.time() - start))


# compile a model with same parameters that was trained, and load the weights of the trained model
print('Training model...')
start = time.time()

main_input = Input(shape=(max_len, data_dim), name='main_input')
# train a 2-layer LSTM with one shared layer
l1 = LSTM(lstmsize, input_shape=(max_len, data_dim), consume_less='gpu', init='glorot_uniform', return_sequences=True, dropout_W=dropout)(main_input) # the shared layer
b1 = BatchNormalization(axis=1)(l1)
l2_3 = LSTM(lstmsize, consume_less='gpu', init='glorot_uniform', return_sequences=False, dropout_W=dropout)(b1) # the layer specialized in outcome prediction
b2_3 = BatchNormalization()(l2_3)
outcome_output = Dense(2, activation='softmax', init='glorot_uniform', name='outcome_output')(b2_3)

model = Model(input=[main_input], output=[outcome_output])
opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)

model.compile(loss={'outcome_output':'binary_crossentropy'}, optimizer=opt)
early_stopping = EarlyStopping(monitor='val_loss', patience=42)
model_checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto')
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

history = model.fit(X, {'outcome_output':y_o}, validation_split=0.2, verbose=2, callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=max_len, nb_epoch=nb_epoch)

print("Done: %s"%(time.time() - start))


with open(os.path.join(output_dir, "loss_files/loss_singletask_%s.csv"%params), 'w') as fout:
    fout.write("epoch;train_loss;val_loss\n")
    for epoch in range(len(history.history['loss'])):
        fout.write("%s;%s;%s\n"%(epoch, history.history['loss'][epoch], history.history['val_loss'][epoch]))


