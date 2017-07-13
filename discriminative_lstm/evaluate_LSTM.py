import pandas as pd
import numpy as np
import sys
import os
import time
import csv
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop
from keras.layers.core import Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
import glob
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sys import argv
from dataset_manager import DatasetManager

datasets = ["bpic2017"]

train_ratio = 0.8
max_len = 20
loss = 'binary_crossentropy'
n_classes = 2
cls_method = "lstm"

lstmsize = int(argv[1])
lstmsize2 = int(argv[2])
dropout = float(argv[3])
learning_rate = float(argv[4])
nb_epoch = int(argv[5])
batch_size = int(argv[6])
sample_size = int(argv[7])
val_sample_size = int(argv[8])

output_dir = "results"
params = "lstmsize%s_lstm2size%s_dropout%s_lr%s_epoch%s_batchsize%s_sample%s"%(lstmsize, lstmsize2, int(dropout*100), int(learning_rate*100000), nb_epoch, batch_size, sample_size)

    
##### MAIN PART ######    

for dataset_name in datasets:
    
    results_file = os.path.join(output_dir, "evaluation_results/results_lstm_%s_%s.csv"%(dataset_name, params))
    checkpoint_prefix = os.path.join(output_dir, "checkpoints/weights_%s_%s"%(dataset_name, params))
    
    print("Loading data...")
    start = time.time()
    dataset_manager = DatasetManager(dataset_name)
    data = dataset_manager.read_dataset()
    train, test = dataset_manager.split_data(data, train_ratio, split="temporal")
    train, val = dataset_manager.get_train_val_data(train, sample_size, val_sample_size)
    print("Done: %s"%(time.time() - start))
    
    print('Encoding data...')
    start = time.time()
    dt_train = dataset_manager.encode_data(train)
    dt_val = dataset_manager.encode_data(val)
    dt_test = dataset_manager.encode_data(test)
    X, y = dataset_manager.generate_3d_data(dt_train, max_len)
    X_val, y_val = dataset_manager.generate_3d_data(dt_val, max_len)
    X_test, y_test = dataset_manager.generate_3d_data(dt_test, max_len)
    print("Done: %s"%(time.time() - start))
        
    print('Building and compiling the model...')
    start = time.time()
    data_dim = X.shape[2]
    model = Sequential()
    model.add(LSTM(lstmsize, input_shape=(max_len, data_dim), return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(lstmsize2, input_shape=(max_len, lstmsize), return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(TimeDistributed(Dense(n_classes, activation='softmax'), input_shape=(max_len, data_dim)))
    model.compile(loss=loss, optimizer=RMSprop(lr=learning_rate), metrics=["acc"])
    print("Done: %s"%(time.time() - start))
        
    print('Loading model weights...')
    start = time.time()
    lstm_weights_file = glob.glob("%s*.hdf5"%checkpoint_prefix)[-1]
    model.load_weights(lstm_weights_file)
    print("Done: %s"%(time.time() - start))
        
    print('Predicting...')
    start = time.time()
    y_pred = model.predict(X)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    print("Done: %s"%(time.time() - start))
        
    print('Evaluating...')
    start = time.time()
    with open(results_file, 'w') as fout:
        csv_writer = csv.writer(fout, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["dataset", "cls", "params", "nr_events", "metric", "score"])

        correct_all_train = 0    
        correct_all_val = 0 
        correct_all_test = 0 
        for i in range(max_len):
            correct_train = np.sum([0 if res < 0.5 else 1 for res in np.ravel(y_pred[:,i,0])] == np.ravel(y[:,i,0]))
            correct_val = np.sum([0 if res < 0.5 else 1 for res in np.ravel(y_pred_val[:,i,0])] == np.ravel(y_val[:,i,0]))
            correct_test = np.sum([0 if res < 0.5 else 1 for res in np.ravel(y_pred_test[:,i,0])] == np.ravel(y_test[:,i,0]))
            print(i, correct_train, correct_val, correct_test)
            csv_writer.writerow([dataset_name, cls_method, params, i, "tp_train", correct_train])
            csv_writer.writerow([dataset_name, cls_method, params, i, "tp_val", correct_val])
            csv_writer.writerow([dataset_name, cls_method, params, i, "tp_test", correct_test])
            csv_writer.writerow([dataset_name, cls_method, params, i, "count_train", X.shape[0]])
            csv_writer.writerow([dataset_name, cls_method, params, i, "count_val", X_val.shape[0]])
            csv_writer.writerow([dataset_name, cls_method, params, i, "count_test", X_test.shape[0]])
            csv_writer.writerow([dataset_name, cls_method, params, i, "acc_train", 1.0 * correct_train / X.shape[0]])
            csv_writer.writerow([dataset_name, cls_method, params, i, "acc_val", 1.0 * correct_val / X_val.shape[0]])
            csv_writer.writerow([dataset_name, cls_method, params, i, "acc_test", 1.0 * correct_test / X_test.shape[0]])

            correct_all_train += correct_train
            correct_all_val += correct_val
            correct_all_test += correct_test

        print("accuracy: ", 
              1.0 * correct_all_train / X.shape[0] / max_len,
              1.0 * correct_all_val / X_val.shape[0] / max_len,
              1.0 * correct_all_test / X_test.shape[0] / max_len)
        
        csv_writer.writerow([dataset_name, cls_method, params, -1, "acc_all_train",
                             1.0 * correct_all_train / X.shape[0] / max_len])
        csv_writer.writerow([dataset_name, cls_method, params, -1, "acc_all_val",
                             1.0 * correct_all_val / X_val.shape[0] / max_len])
        csv_writer.writerow([dataset_name, cls_method, params, -1, "acc_all_test",
                             1.0 * correct_all_test / X_test.shape[0] / max_len])
        
    print("Done: %s"%(time.time() - start))