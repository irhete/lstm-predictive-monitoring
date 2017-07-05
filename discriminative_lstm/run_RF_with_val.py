import pandas as pd
import numpy as np
import sys
import os
import time
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import dataset_confs
import glob
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sys import argv


#datasets = ["bpic2011_f%s"%formula for formula in range(1,2)]
#datasets = ["bpic2015_%s_f%s"%(municipality, formula) for municipality in range(1,6) for formula in range(1,3)]
#datasets = ["insurance_activity", "insurance_followup"]
#datasets = ["traffic_fines_f%s"%formula for formula in range(1,4)]
#datasets = ["bpic2011_f%s"%formula for formula in range(1,5)] + ["bpic2015_%s_f%s"%(municipality, formula) for municipality in range(1,6) for formula in range(1,3)] + ["traffic_fines_f%s"%formula for formula in range(1,4)]
#datasets = ["bpic2011_f1"]
#datasets = ["sepsis_cases"]
datasets = ["bpic2017"]


prefix_lengths = list(range(1,21))

train_ratio = 0.8
sample_size = int(argv[1])
val_sample_size = int(argv[2])
max_len = 20

output_dir = "/storage/anna_irene"

    
##### MAIN PART ######    

for dataset_name in datasets:
    
    #outfile = os.path.join(output_dir, "output/results_%s_sample%s_rf.csv"%(dataset_name, sample_size))
        
    #with open(outfile, 'w') as fout:
        
    pos_label = dataset_confs.pos_label[dataset_name]
    neg_label = dataset_confs.neg_label[dataset_name]
        
    # read dataset settings
    case_id_col = dataset_confs.case_id_col[dataset_name]
    activity_col = dataset_confs.activity_col[dataset_name]
    timestamp_col = dataset_confs.timestamp_col[dataset_name]
    label_col = dataset_confs.label_col[dataset_name]
    pos_label = dataset_confs.pos_label[dataset_name]

    dynamic_cat_cols = dataset_confs.dynamic_cat_cols[dataset_name]
    static_cat_cols = dataset_confs.static_cat_cols[dataset_name]
    dynamic_num_cols = dataset_confs.dynamic_num_cols[dataset_name]
    static_num_cols = dataset_confs.static_num_cols[dataset_name]
        
    data_filepath = dataset_confs.filename[dataset_name]
        
    # specify data types
    dtypes = {col:"object" for col in dynamic_cat_cols+static_cat_cols+[case_id_col, label_col, timestamp_col]}
    for col in dynamic_num_cols + static_num_cols:
        dtypes[col] = "float"

    # read data
    data = pd.read_csv(data_filepath, sep=";", dtype=dtypes)
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])

    # split into train and test using temporal split
    grouped = data.groupby(case_id_col)
    start_timestamps = grouped[timestamp_col].min().reset_index()
    start_timestamps.sort_values(timestamp_col, ascending=1, inplace=True)
    train_ids = list(start_timestamps[case_id_col])[:int(train_ratio*len(start_timestamps))]
    train = data[data[case_id_col].isin(train_ids)].sort_values(timestamp_col, ascending=1)
    test = data[~data[case_id_col].isin(train_ids)].sort_values(timestamp_col, ascending=1)

    grouped_train = train.groupby(case_id_col)
    grouped_test = test.groupby(case_id_col)
        
    test_case_lengths = grouped_test.size()

    # encode data for LSTM
    print('Encoding training data...')
    # encode data for LSTM

    scaler = MinMaxScaler()

    dt_train_scaled = pd.DataFrame(scaler.fit_transform(train[dynamic_num_cols+static_num_cols]),
                                   index=train.index, columns=dynamic_num_cols+static_num_cols)
    dt_train_cat = pd.get_dummies(train[dynamic_cat_cols+static_cat_cols])
    dt_train = pd.concat([dt_train_scaled, dt_train_cat], axis=1)
    dt_train[case_id_col] = train[case_id_col]
    dt_train[label_col] = train[label_col].apply(lambda x: 1 if x == pos_label else 0)

    data_dim = dt_train.shape[1] - 2
        
    grouped = dt_train.groupby(case_id_col)
    if val_sample_size + sample_size > len(grouped):
        sample_size = int(len(grouped) * 0.8)
        val_sample_size = len(grouped) - sample_size
    print("Sample size: ", sample_size, " Val sample size: ", val_sample_size)
    start = time.time()
    X = np.empty((sample_size, max_len, data_dim), dtype=np.float32)
    y = np.zeros(sample_size)
    X_val = np.empty((val_sample_size, max_len, data_dim), dtype=np.float32)
    y_val = np.zeros(val_sample_size)
    idx = 0
    for _, group in grouped:
        label = group[label_col].iloc[0]
        group = group.as_matrix()
        if idx < sample_size:
            X[idx,:,:] = pad_sequences(group[np.newaxis,:max_len,:-2], maxlen=max_len)
            y[idx] = label
            idx += 1
        elif idx < sample_size + val_sample_size:
            X_val[idx-sample_size,:,:] = pad_sequences(group[np.newaxis,:max_len,:-2], maxlen=max_len)
            y_val[idx-sample_size] = label
            idx += 1
        else:
            break
    print("time: ", time.time() - start) 
    
    
    correct_all = 0    
    correct_all_val = 0    
    for nr_events in prefix_lengths:
        X_reshaped = X[:,:nr_events,:].reshape((sample_size, nr_events*data_dim))
        X_reshaped_val = X_val[:,:nr_events,:].reshape((sample_size, nr_events*data_dim))

        cls = RandomForestClassifier(n_estimators=1000, max_features=0.5)
        cls.fit(X_reshaped, y)
        y_pred = cls.predict(X_reshaped)
        y_pred_val = cls.predict(X_reshaped_val)
        correct = np.sum(y == y_pred)
        correct_all += correct
        correct_val = np.sum(y_val == y_pred_val)
        correct_all_val += correct_val
        print(nr_events-1, correct, correct_val)
        #print(confusion_matrix(y, y_pred))
    print("accuracy: ", 1.0 * correct_all / sample_size / max_len, 1.0 * correct_all_val / val_sample_size / max_len)