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
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
import dataset_confs
import glob
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
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
max_len = 20
loss = 'binary_crossentropy'
time_dim = max_len
n_classes = 2

lstmsize = int(argv[1])
lstmsize2 = int(argv[2])
dropout = float(argv[3])
learning_rate = float(argv[4])
nb_epoch = int(argv[5])
batch_size = int(argv[6])
sample_size = int(argv[7])
val_sample_size = int(argv[8])


output_dir = "results"

    
##### MAIN PART ######    

for dataset_name in datasets:
    
    print("Started")
    
    params = "lstmsize%s_lstm2size%s_dropout%s_lr%s_epoch%s_batchsize%s_sample%s"%(lstmsize, lstmsize2, int(dropout*100), int(learning_rate*100000), nb_epoch, batch_size, sample_size)
        
    outfile = os.path.join(output_dir, "results/results_lstm_%s_%s.csv"%(dataset_name, params))
        
    checkpoint_prefix = os.path.join(output_dir, "checkpoints/weights_%s_%s"%(dataset_name, params))
    checkpoint_filepath = "%s.{epoch:02d}-{val_loss:.2f}.hdf5"%checkpoint_prefix
    
    loss_file = os.path.join(output_dir, "loss_files/loss_%s_sample%s_%s.txt"%(dataset_name, sample_size, params))
        
    #with open(outfile, 'w') as fout:
    #    fout.write("%s;%s;%s;%s;%s\n"%("dataset", "method", "nr_events", "metric", "score"))
     
    for xx in range(1):
        
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

        dt_train_scaled = pd.DataFrame(scaler.fit_transform(train[dynamic_num_cols+static_num_cols]), index=train.index, columns=dynamic_num_cols+static_num_cols)
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
        y = np.zeros((sample_size, max_len, n_classes), dtype=np.float32)
        X_val = np.empty((val_sample_size, max_len, data_dim), dtype=np.float32)
        y_val = np.zeros((val_sample_size, max_len, n_classes), dtype=np.float32)
        
        idx = 0
        for _, group in grouped:
            label = [group[label_col].iloc[0], 1-group[label_col].iloc[0]]
            group = group.as_matrix()
            if idx < sample_size:
                X[idx,:,:] = pad_sequences(group[np.newaxis,:max_len,:-2], maxlen=max_len)
                y[idx,:,:] = np.tile(label, (max_len, 1))
                idx += 1
            elif idx < sample_size + val_sample_size:
                X_val[idx-sample_size,:,:] = pad_sequences(group[np.newaxis,:max_len,:-2], maxlen=max_len)
                y_val[idx-sample_size,:,:] = np.tile(label, (max_len, 1))
                idx += 1
            else:
                break
                
        print(time.time() - start) 
        
        classes = np.array([neg_label, pos_label])
        
        data_dim = X.shape[2]
        
        method_name = "lstm"
        
        
        print('Build model...')
        model = Sequential()
        model.add(LSTM(lstmsize, input_shape=(time_dim, data_dim), return_sequences=True))
        model.add(BatchNormalization())
        model.add(LSTM(lstmsize2, input_shape=(time_dim, lstmsize), return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(TimeDistributed(Dense(n_classes, activation='softmax'), input_shape=(time_dim, data_dim)))
        
        print('Compiling model...')
        model.compile(loss=loss, optimizer=RMSprop(lr=learning_rate), metrics=["acc"])
        
        print("Training...")
        checkpointer = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True)
        history = model.fit(X, y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=2, validation_data=(X_val, y_val), callbacks=[checkpointer])
        
        with open(loss_file, 'w') as fout2:
            fout2.write("epoch;train_loss;train_acc;val_loss;val_acc;params;dataset\n")
            for epoch in range(nb_epoch):
                fout2.write("%s;%s;%s;%s;%s;%s;%s\n"%(epoch, history.history['loss'][epoch], history.history['acc'][epoch], history.history['val_loss'][epoch], history.history['val_acc'][epoch], params, dataset_name))
              
        # load the best weights
        lstm_weights_file = glob.glob("%s*.hdf5"%checkpoint_prefix)[-1]
        model.load_weights(lstm_weights_file)
        
        y_pred = model.predict(X)
        y_pred_val = model.predict(X_val)
        
        for i in range(max_len):
            #print(np.ravel(y[:,i,0]))
            #print([0 if res < 0.5 else 1 for res in np.ravel(y_pred[:,i,0])])
            print(i, np.sum([0 if res < 0.5 else 1 for res in np.ravel(y_pred[:,i,0])] == np.ravel(y[:,i,0])), 
                  np.sum([0 if res < 0.5 else 1 for res in np.ravel(y_pred_val[:,i,0])] == np.ravel(y_val[:,i,0])))

            
       