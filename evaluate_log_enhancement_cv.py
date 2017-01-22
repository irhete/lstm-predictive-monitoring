import glob
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM, GRU
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from datetime import datetime, timedelta
from collections import defaultdict
import os


data_path = "/storage/hpc_irheta/bpic2013/"
folds_dir = os.path.join(data_path, "folds")
n_folds = 5
weights_file_template = "bpic2013/bpic2013_fold%s_weights*.hdf5"
results_filename = "bpic2013/results_cv.out"
generated_traces_ratios = [1, 2, 5, 10] # ratios to single fold size


case_id_col = "Case ID"
activity_col = "Activity"
timestamp_col = "Complete Timestamp"
cat_cols = [activity_col]
start_event = "START"
end_event = "END"

# LSTM params
lstmsize = 48
dropout = 0.5
optim = 'rmsprop'
loss = 'categorical_crossentropy'
nb_epoch = 10
activation='softmax'


def get_event_as_onehot(event_idx, data_dim):
    event = np.zeros(data_dim)
    event[event_idx] = 1
    return event

def generate_trace(start_idx, data_dim, end_event, time_dim, col_idxs):
    event_idx = start_idx
    events = get_event_as_onehot(event_idx, data_dim)[np.newaxis,:]
    trace = []
    while col_idxs[event_idx] != end_event:# and len(trace) < max_events:
        event_idx = np.random.choice(len(col_idxs), 1, p=model.predict(pad_sequences(events[np.newaxis,:,:], maxlen=time_dim))[0])[0]
        event = get_event_as_onehot(event_idx, data_dim)
        events = np.vstack([events, get_event_as_onehot(event_idx, data_dim)])
        trace.append(col_idxs[event_idx])
    return tuple(trace[:-1])


with open(results_filename, 'w') as fout:
    fout.write("fold_nr;generated_trace_ratio;total_train;total_val;unique_train;unique_val;unique_diff;unique_union;total_generated;unique_generated;total_generated_in_train;unique_generated_in_train;total_generated_in_val;unique_generated_in_val;total_generated_new;unique_generated_new\n")
                                  
    
    for fold_nr in range(n_folds):
        lstm_weights_file = glob.glob(weights_file_template%(fold_nr))[-1]

        # Read the relevant folds
        fold_files = os.listdir(folds_dir)
        data = pd.DataFrame()
        for file_idx in range(len(fold_files)):
            if file_idx != fold_nr:
                tmp = pd.read_csv(os.path.join(folds_dir, fold_files[file_idx]), sep=";")
                data = pd.concat([data, tmp], axis=0)
            else:
                val_data = pd.read_csv(os.path.join(folds_dir, fold_files[file_idx]), sep=";")

        # which traces exist in the train and val logs
        train_traces = set()
        grouped = data.groupby(case_id_col)
        for name, group in grouped:
            group = group.sort_values(timestamp_col)
            train_traces.add(tuple(group[activity_col]))

        val_traces = set()
        grouped_val = val_data.groupby(case_id_col)
        for name, group in grouped_val:
            group = group.sort_values(timestamp_col)
            val_traces.add(tuple(group[activity_col]))

        # prepare data
        cat_data = pd.get_dummies(data[cat_cols])
        dt_final = pd.concat([data[[case_id_col, timestamp_col]], cat_data], axis=1).fillna(0)
        dt_final[start_event] = 0
        dt_final[end_event] = 0
        grouped = dt_final.groupby(case_id_col)
        n_existing_traces = len(grouped)

        # generate dict of activity idxs
        col_idxs = {idx:col.replace("%s_"%activity_col, "") for idx, col in enumerate(cat_data.columns)}
        col_idxs[len(col_idxs)] = start_event
        col_idxs[len(col_idxs)] = end_event
        start_idx = col_idxs.keys()[col_idxs.values().index(start_event)]


        # load LSTM model
        max_events = grouped.size().max()
        data_dim = dt_final.shape[1] - 2
        time_dim = max_events + 1

        model = Sequential()
        model.add(LSTM(lstmsize, input_shape=(time_dim, data_dim)))
        model.add(Dropout(dropout))
        model.add(Dense(data_dim, activation=activation))
        model.compile(loss=loss, optimizer=optim)

        model.load_weights(lstm_weights_file)

        print("Number of distinct traces in train set: %s, val set: %s, total: %s"%(len(train_traces), len(val_traces), len(train_traces.union(val_traces))))
        print("Number of distinct traces in val set not present in train set: %s"%(len(val_traces.difference(train_traces))))
        print("\n")

        for generated_traces_ratio in generated_traces_ratios:
            n_generated_traces = len(grouped_val) * generated_traces_ratio

            # generate new traces
            traces_existing_in_train = defaultdict(int)
            traces_existing_in_validation = defaultdict(int)
            traces_new = defaultdict(int)
            np.random.seed(22)
            for i in range(n_generated_traces):
                trace = generate_trace(start_idx, data_dim, end_event, time_dim, col_idxs)
                if trace in train_traces:
                    traces_existing_in_train[trace] += 1
                elif trace in val_traces:
                    traces_existing_in_validation[trace] += 1
                else:
                    traces_new[trace] += 1
                    
            fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(fold_nr, generated_traces_ratio, len(grouped), len(grouped_val), len(train_traces), len(val_traces), len(val_traces.difference(train_traces)), len(train_traces.union(val_traces)), n_generated_traces, len(traces_existing_in_train) + len(traces_existing_in_validation) + len(traces_new), sum(traces_existing_in_train.values()), len(traces_existing_in_train), sum(traces_existing_in_validation.values()), len(traces_existing_in_validation), sum(traces_new.values()), len(traces_new)))


