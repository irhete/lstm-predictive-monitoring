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
import xes


#input_filename = "/storage/hpc_irheta/bpic2013/BPIC13_i.csv"
input_filename = "/storage/hpc_irheta/traffic_fines/Road_Traffic_Fine_Management_Process.csv"
#lstm_weights_file = 'bpic2013/bpic2013i_weights.01-0.98.hdf5'
lstm_weights_file = "traffic_fines/traffic_fines_weights.00-1.52.hdf5"
#enhanced_log_template = "bpic2013/enhanced_logs/bpic2013i_enhanced%s.csv"
enhanced_log_template = "traffic_fines/enhanced_logs/traffic_fines_enhanced%s.csv"

added_traces_ratios = [0, 1, 2] # 1 means equal ratio of traces from original log and enhanced

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


def get_event_as_onehot(event_idx):
    event = np.zeros(data_dim)
    event[event_idx] = 1
    return event

def generate_trace():
    event_idx = start_idx
    events = get_event_as_onehot(event_idx)[np.newaxis,:]
    trace = []
    while col_idxs[event_idx] != end_event:# and len(trace) < max_events:
        event_idx = np.random.choice(len(col_idxs), 1, p=model.predict(pad_sequences(events[np.newaxis,:,:], maxlen=time_dim))[0])[0]
        event = get_event_as_onehot(event_idx)
        events = np.vstack([events, get_event_as_onehot(event_idx)])
        trace.append(col_idxs[event_idx])
    return tuple(trace[:-1])


# read original log
data = pd.read_csv(input_filename, sep=";")

# which traces exist in the original log
existing_traces = set()
existing_trace_lengths = defaultdict(int)
grouped = data.groupby(case_id_col)
for name, group in grouped:
    group = group.sort_values(timestamp_col)
    existing_traces.add(tuple(group[activity_col]))
    existing_trace_lengths[len(group)] += 1

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

# generate enhanced logs
for added_trace_ratio in added_traces_ratios:

    n_added_traces = n_existing_traces * added_trace_ratio
    
    with open(enhanced_log_template%added_trace_ratio, "w") as fout:
        fout.write("%s,%s,%s\n"%("Case ID", "Activity", "Complete Timestamp"))
        for row_idx, row in data.iterrows():
            fout.write("%s,%s,%s\n"%(row["Case ID"], row["Activity"], row["Complete Timestamp"]))
        
        # generate new traces
        n_existing = 0
        np.random.seed(22)
        for i in range(n_added_traces):
            trace = generate_trace()
            start_time = datetime.now()
            if trace in existing_traces:
                n_existing += 1
            for event in trace:
                timestamp = datetime.strftime(start_time + timedelta(days=1), '%Y/%m/%d %H:%M:%S.%f')
                fout.write("%s,%s,%s\n"%("new%s"%(i+1), event, timestamp))
    print("Total added: %s, existing: %s, new: %s"%(n_added_traces, n_existing, n_added_traces - n_existing))
            
"""
# generate enhanced logs
for added_trace_ratio in added_traces_ratio:

    log = xes.Log()

    n_added_traces = n_existing_traces * added_trace_ratio

    # add existing traces
    original_traces = data.groupby(case_id_col)
    for name, trace in original_traces:
        t = xes.Trace()
        t.attributes = [
            xes.Attribute(type="name", key="concept:name", value=name),
        ]

        for idx, row in trace.iterrows():
            e = xes.Event()
            e.attributes = [
                xes.Attribute(type="name", key="concept:name", value=row[activity_col]),
                #xes.Attribute(type="2016/11/27 19:14:33.889+02:00", key="time:timestamp", value=row[timestamp_col])
            ]
            t.add_event(e)
            
        log.add_trace(t)

    # generate new traces
    n_existing = 0
    np.random.seed(22)
    for i in range(n_added_traces):
        trace = generate_trace()
        start_time = datetime.now()

        t = xes.Trace()
        t.attributes = [
            xes.Attribute(type="name", key="concept:name", value="new%s"%(i+1)),
        ]
        
        start_time = datetime.now()
        if trace in existing_traces:
            n_existing += 1

        for event in trace:
            e = xes.Event()
            timestamp = datetime.strftime(start_time + timedelta(days=1), '%Y/%m/%d %H:%M:%S.%f')
            e.attributes = [
                xes.Attribute(type="name", key="concept:name", value=event),
                #xes.Attribute(type="2016/11/27 19:14:33.889+02:00", key="time:timestamp", value=timestamp)
            ]
            t.add_event(e)

        log.add_trace(t)

    open(enhanced_log_template%added_trace_ratio, "w").write(str(log))
    
    print("Total added: %s, existing: %s, new: %s"%(n_added_traces, n_existing, n_added_traces - n_existing))
"""