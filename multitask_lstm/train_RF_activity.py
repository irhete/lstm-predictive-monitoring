import numpy as np
import time
from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


eventlog = "data/helpdesk.csv"

timestamp_col = "CompleteTimestamp"
case_id_col = "CaseID"
activity_col = "ActivityID"

def extract_durations(group):
    global counter
    
    group = group.sort_values(timestamp_col, ascending=False)
    
    tmp = group[timestamp_col] - group[timestamp_col].shift(-1)
    tmp = tmp.fillna(0)
    group["timesincelastevent"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 's'))) # m is for minutes
    
    tmp = group[timestamp_col] - group[timestamp_col].iloc[-1]
    tmp = tmp.fillna(0)
    group["timesincecasestart"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 's'))) # m is for minutes
    
    group["timesincemidnight"] = group[timestamp_col].dt.hour * 3600 + group[timestamp_col].dt.minute * 60 + group[timestamp_col].dt.second
    
    group["weekday"] = group[timestamp_col].dt.weekday
    
    dummy_row = pd.DataFrame(group.iloc[:1,:].values, index=[counter], columns=group.columns)
    dummy_row[activity_col] = "LAST"
    dummy_row["timesincelastevent"] = 0
    dummy_row[timestamp_col] = dummy_row[timestamp_col] + pd.Timedelta(seconds=1)
    counter += 1
    group = pd.concat([dummy_row, group], axis=0)
    
    group = group.sort_values(timestamp_col, ascending=True)
    group["event_nr"] = range(1, len(group) + 1)
    
    return group

data = pd.read_csv(eventlog)
data[timestamp_col] = pd.to_datetime(data[timestamp_col])

# split into train and test using temporal split
train_ratio = 2.0/3

grouped = data.groupby(case_id_col)
start_timestamps = grouped[timestamp_col].min().reset_index()
start_timestamps.sort_values(timestamp_col, ascending=1, inplace=True)
train_ids = list(start_timestamps[case_id_col])[:int(train_ratio*len(start_timestamps))]
train = data[data[case_id_col].isin(train_ids)].sort_values(timestamp_col, ascending=1)
test = data[~data[case_id_col].isin(train_ids)].sort_values(timestamp_col, ascending=1)

counter = len(train)
dt_durations = train.groupby(case_id_col).apply(extract_durations)

#counter = len(data)
#dt_durations_all = data.groupby(case_id_col).apply(extract_durations)

# normalize
dt_durations["timesincelastevent"] = dt_durations["timesincelastevent"] / np.mean(dt_durations[dt_durations[activity_col]!="LAST"]["timesincelastevent"])
dt_durations["timesincecasestart"] = dt_durations["timesincecasestart"] / np.mean(dt_durations[dt_durations[activity_col]!="LAST"]["timesincecasestart"])
dt_durations["timesincemidnight"] = dt_durations["timesincemidnight"] / 86400.0
dt_durations["weekday"] = dt_durations["weekday"] / 7.0

dt_durations = dt_durations.sort_values(timestamp_col, ascending=True)

# activity of last event
dt_transformed = pd.get_dummies(dt_durations[activity_col])
activity_cols = ["act_%s"%col for col in dt_transformed.columns]
dt_transformed.columns = activity_cols
dt_all = pd.merge(dt_durations, dt_transformed, left_index=True, right_index=True)

grouped = dt_all.sort_values(case_id_col, ascending=True).groupby(case_id_col)

n_activities = len(dt_all[activity_col].unique())
max_len = dt_durations.groupby(case_id_col).size().max()
data_dim = n_activities + 4
time_cols = ["timesincelastevent", "timesincecasestart", "timesincemidnight", "weekday"]
activity_cols = [col for col in activity_cols if "LAST" not in col] + ["act_LAST"]

X = np.zeros((dt_all.shape[0] - len(grouped), max_len, data_dim), dtype=np.float32)
y_a = np.zeros((dt_all.shape[0] - len(grouped), n_activities), dtype=np.float32)
y_t = np.zeros((dt_all.shape[0] - len(grouped)), dtype=np.float32)

start = time.time()
idx = 0
for name, group in grouped:
    group = group.sort_values(timestamp_col, ascending=True)[activity_cols + ["event_nr"] + time_cols].as_matrix()
    for i in range(1, len(group)):
        row = group[np.newaxis,:i,:]
        row = np.delete(row, n_activities-1, 2)
        X[idx] = pad_sequences(row, maxlen=max_len, dtype=np.float32)
        y_a[idx] = group[np.newaxis,i,:n_activities]
        y_t[idx] = group[np.newaxis,i,n_activities+1]
        idx += 1
print(time.time() - start)

for nr_events in range(2, 10):
        X_reshaped = X[:,:nr_events,:].reshape((dt_all.shape[0], nr_events*data_dim))

        cls = RandomForestClassifier(n_estimators=1000, max_features=0.5)
        cls.fit(X_reshaped, y)
        y_pred = cls.predict(X_reshaped)
        correct = np.sum(y == y_pred)
        correct_all += correct
        print(nr_events-1, correct)
