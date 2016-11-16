import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


data = pd.read_csv("/storage/hpc_irheta/bpm_data/Road_Traffic_Fine_Management_Process_labeled.csv", sep=";")
data['matricola'].replace(0.0, 1, inplace=True)
data['Resource'] = data['Resource'].astype(str)
data['article'] = data['article'].astype(str)

case_id_col = "Case ID"
label_col = "label"
event_nr_col = "event_nr"
cat_cols = ['Activity', 'Resource', 'Variant', 'article', 'dismissal', 'matricola', 'notificationType', 'vehicleClass']
numeric_cols = ['amount', 'expense', 'points']
pos_label = "positive"
neg_label = "negative"

cat_data = pd.get_dummies(data[cat_cols])
dt_final = pd.concat([data[[case_id_col, event_nr_col, label_col]+numeric_cols], cat_data], axis=1).fillna(0)

# divide into train and test data
train_names, test_names = train_test_split( dt_final[case_id_col].unique(), train_size = 4.0/5, random_state = 22 )
train = dt_final[dt_final[case_id_col].isin(train_names)]
test = dt_final[dt_final[case_id_col].isin(test_names)]

# divide into pos and neg
train_pos = train[train[label_col] == pos_label]
train_neg = train[train[label_col] != pos_label]
test_pos = test[test[label_col] == pos_label]
test_neg = test[test[label_col] != pos_label]

train_pos.to_csv("/storage/hpc_irheta/bpm_data/traffic_fines_train_pos.csv", sep=";", index=False)
train_neg.to_csv("/storage/hpc_irheta/bpm_data/traffic_fines_train_neg.csv", sep=";", index=False)
test_pos.to_csv("/storage/hpc_irheta/bpm_data/traffic_fines_test_pos.csv", sep=";", index=False)
test_neg.to_csv("/storage/hpc_irheta/bpm_data/traffic_fines_test_neg.csv", sep=";", index=False)