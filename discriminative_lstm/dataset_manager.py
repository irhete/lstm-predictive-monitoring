import sys

import dataset_confs

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import pad_sequences


class DatasetManager:
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        
        self.case_id_col = dataset_confs.case_id_col[self.dataset_name]
        self.activity_col = dataset_confs.activity_col[self.dataset_name]
        self.timestamp_col = dataset_confs.timestamp_col[self.dataset_name]
        self.label_col = dataset_confs.label_col[self.dataset_name]
        self.pos_label = dataset_confs.pos_label[self.dataset_name]

        self.dynamic_cat_cols = dataset_confs.dynamic_cat_cols[self.dataset_name]
        self.static_cat_cols = dataset_confs.static_cat_cols[self.dataset_name]
        self.dynamic_num_cols = dataset_confs.dynamic_num_cols[self.dataset_name]
        self.static_num_cols = dataset_confs.static_num_cols[self.dataset_name]
        
        self.scaler = None
        self.encoded_cols = None
        
    
    def read_dataset(self):
        # read dataset
        dtypes = {col:"object" for col in self.dynamic_cat_cols+self.static_cat_cols+[self.case_id_col, self.label_col, self.timestamp_col]}
        for col in self.dynamic_num_cols + self.static_num_cols:
            dtypes[col] = "float"

        data = pd.read_csv(dataset_confs.filename[self.dataset_name], sep=";", dtype=dtypes)
        data[self.timestamp_col] = pd.to_datetime(data[self.timestamp_col])

        return data


    def split_data(self, data, train_ratio, split="temporal"):  
        # split into train and test using temporal split
        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        if split == "temporal":
            start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True)
        train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True)
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True)

        return (train, test)
    
    def get_train_val_data(self, data, sample_size=None, val_sample_size=None, train_ratio=0.8):
        # adjust sample sizes if the total number of cases is smaller than required
        grouped = data.groupby(self.case_id_col)
        if sample_size is None or val_sample_size is None or val_sample_size + sample_size > len(grouped):
            sample_size = int(len(grouped) * train_ratio)
            val_sample_size = len(grouped) - sample_size
            print("Adjusted sample sizes, new sample_size: ", sample_size, ", new val_sample_size: ", val_sample_size)
        
        case_ids = [name for name, _ in grouped]
        train = data[data[self.case_id_col].isin(case_ids[:sample_size])]
        val = data[data[self.case_id_col].isin(case_ids[sample_size:sample_size+val_sample_size])]
        return (train, val)
    
    def get_train_sample(self, data, sample_size=None):
        # adjust sample sizes if the total number of cases is smaller than required
        grouped = data.groupby(self.case_id_col)
        if sample_size is None or sample_size > len(grouped):
            print("Adjusted sample size, new sample_size: ", len(grouped))
            return(data)
        
        case_ids = [name for name, _ in grouped]
        train = data[data[self.case_id_col].isin(case_ids[:sample_size])]
        return train
    
    def encode_data(self, data):
        
        # scale numeric cols
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            dt_scaled = pd.DataFrame(self.scaler.fit_transform(data[self.dynamic_num_cols+self.static_num_cols]), index=data.index, columns=self.dynamic_num_cols+self.static_num_cols)
        else:
            dt_scaled = pd.DataFrame(self.scaler.transform(data[self.dynamic_num_cols+self.static_num_cols]), index=data.index, columns=self.dynamic_num_cols+self.static_num_cols)
            
        # one-hot encode categorical cols
        dt_cat = pd.get_dummies(data[self.dynamic_cat_cols+self.static_cat_cols])
        
        # merge
        dt_all = pd.concat([dt_scaled, dt_cat], axis=1)
        dt_all[self.case_id_col] = data[self.case_id_col]
        dt_all[self.label_col] = data[self.label_col].apply(lambda x: 1 if x == self.pos_label else 0)
        
        # add missing columns if necessary
        if self.encoded_cols is None:
            self.encoded_cols = dt_all.columns
        else:
            for col in self.encoded_cols:
                if col not in dt_all.columns:
                    dt_all[col] = 0
        
        return dt_all[self.encoded_cols]

    
    def generate_3d_data(self, data, max_len):
        grouped = data.groupby(self.case_id_col)
        
        data_dim = data.shape[1] - 2 # all cols minus the class label and case id
        n_classes = len(data[self.label_col].unique())
        
        X = np.empty((len(grouped), max_len, data_dim), dtype=np.float32)
        y = np.zeros((len(grouped), max_len, n_classes), dtype=np.float32)
        
        idx = 0
        for _, group in grouped:
            label = [group[self.label_col].iloc[0], 1-group[self.label_col].iloc[0]]
            group = group.as_matrix()
            X[idx,:,:] = pad_sequences(group[np.newaxis,:max_len,:-2], maxlen=max_len)
            y[idx,:,:] = np.tile(label, (max_len, 1))
            idx += 1
                
        return (X, y)


    def generate_prefix_data(self, data, min_length, max_length):
        # generate prefix data (each possible prefix becomes a trace)
        data['case_length'] = data.groupby(self.case_id_col)[self.activity_col].transform(len)

        dt_prefixes = data[data['case_length'] >= min_length].groupby(self.case_id_col).head(min_length)
        for nr_events in range(min_length+1, max_length+1):
            tmp = data[data['case_length'] >= nr_events].groupby(self.case_id_col).head(nr_events)
            tmp[self.case_id_col] = tmp[self.case_id_col].apply(lambda x: "%s_%s"%(x, nr_events))
            dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)
        
        dt_prefixes['case_length'] = dt_prefixes.groupby(self.case_id_col)[self.activity_col].transform(len)
        
        return dt_prefixes


    def get_pos_case_length_quantile(self, data, quantile=0.90):
        return int(np.ceil(data[data[self.label_col]==self.pos_label].groupby(self.case_id_col).size().quantile(quantile)))

    def get_indexes(self, data):
        return data.groupby(self.case_id_col).first().index

    def get_relevant_data_by_indexes(self, data, indexes):
        return data[data[self.case_id_col].isin(indexes)]

    def get_label(self, data):
        return data.groupby(self.case_id_col).first()[self.label_col]
    
    def get_label_numeric(self, data):
        y = self.get_label(data) # one row per case
        return [1 if label == self.pos_label else 0 for label in y]
    
    def get_class_ratio(self, data):
        class_freqs = data[self.label_col].value_counts()
        return class_freqs[self.pos_label] / class_freqs.sum()
    
    def get_stratified_split_generator(self, data, n_splits=5, shuffle=True, random_state=22):
        grouped_firsts = data.groupby(self.case_id_col, as_index=False).first()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        for train_index, test_index in skf.split(grouped_firsts, grouped_firsts[self.label_col]):
            current_train_names = grouped_firsts[self.case_id_col][train_index]
            train_chunk = data[data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col, ascending=True)
            test_chunk = data[~data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col, ascending=True)
            yield (train_chunk, test_chunk)