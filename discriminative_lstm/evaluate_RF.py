import pandas as pd
import numpy as np
import sys
import os
import time
import glob
from sklearn.ensemble import RandomForestClassifier
from sys import argv
import csv
from dataset_manager import DatasetManager
from sklearn.metrics import roc_auc_score


datasets = ["bpic2017"]

train_ratio = 0.8
max_len = 20
cls_method = "rf"

sample_size = int(argv[1])
#val_sample_size = int(argv[2])

output_dir = "results"
n_estimators = 1000
max_features = 0.5
params = "nestimators%s_maxfeatures%s"%(n_estimators, max_features)

    
##### MAIN PART ######    

for dataset_name in datasets:
    
    results_file = os.path.join(output_dir, "evaluation_results/results_%s_%s_%s.csv"%(cls_method, dataset_name, params))
        
    print("Loading data...")
    start = time.time()
    dataset_manager = DatasetManager(dataset_name)
    data = dataset_manager.read_dataset()
    train, test = dataset_manager.split_data(data, train_ratio, split="temporal")
    train = dataset_manager.get_train_sample(train, sample_size)
    #train, val = dataset_manager.get_train_val_data(train, sample_size, val_sample_size)
    print("Done: %s"%(time.time() - start))
    
    print('Encoding data...')
    start = time.time()
    dt_train = dataset_manager.encode_data(train)
    #dt_val = dataset_manager.encode_data(val)
    dt_test = dataset_manager.encode_data(test)
    #X, y = dataset_manager.generate_3d_data(dt_train, max_len)
    #X_val, y_val = dataset_manager.generate_3d_data(dt_val, max_len)
    #X_test, y_test = dataset_manager.generate_3d_data(dt_test, max_len)
    #y = y[:,0,0].reshape(y.shape[0])
    #y_test = y_test[:,0,0].reshape(y_test.shape[0])
    print("Done: %s"%(time.time() - start))
    
    print('Evaluating...')
    start = time.time()
    with open(results_file, 'w') as fout:
        csv_writer = csv.writer(fout, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["dataset", "cls", "params", "nr_events", "metric", "score"])

        correct_all_train = 0    
        correct_all_test = 0 
        for nr_events in range(1, max_len+1):
            X, y = dataset_manager.generate_3d_data_for_prefix_length(dt_train, nr_events, nr_events)
            X_test, y_test = dataset_manager.generate_3d_data_for_prefix_length(dt_test, nr_events, nr_events)
            y = y[:,0,0].reshape(y.shape[0])
            y_test = y_test[:,0,0].reshape(y_test.shape[0])
            
            X_reshaped = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
            X_reshaped_test = X_test.reshape((X_test.shape[0], X.shape[1]*X.shape[2]))

            cls = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features)
            cls.fit(X_reshaped, y)
            y_pred = cls.predict(X_reshaped)
            y_pred_test = cls.predict(X_reshaped_test)
            correct_train = np.sum(y == y_pred)
            correct_test = np.sum(y_test == y_pred_test)
            
            print(nr_events-1, correct_train, correct_test)
            csv_writer.writerow([dataset_name, cls_method, params, nr_events-1, "tp_train", correct_train])
            csv_writer.writerow([dataset_name, cls_method, params, nr_events-1, "tp_test", correct_test])
            csv_writer.writerow([dataset_name, cls_method, params, nr_events-1, "count_train", X.shape[0]])
            csv_writer.writerow([dataset_name, cls_method, params, nr_events-1, "count_test", X_test.shape[0]])
            csv_writer.writerow([dataset_name, cls_method, params, nr_events-1, "acc_train", 1.0 * correct_train / X.shape[0]])
            csv_writer.writerow([dataset_name, cls_method, params, nr_events-1, "acc_test", 1.0 * correct_test / X_test.shape[0]])
            csv_writer.writerow([dataset_name, cls_method, params, nr_events-1, "auc_test", roc_auc_score(y_test, y_pred_test)])

            correct_all_train += correct_train
            correct_all_test += correct_test

        print("accuracy: ", 
              1.0 * correct_all_train / X.shape[0] / max_len,
              1.0 * correct_all_test / X_test.shape[0] / max_len)
        
        csv_writer.writerow([dataset_name, cls_method, params, -1, "acc_all_train",
                             1.0 * correct_all_train / X.shape[0] / max_len])
        csv_writer.writerow([dataset_name, cls_method, params, -1, "acc_all_test",
                             1.0 * correct_all_test / X_test.shape[0] / max_len])
        
    print("Done: %s"%(time.time() - start))
    