import pandas as pd
import numpy as np
import sys
import os
import time
import glob
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
from sys import argv
import csv
from dataset_manager import DatasetManager


dataset_name = "bpic2017"
cls_method = "rf_multitask"

train_ratio = 2.0 / 3

output_dir = "results"
n_estimators = 1000
max_features = 0.5
params = "nestimators%s_maxfeatures%s"%(n_estimators, max_features)
results_file = os.path.join(output_dir, "evaluation_results/results_%s_%s_%s.csv"%(cls_method, dataset_name, params))


##### MAIN PART ###### 

print('Preparing data...')
start = time.time()

dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()
train, test = dataset_manager.split_data(data, train_ratio, split="temporal") # to reproduce results of Tax et al., use 'ordered' instead of 'temporal'

dt_train = dataset_manager.encode_data(train)
dt_test = dataset_manager.encode_data(test)

dataset_manager.calculate_divisors(dt_train)
dt_train = dataset_manager.normalize_data(dt_train)
dt_test = dataset_manager.normalize_data(dt_test)
print("Done: %s"%(time.time() - start))

   
max_len = dataset_manager.get_max_case_length(dt_train)

print('Evaluating...')
start = time.time()
with open(results_file, 'w') as fout:
    csv_writer = csv.writer(fout, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["dataset", "cls", "params", "nr_events", "metric", "score"])

    total = 0
    total_acc = 0
    total_mae = 0
    for nr_events in range(2, max_len):
        
        # encode only prefixes of this length
        X, y_a, y_t = dataset_manager.generate_3d_data_for_prefix_length_no_padding(dt_train, nr_events, mode="train")
        X_test, y_a_test, y_t_test = dataset_manager.generate_3d_data_for_prefix_length_no_padding(dt_test, nr_events, mode="test")
        
        if X.shape[0] == 0 or X_test.shape[0] == 0:
            break
            
        y_t_test = y_t_test * dataset_manager.divisors["timesincelastevent"]
        print(nr_events, X.shape, X_test.shape, y_a_test.shape, y_t_test.shape)

        # train models
        cls_a = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features)
        cls_a.fit(X, y_a)
        
        cls_t = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features)
        cls_t.fit(X, y_t)

        pred_y_a = cls_a.predict(X_test)
        pred_y_t = cls_t.predict(X_test)
        pred_y_t[pred_y_t < 0] = 0
        pred_y_t = pred_y_t * dataset_manager.divisors["timesincelastevent"]
        
        acc = accuracy_score(np.argmax(y_a_test, axis=1), np.argmax(pred_y_a, axis=1))
        mae = mean_absolute_error(y_t_test, pred_y_t)
        total += X_test.shape[0]
        total_acc += acc * X_test.shape[0]
        total_mae += mae * X_test.shape[0]
        
        print("prefix = %s, n_cases = %s, acc = %s, mae = %s"%(nr_events, X_test.shape[0], acc, mae / 86400))
        csv_writer.writerow([dataset_name, cls_method, params, nr_events, "n_cases", X_test.shape[0]])
        csv_writer.writerow([dataset_name, cls_method, params, nr_events, "acc", acc])
        csv_writer.writerow([dataset_name, cls_method, params, nr_events, "mae", mae / 86400])

    csv_writer.writerow([dataset_name, cls_method, params, -1, "total_acc", total_acc / total])
    csv_writer.writerow([dataset_name, cls_method, params, -1, "total_mae", total_mae / total / 86400])
    
print("Done: %s"%(time.time() - start))
        
print("total acc: ", total_acc / total)
print("total mae: ", total_mae / total / 86400)

