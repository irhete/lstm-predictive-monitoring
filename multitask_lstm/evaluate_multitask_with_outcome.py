# Based on https://github.com/verenich/ProcessSequencePrediction

import time
from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Input
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
import csv
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score
import os
from dataset_manager import DatasetManager
import glob

import pandas as pd
import numpy as np

dataset_name = "bpic2017"
cls_method = "lstm_multitask"

data_split_type = "temporal"
normalize_over = "train"

train_ratio = 0.8

lstmsize = 100
dropout = 0.2
n_shared_layers = 1
n_specialized_layers = 1

output_dir = "results"
params = "pd_fixed_trainratio80_outcome"
#params = "lstmsize%s_dropout%s_shared%s_specialized%s"%(lstmsize, dropout, n_shared_layers, n_specialized_layers)
checkpoint_prefix = os.path.join(output_dir, "checkpoints/model_%s_%s"%(dataset_name, params))
model_filename = glob.glob("%s*.hdf5"%checkpoint_prefix)[-1]
#model_filename = "code/output_files/models/model_28-1.51.h5"
results_file = os.path.join(output_dir, "evaluation_results/results_%s_%s_%s.csv"%(cls_method, dataset_name, params))


##### MAIN PART ###### 

print('Preparing data...')
start = time.time()

dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()
train, test = dataset_manager.split_data(data, train_ratio, split=data_split_type) # to reproduce results of Tax et al., use 'ordered' instead of 'temporal'

dt_train = dataset_manager.encode_data_with_label(train)
dt_test = dataset_manager.encode_data_with_label(test)

if normalize_over == "train":
    dataset_manager.calculate_divisors(dt_train)
elif normalize_over == "all":
    dt_all = dataset_manager.extract_timestamp_features(data)
    dt_all = dataset_manager.extract_duration_features(dt_all)
    dataset_manager.calculate_divisors(dt_all)
else:
    print("unknown normalization mode")

dt_test = dataset_manager.normalize_data(dt_test)

print("Done: %s"%(time.time() - start))


max_len = dataset_manager.get_max_case_length(dt_train)
activity_cols = [col for col in dt_train.columns if col.startswith("act")]
n_activities = len(activity_cols)
data_dim = n_activities + 5


# compile a model with same parameters that was trained, and load the weights of the trained model
print('Building model...')
start = time.time()
main_input = Input(shape=(max_len, data_dim), name='main_input')
# train a 2-layer LSTM with one shared layer
l1 = LSTM(lstmsize, input_shape=(max_len, data_dim), consume_less='gpu', init='glorot_uniform', return_sequences=True, dropout_W=dropout)(main_input) # the shared layer
b1 = BatchNormalization(axis=1)(l1)
l2_1 = LSTM(lstmsize, consume_less='gpu', init='glorot_uniform', return_sequences=False, dropout_W=dropout)(b1) # the layer specialized in activity prediction
b2_1 = BatchNormalization()(l2_1)
l2_2 = LSTM(lstmsize, consume_less='gpu', init='glorot_uniform', return_sequences=False, dropout_W=dropout)(b1) # the layer specialized in time prediction
b2_2 = BatchNormalization()(l2_2)
l2_3 = LSTM(lstmsize, consume_less='gpu', init='glorot_uniform', return_sequences=False, dropout_W=dropout)(b1) # the layer specialized in outcome prediction
b2_3 = BatchNormalization()(l2_3)
act_output = Dense(n_activities+1, activation='softmax', init='glorot_uniform', name='act_output')(b2_1)
time_output = Dense(1, init='glorot_uniform', name='time_output')(b2_2)
outcome_output = Dense(2, activation='softmax', init='glorot_uniform', name='outcome_output')(b2_3)


model = Model(input=[main_input], output=[act_output, time_output, outcome_output])
opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
model.compile(loss={'act_output':'categorical_crossentropy', 'time_output':'mae', 'outcome_output':'binary_crossentropy'}, optimizer=opt)
model.load_weights(model_filename)
print("Done: %s"%(time.time() - start))


print('Evaluating...')
start = time.time()
with open(results_file, 'w') as fout:
    csv_writer = csv.writer(fout, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["dataset", "cls", "params", "nr_events", "metric", "score"])

    total = 0
    total_acc = 0
    total_mae = 0
    total_auc_outcome = 0
    for nr_events in range(2, max_len-1):
        
        # encode only prefixes of this length
        X, y_a, y_t, y_o = dataset_manager.generate_3d_data_for_prefix_length_with_label(dt_test, max_len, nr_events)
        #X, y_a, y_t = dataset_manager.generate_3d_data_for_prefix_length(dt_test, max_len, nr_events)
        if X.shape[0] == 0:
            break
        
        y_t = y_t * dataset_manager.divisors["timesincelastevent"]
        
        pred_y = model.predict(X, verbose=0)
        pred_y_a = pred_y[0] 
        pred_y_t = pred_y[1]
        pred_y_t = pred_y_t.flatten()
        pred_y_t[pred_y_t < 0] = 0
        pred_y_t = pred_y_t * dataset_manager.divisors["timesincelastevent"]
        pred_y_o = pred_y[2]
        acc = accuracy_score(np.argmax(y_a, axis=1), np.argmax(pred_y_a, axis=1))
        mae = mean_absolute_error(y_t, pred_y_t)
        try:
            auc_outcome = roc_auc_score(y_o[:,0], pred_y_o[:,0])
        except ValueError:
            auc_outcome = 0.5
        total += X.shape[0]
        total_acc += acc * X.shape[0]
        total_mae += mae * X.shape[0]
        total_auc_outcome += auc_outcome * X.shape[0]
        
        print("prefix = %s, n_cases = %s, acc = %s, mae = %s, auc_outcome = %s"%(nr_events, X.shape[0], acc, mae / 86400, auc_outcome))
        csv_writer.writerow([dataset_name, cls_method, params, nr_events, "n_cases", X.shape[0]])
        csv_writer.writerow([dataset_name, cls_method, params, nr_events, "acc", acc])
        csv_writer.writerow([dataset_name, cls_method, params, nr_events, "mae", mae / 86400])
        csv_writer.writerow([dataset_name, cls_method, params, nr_events, "auc_outcome", auc_outcome])

    csv_writer.writerow([dataset_name, cls_method, params, -1, "total_acc", total_acc / total])
    csv_writer.writerow([dataset_name, cls_method, params, -1, "total_mae", total_mae / total / 86400])
    csv_writer.writerow([dataset_name, cls_method, params, -1, "total_auc_outcome", total_auc_outcome / total])
    
print("Done: %s"%(time.time() - start))
        
print("total acc: ", total_acc / total)
print("total mae: ", total_mae / total / 86400)
print("total auc_outcome: ", total_auc_outcome / total)
