from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from sys import argv
from sklearn.preprocessing import MinMaxScaler


#input_filename = "/storage/hpc_irheta/bpm_data/traffic_fines_train_pos.csv"
input_filename = argv[1]
output_filename = argv[2]

max_len = 10
data_dim = 490
lstmsize = 256
dropout = 0.5
optim = 'rmsprop'
loss = 'mean_squared_error'
nb_epoch = 50
batch_size = 1

case_id_col = "Case ID"
label_col = "label"
event_nr_col = "event_nr"



data = pd.read_csv(input_filename, sep=";")
scaler = MinMaxScaler()
data = pd.concat([data.iloc[:,:3], pd.DataFrame(scaler.fit_transform(data.iloc[:,3:]), index=data.index, columns=data.columns[3:])], axis=1)

### TRAIN MODEL ###
grouped = data.groupby(case_id_col)

X = np.zeros((0,max_len-1,data_dim))
y = np.zeros((0,data_dim))
for _, group in grouped:
    tmp2 = group.sort_values(event_nr_col).as_matrix()[:,3:]
    for i in range(1,len(tmp2)):
        X = np.concatenate([X, pad_sequences(tmp2[np.newaxis,:i,:], maxlen=max_len-1)], axis=0)
        y = np.concatenate([y, tmp2[np.newaxis,i,:]], axis=0)

print('Build model...')
model = Sequential()
model.add(LSTM(lstmsize, return_sequences=True, input_shape=(max_len-1, data_dim)))
model.add(Dropout(dropout))
model.add(TimeDistributedDense(data_dim))

print('Compiling model...')
model.compile(loss=loss, optimizer=optim)

print("Training...")
model.fit(X, y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=2)

### SAVE MODEL ###
model.save(output_filename)
    
### PREDICT ###
#predicted = model.predict(X_train)