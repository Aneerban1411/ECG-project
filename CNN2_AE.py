import numpy as np
import importlib 
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, UpSampling2D, Reshape

from keras.layers.normalization import BatchNormalization

import pickle
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import h5py    



from keras import backend as K
K.tensorflow_backend._get_available_gpus()



get_ipython().system('pip install keras')




def read_h5_file(file_name, scaler = None, preprocess = False):
    h5_file = h5py.File(train_ecg_dir + file_name, 'r')
    a_group_key = list(h5_file.keys())[0]
    ecg_data = np.array(h5_file[a_group_key]).T
    if preprocess:
        ecg_data = scaler.transform(ecg_data)
    return ecg_data




def train_scaler(scaler, train_ecg_names, log = False):
    i = 0
    for ecg_name in train_ecg_names:
        if log:
            print("{} from {}".format(i, len(train_ecg_names)))
            print("reading:{}".format(ecg_name))
        data = read_h5_file(ecg_name)
        i = i+1
        scaler.fit(data)
        if log:
            print("trained on {}".format(ecg_name))
            
def save_scaler(path,scaler):
    pickle.dump(scaler, open(path, 'wb'))
def load_scaler(path):
    scaler = pickle.load(open(path, 'rb'))
    return scaler



train_ecg_dir = "./data/train/"
trained_scaler_path = None



train_ecg_dir = "./data/train/"
all_train_ecg_names = [x for x in os.listdir(train_ecg_dir) 
                if x[-3:] == ".h5"]
ecg_num = len(all_train_ecg_names)
print("Number of ECG overall:", ecg_num)



if trained_scaler_path:
    scaler = load_scaler(trained_scaler_path)
else:
    scaler = StandardScaler()
    print("Params before training ", scaler.get_params())
    train_scaler(scaler, all_train_ecg_names, log = True)
    print("Params after training ", scaler.get_params())
    save_scaler("./StandardScaler.p", scaler)



window_size = 10
encoding_dim = 50

cnnencoder = Sequential((
    Conv2D(nb_filter=5, kernel_size=(5, 5), activation='relu', padding='valid', input_shape=(window_size, 58, 1)),
    Dropout(0.6),
    MaxPooling2D(),
    
    Flatten(),
    Dense(encoding_dim, activation='relu'),
))
cnnencoder.summary()
print(cnnencoder.output_shape)




cnndecoder = Sequential((
    Dense(405, activation='relu', input_shape=(encoding_dim,)),
    Reshape((-1, 27, 5)),
    UpSampling2D(),
    Conv2DTranspose(nb_filter=5, kernel_size=(5, 5), activation='relu',  padding='valid'),
    Dense(1, activation='relu')
))
cnndecoder.summary()
print(cnndecoder.output_shape)





from keras.models import Model
from keras.layers import Input

input_ = Input(shape=(window_size, 58, 1))

autoencoder = Model(input_, cnndecoder(cnnencoder(input_)), name="autoencoder")
autoencoder.compile(loss='mse', optimizer='adam') # .compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.summary()





b1, b2, b3 = 'shuhova_08022017_rest_ecg_processed.h5', 'zavrib_post_ecg_eyesopen15021500_processed.h5', 'zavrin_15021500_eyesclosed_post_ecg_processed.h5'





all_train_ecg_names = np.array(all_train_ecg_names)
all_train_ecg_names = all_train_ecg_names[(all_train_ecg_names != b1) & (all_train_ecg_names != b2) & (all_train_ecg_names != b3)]





overall_epoch_num = 10
file_epoch_num = 1
history_path = "train_hist"

test_ecg_name = np.random.choice(all_train_ecg_names)
train_ecg_names = np.array(all_train_ecg_names)
print("test_ecg_name is ", test_ecg_name)
test_data = read_h5_file(test_ecg_name, scaler, True)




learn_file_length = 300000





import threading
class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()
    def __iter__(self):
        return self
    def __next__(self):
        with self.lock:
            return next(self.it)

def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g





batch_length = 10
def generate_batch():
    while True:
        cur_files = [[] for _ in range(len(train_ecg_names))]
        batches = [[] for _ in range(len(train_ecg_names))]
        files_count = len(train_ecg_names)
        while files_count > 0:
            file_ind = np.random.choice(np.arange(len(train_ecg_names)))
            if len(cur_files[file_ind]) == 1 and cur_files[file_ind] == -1: continue
            elif len(cur_files[file_ind]) < 1:
                raw = read_h5_file(train_ecg_names[file_ind], scaler, True)
                cur_files[file_ind] = raw
                #print(raw.shape[0] // batch_length)
                batches[file_ind] = np.arange(raw.shape[0] // batch_length)
                
            begin = np.random.choice(np.arange(len(batches[file_ind])))
            
            data = cur_files[file_ind][begin:begin+batch_length, :]
            yield data.reshape(-1, window_size, 58, 1), data.reshape(-1, window_size, 58, 1) # add noise later
            
            batches[file_ind] = np.delete(batches[file_ind], begin)
            if len(batches[file_ind]) == 0:
                cur_files[file_ind] = -1
                files_count -= 1





history = autoencoder.fit_generator(generate_batch(), 
                                    samples_per_epoch=30000, 
                                    verbose=1,
                                    nb_epoch=10,
                                    validation_data=(test_data.reshape(-1, window_size, 58, 1), test_data.reshape(-1, window_size, 58, 1))
                                   )





history_path = "train_hist_51.txt"




cnnencoder.save('CNN_encoder50.p')
autoencoder.save('CNN_autoencoder50.p')
with open(history_path, 'wb') as file:
    pickle.dump(history.history, file)





for epoch in range(overall_epoch_num//file_epoch_num):
    for name in train_ecg_names:
        train_data = read_h5_file(name, scaler, True)
        if len(train_data) > learn_file_length:
            train_data = train_data[:learn_file_length]
        print("epoch: {}, file: {}".format(epoch, name))
        history = autoencoder.fit(train_data.reshape(-1, window_size, 58, 1), train_data.reshape(-1, window_size, 58, 1), 
                                  verbose=1, 
                                  epochs=file_epoch_num,
                                  batch_size = 10,
                                  validation_data=(test_data.reshape(-1, window_size, 58, 1), test_data.reshape(-1, window_size, 58, 1)))
cnnencoder.save('CNN_encoder3.p')
autoencoder.save('CNN_autoencoder3.p')
with open(history_path, 'wb') as file:
    pickle.dump(history.history, file)


test_data.reshape(-1, 60, 58, 1)

test_data.shape
