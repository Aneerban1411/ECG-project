#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Model
from keras.layers import Dense, Input
from keras.datasets import mnist
import numpy as np
import mne


# In[2]:


batch_size = 128
nb_epoch = 100

# Parameters for mne dataset
channels_num = 61
encoding_dim = 32


# In[3]:


raw = mne.io.read_raw_brainvision("data/resting_state/zavrin_open_eyes_ecg_15021500.vhdr", preload=True)
data = raw.get_data().T
data.shape


# In[4]:


# Build autoencoder model
input_ = Input(shape=(channels_num,))
encoded = Dense(encoding_dim, activation='relu')(input_)

input_encoded = Input(shape=(encoding_dim,))
decoded = Dense(channels_num, activation='sigmoid')(input_encoded)

encoder = Model(input_, encoded, name="encoder")
decoder = Model(input_encoded, decoded, name="decoder")

autoencoder = Model(input_, decoder(encoder(input_)), name="autoencoder")

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()


# In[5]:


from sklearn.model_selection import train_test_split

X_train, X_test, _, _ = train_test_split(data, data, test_size=0.3)
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

print(X_train.shape)
print(X_test.shape)


# In[6]:


X_train_noise = X_train + 0.3 * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noise = X_test + 0.3 * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
X_train_noise = np.clip(X_train_noise, 0., 1.)
X_test_noise = np.clip(X_test_noise, 0., 1.)
print(X_train_noise.shape)
print(X_test_noise.shape)


# In[7]:


# Train
autoencoder.fit(X_train_noise, X_train, verbose=1,
                validation_data=(X_test_noise, X_test))


# ### Сжатие в формат pickle

# In[8]:


import types
import tempfile
import keras.models
import pickle
import mne as mn
from sklearn.metrics import mean_squared_error


# In[9]:


def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


# In[10]:


make_keras_picklable()
pickle.dump(encoder, open("encoder.p", "wb"))
pickle.dump(decoder, open("decoder.p", "wb"))


# In[ ]:





# In[ ]:


load_encoder = pickle.load(open("encoder.p", "rb"))
load_decoder = pickle.load(open("decoder.p", "rb"))
x_reduce = load_encoder.predict(x_test)
alpha = x_reduce.shape[1] / 64
x_pred = load_decoder.predict(x_reduce)
mse = mean_squared_error(x_train, x_test)
score = (1 + mse) * alpha
if (alpha > 0.9): 
    score = 100000 #infty
print(score)

