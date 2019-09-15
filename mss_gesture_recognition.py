# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 05:44:36 2018

@author: vishw
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:45:16 2018

@author: vishw
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 14:29:03 2018

@author: vishw
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 11:31:22 2018

@author: vishw

"""
import os
import glob
import theano
os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
#from keras.optimizers import SGD,RMSprop,adam
from keras import optimizers
from keras.utils import np_utils
import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


#COnvert data to numpy format & use in models
def convert(data_passed):
        data_str = data_passed.to_string()
        data_samp = []
        idx_list = []
        for t in data_str.split():        
            try:
                data_samp.append(float(t))
            except ValueError:
                pass
        for i in range(0, len(data_samp), 2):
            idx_list.append(data_samp[i])    
        fin_data_list = []
        for i in idx_list:
            #print( i)
            k = float(i)
            fin_data_list.append(k)
        fin_data_list = np.array(fin_data_list)
        len1=len(fin_data_list)
        rows=len1/197
        rows=int(rows)
        cols=197
        fin_data_list = np.reshape(fin_data_list,(rows,cols))
            fin_data_list = np.resize(fin_data_list,(45,197))
        return fin_data_list     


all_files = os.listdir("PALM")
data_dict = {}
data_list = []
for i in all_files:
    df1 = pd.read_csv(i,delimiter=':')
    data_dict[i] = convert(df1)    
    data_list.append(data_dict[i])
k1=np.array(data_list)
k1 = np.reshape(k1,(k1.shape[0]*k1.shape[1],197))
        

all_files = os.listdir("FIST")
data_dict = {}
data_list = []
for i in all_files:
    df1 = pd.read_csv(i,delimiter=':')
    data_dict[i] = convert(df1)    
    data_list.append(data_dict[i])
k2=np.array(data_list)
k2 = np.reshape(k2,(k2.shape[0]*k2.shape[1],197))


all_files = os.listdir("PEACE")
data_dict = {}
data_list = []
for i in all_files:
    df1 = pd.read_csv(i,delimiter=':')
    data_dict[i] = convert(df1)    
    data_list.append(data_dict[i])
k3=np.array(data_list)
k3 = np.reshape(k3,(k3.shape[0]*k3.shape[1],197))

fin_data = np.vstack((k1,k2,k3))
                               
fin_data = fin_data/fin_data.max()
len1 = len(k1)
len2 = len(k2)
len3 = len(k3)

num_samples=len(fin_data)
label=np.ones((num_samples,),dtype = int)
label[0:len1] = 0
label[len1:len1+len2] = 1
label[len1+len2:] = 2

data,Label = shuffle(fin_data,label,random_state=2)

train_data = [data,Label]

(X1, y) = (train_data[0],train_data[1])
X = np.expand_dims(X1, axis=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


batch_size = 16

nb_classes = 3

nb_epoch = 35

nb_filters = 32

nb_pool = 3

nb_conv = 10

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

adam = optimizers.Adam(lr=0.0008, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = optimizers.SGD(lr=0.005)
model = Sequential()
model.add(Convolution1D(nb_filters, nb_conv, activation='relu',kernel_initializer='he_normal', input_shape=(197,1)))
model.add(Convolution1D(nb_filters*2, nb_conv, activation='relu',kernel_initializer='he_normal'))
model.add(Dropout(0.4))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, 
                     verbose=1,validation_data=(X_test,Y_test ))

data4 = pd.read_csv('v_palm3.txt',delimiter=':')
data_fin_4 = convert(data4)
labeln=np.ones((len(data_fin_4),),dtype = int)
labeln[0:len(labeln)]=2
data_fin_4 = np.expand_dims(data_fin_4, axis=2)
print(model.predict_classes(data_fin_4))



data4 = pd.read_csv('ani_fist1.txt',delimiter=':')
data_fin_4 = convert(data4)
labeln=np.ones((len(data_fin_4),),dtype = int)
labeln[0:len(labeln)]=2
data_fin_4 = np.expand_dims(data_fin_4, axis=2)
print(model.predict_classes(data_fin_4))


data4 = pd.read_csv('v_peace2.txt',delimiter=':')
data_fin_4 = convert(data4)
labeln=np.ones((len(data_fin_4),),dtype = int)
labeln[0:len(labeln)]=2
data_fin_4 = np.expand_dims(data_fin_4, axis=2)
print(model.predict_classes(data_fin_4))


#
#print(model.predict_classes(X_test[21:25]))
#print(Y_test[21:25])











    