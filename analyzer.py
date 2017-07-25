#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: omayr
@description: POC: Exploratody Analysis of CampusKlubi Sensor Data

"""

import pandas as pd
import numpy as np
import time, datetime
from os import listdir
from os import path
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from sklearn.externals import joblib

def read_file():

    p = str(path.dirname(path.abspath("__file__"))) + "/data/sample/exp/"
    files = [f for f in listdir(p) if path.isfile(path.join(p, f))]
    cols = ['utc_org_rec_time']
    cols.extend([x.split('.')[0] for x in files])
    data = pd.DataFrame()

    for file in files:
        temp = pd.read_csv("data/sample/exp/"+file)
        temp.columns = ['time','value']
        temp['time'] = pd.to_datetime(temp['time'])
        temp['time'] = pd.DatetimeIndex(temp['time'])
        temp = temp.drop_duplicates()
        temp = missing_values_filler(file, temp)
        data = pd.concat((data, temp[['value']]), axis=1)
        print "done!"

    data = pd.concat((temp[['time']], data), axis=1)
    data['time'] = temp['time']
    data.columns = ['time', 'ACS', 'AQ', 'ART', 'CS', 'CV', 'EAT', 'HV', 'PI', 'RT', 'STATUS']
    #data = data.set_index(['time'])

    data.to_csv('data/sample/sample.csv', index=False)
    return data

def reconstruction(i,data,s,t):

    cut = i
    i = i+1
    itr = s + datetime.timedelta(minutes=5)
    temp = pd.DataFrame(columns=['time','value'])

    while itr < t:
        temp.set_value(i, 'time', itr)
        temp.set_value(i, 'value', 555555)
        itr += datetime.timedelta(minutes=5)
        i += 1

    data = pd.concat([data.ix[:cut], temp, data.ix[cut+1:]]).reset_index(drop=True)

    return data

def missing_values_filler(file, data):


    delta = datetime.timedelta(minutes=6)
    counter = 0
    size = len(data)-1

    while size > counter:
        for i in range(len(data)):
            counter += 1
            if i < len(data) - 1:
                temp_delta = data.iloc[i+1,0] - data.iloc[i,0]
                if temp_delta > delta:
                    data = reconstruction(i,data,data.iloc[i,0],data.iloc[i+1,0])
                    size = len(data)-1
                    counter = 0
                    break

    print file
    print len(data)

    return data

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg

def split_test_train(ts_univariate, split, look_back):

    # split into train and test sets
    train_size = int(look_back * (split*100))

    train = ts_univariate[0:train_size, :]
    test = ts_univariate[train_size:len(ts_univariate)-1, :]
    train_X, train_y = train[:, 0:look_back], train[:, look_back:]
    test_X, test_y = test[:,0:look_back], test[:,look_back:]

    return train_X, train_y, test_X, test_y


def main():

    np.random.seed(7)
    split = 0.8
    look_back = 288
    data = pd.read_csv("data/sample/sample.csv")


    # loading and normalizing univariate data
    univariate = data['RT'].values


    # Filling missing values.
    avg = data['RT'].mean()
    univariate[np.isnan(univariate)] = avg
    #univariate[np.isnan(univariate)] = 0

    #univariate = univariate.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    univariate = scaler.fit_transform(univariate)
    univariate = univariate.tolist()
    ts_univariate = series_to_supervised(univariate,look_back,look_back)
    ts_univariate = np.array(ts_univariate)

    train_X, train_y, test_X, test_y = split_test_train(ts_univariate, split, look_back)

    # reshape input to be [samples, time steps, features]
    train_X = np.reshape(train_X, (train_X.shape[0],1, train_X.shape[1]))
    test_X = np.reshape(test_X, (test_X.shape[0],1, test_X.shape[1]))

    # Model
    model = Sequential()

    # LSTM
    model.add(LSTM(look_back+1, batch_input_shape=(look_back, train_X.shape[1], train_X.shape[2]), stateful=True, return_sequences=True,))
    model.add(LSTM(look_back, return_sequences=True, stateful=True))
    model.add(LSTM(look_back, stateful=True))

    #model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(look_back)) #train_y.shape[1]
    model.add(Activation('relu'))

    #model.add(RepeatVector(look_back))
    #model.add(LSTM(look_back, stateful=True))
    #model.add(Dense(look_back))

    model.compile(loss='mean_squared_error', optimizer='rmsprop')

    # fit network
    for i in range(100):
        print "Epoch:",i+1
        model.fit(np.concatenate((train_X,test_X),axis=0), np.concatenate((train_y,test_y),axis=0), epochs=1, batch_size=look_back, verbose=1, shuffle=False)
        model.reset_states()

    model_json = model.to_json()
    with open("data/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("data/model.h5")
    print("Saved model to disk")

    # make predictions
    trainPredict = model.predict(train_X,batch_size=look_back)
    testPredict = model.predict(test_X,batch_size=look_back)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    train_y = scaler.inverse_transform(train_y)
    testPredict = scaler.inverse_transform(testPredict)
    test_y = scaler.inverse_transform(test_y)

    for i in range(0,test_y.shape[1]):
        print test_y[len(test_y)-1,i], testPredict[len(testPredict)-1,i]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.plot(test_y[len(test_y)-1,:], 'g', label='Actual Temperature')
    ax.plot(testPredict[len(testPredict)-1,:], 'r', label='Predicted Temperature')
    plt.title('Actual Room Temperature vs. Predicted Room Temperature')
    plt.axis([0, 300, 20, 25])
    ax.legend(loc='best')


    # calculate root mean squared error
    train_score, test_score = [], []

    for i in range(0,len(train_y)):
        train_score.append(math.sqrt(mean_squared_error(train_y[i,:], trainPredict[i,:])))
    for i in range(0, len(test_y)):
        test_score.append(math.sqrt(mean_squared_error(test_y[i,:], testPredict[i,:])))

    print('Train Score: %.4f RMSE' % (np.mean(train_score)))
    print('Test Score: %.4f RMSE' % (np.mean(test_score)))
    plt.show()


if __name__ == "__main__": main()

