#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: omayr
@description: Training models for temperature predictions of KampusKlaubi
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
from sys import platform
import pymysql.cursors
import json


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

def modeling(univariate, look_back, room_name):

    split = 0.9
    scaler = MinMaxScaler(feature_range=(0, 1))
    univariate = scaler.fit_transform(univariate)
    univariate = univariate.tolist()
    ts_univariate = series_to_supervised(univariate, look_back, look_back)
    ts_univariate = np.array(ts_univariate)

    train_X, train_y, test_X, test_y = split_test_train(ts_univariate, split, look_back)

    # reshape input to be [samples, time steps, features]
    train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
    test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

    # Model
    model = Sequential()

    # LSTM
    model.add(LSTM(look_back + 1, batch_input_shape=(look_back, train_X.shape[1], train_X.shape[2]), stateful=True,
                   return_sequences=True, ))
    model.add(LSTM(look_back, return_sequences=True, stateful=True))
    model.add(LSTM(look_back, stateful=True))

    # model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(look_back))  # train_y.shape[1]
    model.add(Activation('relu'))

    # model.add(RepeatVector(look_back))
    # model.add(LSTM(look_back, stateful=True))
    # model.add(Dense(look_back))

    model.compile(loss='mean_squared_error', optimizer='rmsprop')

    # fit network
    for i in range(100):
        print "Epoch:", i + 1
        model.fit(np.concatenate((train_X, test_X), axis=0), np.concatenate((train_y, test_y), axis=0), epochs=1,
                  batch_size=look_back, verbose=1, shuffle=False)
        model.reset_states()

    model_json = model.to_json()
    with open("data/models/" + room_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("data/models/" + room_name + ".h5")
    print("Saved model to disk")

    # make predictions
    trainPredict = model.predict(train_X, batch_size=look_back)
    testPredict = model.predict(test_X, batch_size=look_back)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    train_y = scaler.inverse_transform(train_y)
    testPredict = scaler.inverse_transform(testPredict)
    test_y = scaler.inverse_transform(test_y)

    # calculate root mean squared error
    train_score, test_score = [], []

    for i in range(0, len(train_y)):
        train_score.append(math.sqrt(mean_squared_error(train_y[i, :], trainPredict[i, :])))
    for i in range(0, len(test_y)):
        test_score.append(math.sqrt(mean_squared_error(test_y[i, :], testPredict[i, :])))

    print room_name
    print('Train Score: %.4f RMSE' % (np.mean(train_score)))
    print('Test Score: %.4f RMSE' % (np.mean(test_score)))


def main():

    np.random.seed(7)
    look_back = 288

    if platform == 'linux' or platform == 'linux2':
        sock = "/var/run/mysqld/mysqld.sock"
    if platform == 'darwin':
        sock = "/tmp/mysql.sock"

    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='',
                                 db='smart_energy',
                                 unix_socket=sock,
                                 cursorclass=pymysql.cursors.DictCursor)
    cursorObject = connection.cursor()

    # nodes with sufficient data
    nodes = ['402176', '402177', '402178', '402179', '402180', '402181', '402182', '402183', '402184',
             '402185', '402187', '402188', '496908', '496909', '496910', '496912', '496913']

    # nodes with in sufficient data
    non_nodes = ['491179', '491180', '491181', '491182', '491183', '491184', '491185', '496911']

    query = "SELECT COUNT(*) FROM t_" + nodes[0]
    cursorObject.execute(query)
    rows = cursorObject.fetchall()
    rows = int(rows[0].popitem()[1])

    with open('data/Temperature_Occupancy_Meters.json') as k:
        rooms = json.load(k,encoding='latin1')

    meter_count = 0
    agg = np.zeros(shape=(rows,))

    for room in rooms: # Iterating through nodes
        for node in room['RoomNodes']:  #Iterating through room nodes
            if str(node['NodeId']) not in non_nodes: # skipping nodes with insufficient data
                for meter in node['TemperatureMeters']:
                    query = "SELECT c_"+str(meter['MeterId'])+" FROM t_"+str(node['NodeId'])+";"
                    df = pd.read_sql(query, con = connection)
                    raw_values = df["c_"+str(meter['MeterId'])]

                    if np.count_nonzero(raw_values):
                        avg = df["c_" + str(meter['MeterId'])].mean()
                        raw_values[np.isnan(raw_values)] = avg
                        agg += raw_values
                        meter_count += 1

        if meter_count > 0:
            univariate = agg / float(meter_count)
            try:
                modeling(univariate,look_back,room['RoomName'])
            except Exception, e:
                print str(e)
                print room['RoomName']

        meter_count = 0
        agg = np.zeros(shape=(40320,))

if __name__ == "__main__": main()