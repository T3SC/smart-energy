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

def main():

    read_file()
    # data = pd.read_csv("data/sample/sample.csv")
    #
    #
    # univariate = data['RT']
    # print len(univariate)
    #
    # ts_univariate = series_to_supervised(univariate,288,288)
    # print len(list(ts_univariate))

if __name__ == "__main__": main()

