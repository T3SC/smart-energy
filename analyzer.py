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

    p = str(path.dirname(path.abspath("__file__"))) + "/data/sample/"
    files = [f for f in listdir(p) if path.isfile(path.join(p, f))]
    cols = ['utc_org_rec_time']
    cols.extend([x.split('.')[0] for x in files])
    data = pd.DataFrame()

    for file in files:
        temp = pd.read_csv("data/sample/"+file)
        temp.columns = ['time','value']
        temp['time'] = pd.to_datetime(temp['time'])
        temp['time'] = pd.DatetimeIndex(temp['time'])
        temp = missing_values_filler(file, temp)
        data = pd.concat((data, temp[['value']]), axis=1)

    data = pd.concat((temp[['time']], data), axis=1)

    data['time'] = temp['time']
    data = data.set_index(['time'])


    data.columns = ['time', 'ACS', 'AQ', 'ART', 'CS', 'CV', 'EAT', 'HV', 'PI', 'RT', 'STATUS']

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
            if i > 0 and i < len(data) - 1:
                temp_delta = data.iloc[i+1,0] - data.iloc[i,0]
                if temp_delta > delta:
                    data = reconstruction(i,data,data.iloc[i,0],data.iloc[i+1,0])
                    size = len(data)-1
                    counter = 0
                    break
    
    return data

def main():

    data = read_file()
    print len(data)

if __name__ == "__main__": main()

