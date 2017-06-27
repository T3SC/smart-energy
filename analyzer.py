#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: omayr
@description: POC: Exploratody Analysis of CampusKlubi Sensor Data

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os import path

def read_file():

    p = str(path.dirname(path.abspath(__file__)))+"/data/sample/"
    files = [f for f in listdir(p) if path.isfile(path.join(p,f))]
    cols = ['utc_org_rec_time']
    cols.extend([x.split('.')[0] for x in files])

    data = pd.DataFrame()

    for file in files:
        print file
        temp = pd.read_csv("data/sample/"+file)
        data = pd.concat((data,temp[['value']]),axis=1)

    data = pd.concat((temp[['utc_org_rec_time']],data),axis=1)


    data['utc_org_rec_time'] = temp['utc_org_rec_time']
    data['utc_org_rec_time'] = pd.to_datetime(data['utc_org_rec_time'])
    data['utc_org_rec_time'] = pd.DatetimeIndex(data['utc_org_rec_time'])

    data.columns = cols

    #print data[:10]
    #visualize_corr(data)
    exit(0)

def visualize_corr(data):

    #data = data.drop(['utc_org_rec_time','room_temperature_1710876'],axis=1)
    cd = data.corr()
    print "hello"

    plt.matshow(data.corr())
    plt.show()


def main():

    X,y = read_file()

if __name__ == "__main__": main()

