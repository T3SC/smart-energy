#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: omayr
@description: A script to sequentially retrieve sensor data from Siemens Navigator
"""

import time
import numpy as np
import datetime
from threading import Thread
import base64, urllib2, json, requests
import pandas as pd
import json
import os
import pymysql.cursors
from sqlalchemy import create_engine
from sys import platform



def url_builder(base_url,meter_id, ts):
    return base_url+"/meter/"+meter_id+"/readings?utcStartTimestamp="\
           +ts[0]+"&utcEndTimestamp="+ts[1]

def fetch_data(base_url, url, meter_id, ts, path):

    """
    https://eadvantage.siemens.com/remote/release/meter/
    1710876/readings?utcStartTimestamp=04/20/2017 0:00:00
    &utcEndTimestamp=05/20/2017 11:59:59
    """

    print meter_id
    username = "AkhtarTU"
    password = "BQYQELn7$!9vN]=0"
    b64 = base64.b64encode(username + ":" + password)
    token = "Basic " + b64

    r = requests.get(url,headers={"Authorization":token})
    j = json.loads(str(r._content))

    page = 0
    counter = 0
    df = pd.DataFrame(columns=('time','value'))
    new_request = False

    if len(j['items']) > 0:

        while True:

            print "***",str(meter_id)+"-"+str(page)

            for items in j['items']:
                df.loc[counter] = [items['utcOrgRectime'],items['value']]
                counter += 1

            if j['nextPage'] == None:
                ts_end = ts[1].replace("%20", " ")
                #print str(df.iloc[len(df) - 1]['time'])
                time_last = datetime.datetime.strptime(str(df.iloc[len(df) - 1]['time']), "%m/%d/%Y %I:%M:%S %p")
                time_end = datetime.datetime.strptime(ts_end, "%m/%d/%Y %H:%M:%S") - datetime.timedelta(minutes=6)
                if time_end > time_last:
                    new_request = True
                    time_start = time_last.strftime("%m/%d/%Y %H:%M:%S")
                    time_start = time_start.replace(" ","%20")

                    url = url_builder(base_url,meter_id,(time_start,ts[1]))
                    r = requests.get(url, headers={"Authorization": token})
                    j = json.loads(str(r._content))
                else:
                    break

            if new_request == False:
                url = base_url + str(j['nextPage'])
                r = requests.get(url, headers={"Authorization": token})
                j = json.loads(str(r._content))

            page += 1
            new_request = False

        df['time'] = pd.to_datetime(df['time'])
        df['time'] = pd.DatetimeIndex(df['time'])
        df = missing_values_filler(df)
        df = df.drop_duplicates()
        df.to_csv(path+"/"+meter_id+ '.csv', index=False)
        print "Total Values Fetched", counter

def missing_values_filler(data):

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

    return data

def reconstruction(i,data,s,t):

    cut = i
    i = i+1
    itr = s + datetime.timedelta(minutes=5)
    temp = pd.DataFrame(columns=['time','value'])

    while itr < t:
        temp.set_value(i, 'time', itr)
        temp.set_value(i, 'value', np.nan)
        itr += datetime.timedelta(minutes=5)
        i += 1

    data = pd.concat([data.ix[:cut], temp, data.ix[cut+1:]]).reset_index(drop=True)

    return data

def create_tables():

    try:
        with open('data/hierarchy.json') as j:
            hierarchy = json.load(j)

        if platform == 'linux' or platform == 'linux2':
            sock = "/var/run/mysqld/mysqld.sock"
        if platform == 'darwin':
            sock = "/tmp/mysql.sock "

        engine_string =  "mysql+pymysql://root:@localhost/smart_energy?host=localhost?port=3306?unix_socket="+sock
        engine = create_engine(engine_string)


        p = str(os.path.dirname(os.path.abspath("__file__"))) + "/data/nodes/"
        n = os.walk(p)
        for node in n:
            nodes = node[1]
            break

        for node in nodes:

            full_path = p+str(node)
            files = [f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))]
            cols = ['time']
            cols.extend(['c_'+x.split('.')[0] for x in files])
            data = pd.DataFrame()

            for file in files:

                temp = pd.read_csv(p+str(node)+"/" + file)
                temp.columns = ['time', 'value']
                temp['time'] = pd.to_datetime(temp['time'])
                temp['time'] = pd.DatetimeIndex(temp['time'])
                data = pd.concat((data, temp[['value']]), axis=1)


            data = pd.concat((temp[['time']], data), axis=1)
            data['time'] = temp['time']

            data.columns = cols
            # data = data.set_index(['time'])

            data.to_sql(con=engine, name='t_'+str(node), if_exists='replace')

    except Exception, e:
        print str(e)

def main():

    start_time = time.time()

    with open('data/hierarchy.json') as j:
        hierarchy = json.load(j)

    base_url = "https://eadvantage.siemens.com/remote/release"

    ts = ("03/06/2017%2000:00:00", "07/25/2017%2023:59:59")
    rooms = hierarchy['BuildingFloors'][0]['FloorRooms']
    base_path = os.path.dirname(os.path.realpath(__file__))

    for room in rooms:
        for node in room['RoomNodes']:
            p = base_path+"/data/nodes/"+str(node['NodeId'])
            if not os.path.exists(p):
                os.makedirs(p)
                for meter in node['NodeMeters']:
                    url = url_builder(base_url, str(meter['MeterId']), ts)
                    fetch_data(base_url, url, str(meter['MeterId']), ts, p)

    create_tables()

    print "--- %s Minutes ---" % ((time.time() - start_time) / 60)

if __name__ == "__main__": main()
