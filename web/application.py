#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: omayr
@description: A light weight service layer for temperate predicitons
"""


import os
from flask import Flask, render_template, send_from_directory, request
import pytz, datetime, time
import json
import numpy as np
import pandas as pd
import base64, urllib2, json, requests
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pymysql.cursors
from sys import platform



# initialization
app = Flask(__name__)

app.config.update(
    DEBUG = True,
)

# controllers

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'ico/favicon.ico')

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['GET'])
def forecast():

    room_name = request.args.get('room_name')


    try:
        # load json and create model
        json_file = open("models/" + room_name + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("models/" + room_name + ".h5")
        print("Loaded model from disk")
    except Exception, e:
        return ("room not found")

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

    query = "SELECT COUNT(*) FROM t_402176"
    cursorObject.execute(query)
    rows = cursorObject.fetchall()
    rows = int(rows[0].popitem()[1])

    # nodes with sufficient data
    nodes = ['402176', '402177', '402178', '402179', '402180', '402181', '402182', '402183', '402184',
             '402185', '402187', '402188', '496908', '496909', '496910', '496912', '496913']

    # nodes with in sufficient data
    non_nodes = ['491179', '491180', '491181', '491182', '491183', '491184', '491185', '496911']

    with open('Temperature_Occupancy_Meters.json') as k:
        rooms = json.load(k,encoding='latin1')

    meter_count = 0
    agg = np.zeros(shape=(288,))

    for room in rooms: # Iterating through nodes
        if room['RoomName'] == room_name:
            break

    for node in room['RoomNodes']:  # Iterating through room nodes
        if str(node['NodeId']) not in non_nodes:  # skipping nodes with insufficient data
            for meter in node['TemperatureMeters']:

                meter_id = str(meter['MeterId'])
                base_url = "https://eadvantage.siemens.com/remote/release"
                tz = pytz.timezone('Europe/Helsinki')
                current_time = datetime.datetime.now(tz)

                start_time = current_time - datetime.timedelta(hours=28)
                st = start_time.strftime("%m/%d/%Y %H:%M:%S")
                ct = current_time.strftime("%m/%d/%Y %H:%M:%S")
                ts = (st,ct)

                url = url_builder(base_url,meter_id,ts)
                data = fetch_data(url, meter_id, ts)
                data['time'] = data['time'].apply(lambda x: x.strftime("%m-%d-%Y %H:%M:%S"))

                temperature = data.iloc[-288:,1]

                if np.count_nonzero(temperature):
                    avg = data['value'].mean()
                    temperature[np.isnan(temperature)] = avg
                    agg += temperature
                    meter_count += 1

    avg_temperature = agg/float(meter_count)

    scaler = MinMaxScaler(feature_range=(0, 1))
    avg_temperature = scaler.fit_transform(avg_temperature)

    X = np.zeros(shape=(288, 288))
    for i in range(len(temperature)):
        if i > 0:
            X[i, 0:i] = temperature[-i:]
            X[i, i:288] = temperature[: 288 - i]
        else:
            X[i] = temperature

    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    # compiling model and making predictions
    loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop')

    forecast = loaded_model.predict(X, batch_size=288)
    forecast = scaler.inverse_transform(forecast)
    f = forecast[0, :]
    base_time = current_time - datetime.timedelta(minutes=current_time.minute % 5, seconds=current_time.second)

    temp_time = base_time
    d = dict()

    for i in range(len(forecast)):
        temp_time += datetime.timedelta(minutes=5)
        d[temp_time.strftime("%m/%d/%Y %H:%M:%S")] = str(f[i])

    d = json.dumps(d)

    return d

@app.route('/roomoccupancy', methods=['GET'])
def room_occupancy():

    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        room_name = request.args.get('room_name')

        with open('Temperature_Occupancy_Meters.json') as k:
            rooms = json.load(k,encoding='latin1')


        start_date += " 00:00:00"
        end_date += " 23:59:00"

        # implement dictionary for NodeId to MeterId
        ts_counter = []

        if platform == 'linux' or platform == 'linux2':
            sock = "/var/run/mysqld/mysqld.sock"
        if platform == 'darwin':
            sock = "/tmp/mysql.sock"

        connection = pymysql.connect(host='localhost',
                                     user='root',
                                     password='',
                                     db='smart_energy',
                                     unix_socket = sock,
                                     cursorclass=pymysql.cursors.DictCursor)
        cursorObject = connection.cursor()

        for room in rooms:
            if room['RoomName'] == room_name:
                break

        for node in room['RoomNodes']:
            node_id = str(node['NodeId'])
            for meter in node['OccupancyMeters']:
                meter_id = str(meter['MeterId'])
                query = "SELECT time " \
                        "FROM t_"+node_id+" " \
                        "WHERE (time > \""+start_date+"\") " \
                        "AND (time < \""+end_date+"\") " \
                        "AND (c_"+meter_id+" = 1)"

                cursorObject.execute(query)
                rows = cursorObject.fetchall()
                # print rows
                if len(rows) > 0:
                    # print len(rows)
                    temp = [str(x['time']) for x in rows]
                    ts_counter = list(set().union(ts_counter, temp))

        res = {'count':str(len(ts_counter))}
        res = json.dumps(res)

        return res

    except Exception, e:
        res = {"error":str(e)}
        res = json.dumps(res)

        return res


@app.route('/roompopularity', methods=['GET'])
def room_popularity():


    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    start_date += " 00:00:00"
    end_date += " 23:59:00"

    # implement dictionary for NodeId to MeterId
    lookup = {'402187':"1710884", "402188":"1710873"}
    count_dict = {}

    # unix_socket="/var/run/mysqld/mysqld.sock"
    sock = "/tmp/mysql.sock"

    if platform == 'linux' or platform == 'linux2':
        sock = "/var/run/mysqld/mysqld.sock"
    if platform == 'darwin':
        sock = "/tmp/mysql.sock "

    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='',
                                 db='smart_energy',
                                 unix_socket = sock,
                                 cursorclass=pymysql.cursors.DictCursor)
    cursorObject = connection.cursor()

    for key, value in lookup.iteritems():
        query = "SELECT count(c_"+value+") as c_"+value+ " " \
                "FROM t_"+key+" " \
                "WHERE (time > \""+start_date+"\") " \
                "AND (time < \""+end_date+"\") " \
                "AND (c_"+value+" = 1)"

        cursorObject.execute(query)
        rows = cursorObject.fetchall()
        count_dict[value] = rows[0]['c_'+value]



    count_dict = json.dumps(count_dict)

    return count_dict


def fetch_data( url, meter_id, ts):

    username = "AkhtarTU"
    password = "BQYQELn7$!9vN]=0"
    b64 = base64.b64encode(username + ":" + password)
    token = "Basic " + b64

    r = requests.get(url,headers={"Authorization":token})
    j = json.loads(str(r._content))

    counter = 0
    data = pd.DataFrame(columns=('time','value'))

    for items in j['items']:
        data.loc[counter] = [items['utcOrgRectime'],items['value']]
        counter += 1


    data['time'] = pd.to_datetime(data['time'])
    #data['time'] = pd.DatetimeIndex(data['time'])
    data = data.drop_duplicates()
    data = missing_values_filler(data)
    #df.to_csv('data/sample/tmp' + thread_name+'_'+meter_id+ '.csv', index=False)
    print "Total Values Fetched", counter
    print len(data)
    return data

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

def url_builder(base_url,meter_id, ts):
    return base_url+"/meter/"+meter_id+"/readings?utcStartTimestamp="\
           +ts[0]+"&utcEndTimestamp="+ts[1]

# launch
@app.route("/")
def hello():
    return "Hello World!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)


