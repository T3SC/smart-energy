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


    meter_id = request.args.get('meter_id')

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

    avg = data['value'].mean()
    temperature = data.iloc[-288:,1]
    temperature[np.isnan(temperature)] = avg
    scaler = MinMaxScaler(feature_range=(0, 1))
    temperature = scaler.fit_transform(temperature)

    X = np.zeros(shape=(288, 288))
    for i in range(len(temperature)):
        if i > 0:
            X[i, 0:i] = temperature[-i:]
            X[i, i:288] = temperature[: 288 - i]
        else:
            X[i] = temperature

    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))


    # load json and create model
    json_file = open('models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/model.h5")
    print("Loaded model from disk")

    # compiling model and making predictions
    loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop')

    prediction = loaded_model.predict(X, batch_size=288)
    prediction = scaler.inverse_transform(prediction)
    p = prediction[0, :]
    base_time = current_time - datetime.timedelta(minutes=current_time.minute % 5, seconds=current_time.second)

    temp_time = base_time
    d = dict()

    for i in range(len(prediction)):
        temp_time += datetime.timedelta(minutes=5)
        d[temp_time.strftime("%m/%d/%Y %H:%M:%S")] = str(p[i])

    d = json.dumps(d)

    return d

@app.route('/roomoccupancy', methods=['GET'])
def room_occupancy():

    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    room_name = request.args.get('room_name')

    with open('Temperature_Occupancy_Meters.json') as j:
        rooms = json.load(j)


    start_date += " 00:00:00"
    end_date += " 23:59:00"

    # implement dictionary for NodeId to MeterId
    count_dict = {}

    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='',
                                 db='smart_energy',
                                 cursorclass=pymysql.cursors.DictCursor)
    cursorObject = connection.cursor()

    for room in rooms:
        if room['RoomName'] == room_name:
            break

    for node in room['RoomNodes']:
        node_id = node['NodeId']
        for meter in node['OccupancyMeters']:
            meter_id = meter['MeterId']
            query = "SELECT count(c_"+meter_id+") as c_"+meter_id+ " " \
                    "FROM t_"+node_id+" " \
                    "WHERE (time > \""+start_date+"\") " \
                    "AND (time < \""+end_date+"\") " \
                    "AND (c_"+meter_id+" = 1)"

            cursorObject.execute(query)
            rows = cursorObject.fetchall()
            count_dict[meter_id] = rows[0]['c_'+meter_id]


    count_dict = json.dumps(count_dict)

    return count_dict


@app.route('/roompopularity', methods=['GET'])
def room_popularity():


    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    start_date += " 00:00:00"
    end_date += " 23:59:00"

    # implement dictionary for NodeId to MeterId
    lookup = {'402187':"1710884", "402188":"1710873"}
    count_dict = {}

    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='',
                                 db='smart_energy',
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
if __name__ == "__main__":
    app.run()

forecast()