#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: omayr
@description: A multi-threading supported script to retrieve sensor data from Siemens Navigator.
"""

import time
import datetime
from threading import Thread
import base64, urllib2, json, requests
import pandas as pd


"""
Sensor Information of Keras 5, Room A510a Avotoimisto

meter_id |  room               |    sensor_type

1710876	 |  A510a Avotoimisto  |  Huonelämpötila                        (Room Temperature)
1710877	 |  A510a Avotoimisto  |  Huonelämpötilan keskiarvo             (Average room temperature)
1710868	 |  A510a Avotoimisto  |  Ilmanlaatu                            (Air Quality)
1710869	 |  A510a Avotoimisto  |  Ilmavirran jäähdytyksen asetusarvo    (Air flow cooling setpoint)
1710874	 |  A510a Avotoimisto  |  Jäähdytyksen asetusarvo               (Cooling setpoint)
1710879	 |  A510a Avotoimisto  |  Jäähdytysventtiili                    (Cooling valve)
1710872	 |  A510a Avotoimisto  |  Käyntitila                            (Running mode)
1710878	 |  A510a Avotoimisto  |  Lämmitysventtiili                     (Heating valve)
1710873	 |  A510a Avotoimisto  |  Läsnäolotieto                         (Presence Information)
1710870	 |  A510a Avotoimisto  |  Poistoilman lämpötila                 (Exhaust air temperature)

"""


def url_builder(base_url,meter_id, ts):
    return base_url+"/meter/"+meter_id+"/readings?utcStartTimestamp="\
           +ts[0]+"&utcEndTimestamp="+ts[1]

def fetch_data(thread_name, base_url, url, meter_id, ts):

    print thread_name
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

    while True:

        print "***",str(thread_name)+"-"+str(page)

        for items in j['items']:
            df.loc[counter] = [items['utcOrgRectime'],items['value']]
            counter += 1

        if j['nextPage'] == None:
            ts_end = ts[1].replace("%20", " ")
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
    df = df.drop_duplicates()
    df.to_csv('data/sample/tmp' + thread_name+'_'+meter_id+ '.csv', index=False)
    print "Total Values Fetched", counter


def main():

    """
    https://eadvantage.siemens.com/remote/release/meter/
    1710876/readings?utcStartTimestamp=04/20/2017 0:00:00
    &utcEndTimestamp=05/20/2017 11:59:59
    """

    start_time = time.time()
    with open('data/named_sensors.txt','r') as f:
        lines = f.readlines()

    meter_id_dict = {}

    for l in lines:
        l = l.rstrip()
        x = l.split(',')
        meter_id_dict[x[0]] = x[1]


    # meter_id_dict = {"1710876":"room_temperature",
    #                  "1710877":"average_room_temperateure",
    #                  "1710868":"air_quality",
    #                  "1710869":"air_flow_cooling_setpoint",
    #                  "1710874":"cooling_setpoint",
    #                  "1710879":"cooling_val",
    #                  "1710872":"running_mode",
    #                  "1710878":"heating_valve",
    #                  "1710873":"presence_information",
    #                  "1710870":"exhaust_air_temperature"}

    base_url = "https://eadvantage.siemens.com/remote/release"
    ts = ("03/01/2017%2011:00:00", "03/01/2017%2023:59:59")

    thread_list = []
    for key,value in meter_id_dict.iteritems():
        url = url_builder(base_url, key, ts)
        fetch_data(value, base_url, url, key, ts)
    #     t = Thread(target=fetch_data, args=(value, base_url, url, key, ts))
    #     t.start()
    #     thread_list.append(t)
    #
    # for t in thread_list:
    #     t.join()


    print "--- %s Minutes ---" % ((time.time() - start_time) / 60)

if __name__ == "__main__": main()
