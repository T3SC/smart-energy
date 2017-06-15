#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: omayr
@description: A multi-threading supported script to retrieve sensor data from Siemens Navigator.
"""

import thread,time
import base64, urllib2, json, requests


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


def url_builder(base_url,meter_id,utc_start_timestamp,utc_end_timestamp):
    return base_url+meter_id+"/readings?utcStartTimestamp="\
           +utc_start_timestamp+"&utcEndTimestamp="+utc_end_timestamp

def fetch_data(url,meter_id):

    username = "AkhtarTU"
    password = "BQYQELn7$!9vN]=0"
    b64 = base64.b64encode(username + ":" + password)
    token = "Basic " + b64
    print token
    print url
    r = requests.get(url,headers={"Authorization":token})
    j = json.loads(str(r._content))
    print j['items'][0]
    print "Hello Word"

def main():



    """
    https://eadvantage.siemens.com/remote/release/meter/
    1710876/readings?utcStartTimestamp=04/20/2017 0:00:00
    &utcEndTimestamp=05/20/2017 11:59:59
    """

    start_time = time.time()
    meter_id_dict = {"1710876":"room_temperature",
                     "1710877":"average_room_temperateure",
                     "1710868":"air_quality",
                     "1710869":"air_flow_cooling_setpoint",
                     "1710874":"cooling_setpoint",
                     "1710879":"cooling_val",
                     "1710872":"running_mode",
                     "1710878":"heating_valve",
                     "1710873":"presence_information",
                     "1710870":"exhaust_air_temperature"}

    base_url = "https://eadvantage.siemens.com/remote/release/meter/"
    utc_start_timestamp = "05/20/2017%2011:00:00"
    utc_end_timestamp = "05/20/2017%2011:59:59"

    for meter_id in meter_id_dict:
        url = url_builder(base_url,meter_id,utc_start_timestamp,utc_end_timestamp)
        fetch_data(url,meter_id)
        exit(0)

    print "--- %s Minutes ---" % ((time.time() - start_time) / 60)

if __name__ == "__main__": main()
