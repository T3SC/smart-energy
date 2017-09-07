# Smart Energy

Smart Energy is a project hosted by Floworks and initiated by SYK in cooperation with VIRPA-C. The vision of the project is to use sensor data from smart buildings, process it, analyze it and build applications on top of it which can benefit SYK and other stakeholders such as university, faculty and students. 

Following are the description of the project files:

## analyzer.py
The script does basic preprocessinga and reconstruction of missing data and convert the time series data into LSTM feedable format and trains a basic LSTM model for temperature forecasting. 

## dumper.py
The script retrieves data via Siemens API, implements a workaround for API pagination issue, reconstructs missing data and dumps it into a database. 

## exploratory_analysis.ipynb
The script does some basic exploratory analysis of the data which mostly includes correlation visualization. 

## occupancy_analysis.ipynb 
The script does some basic room occupancy comparision on data from two different vendors Helvar and Siemens. 

## temperature_modeler.py
The script uses data from several Kampusklubi rooms with sensors and uses the same architecture in analyzer.py to train room specific models. 

## ts_analysis.py
The script does some basic time series analysis to observer trend and seasonality of the data and further uses ARIMA to do temperature forecasting. 
