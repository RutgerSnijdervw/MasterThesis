#from alliander_common_python import common, get_authentication
#from alliander_common_python.connector import base_connector
#from alliander_common_python.connector.hana_connector import HanaConnector
from typing import List, Tuple
import pandas as pd
#from hdbcli import dbapi
from datetime import datetime,timezone
from sqlalchemy import types, create_engine
from sqlalchemy.sql import text
import Link_Weather_Station as lws
import logging
from fastapi import HTTPException





#Change time zone of Alliander data
def reformat_alliander_data(df_alliander):
    df_alliander = df_alliander.drop("DATUM", axis=1)  #idk why there is a datum column, but it sucks
    time = df_alliander["DATUM_TIJDSTIP"].tolist()
    #Alliander has timezone UTC
    #KNMI has local timezone (Dutch timezone)
    #Take the string and type cast it to a datetime
    #Then use datetime to change from utc to local time
    #Then cast the datetime back to a string
    #Do this as a map for all values for efficiency
    def utc_to_local(utc_dt): #Change timezones
        utc_dt = datetime.strptime(utc_dt, "%Y-%m-%d %H:%M")
        utc_dt = utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=None)
        return utc_dt.strftime('%Y-%m-%d %H:00')
    new_time = list(map(utc_to_local, time))
    df_alliander["DATUM_TIJDSTIP"] = new_time
    return df_alliander


#Given data of every quarter (15 min) returns it resized to data for every hour
def quarter_to_hourly(df, column):
    df = df[["DATUM_TIJDSTIP", column]].copy()
    df['DATUM_TIJDSTIP'] = pd.to_datetime(df['DATUM_TIJDSTIP'])
    return df.resample('H', on='DATUM_TIJDSTIP').mean().dropna()

#Same here but on every column
#TODO: Sometimes works a bit 'eehhhh'
def quarter_to_hourly_multi_column(df, columns):
    df_copy = quarter_to_hourly(df, columns[0])
    for column in columns[1:]:
        df_ = quarter_to_hourly(df, column)
        df_copy[column] = df_[column].tolist()
    return df_copy.dropna()

#Same as before but now daily
def quarter_to_daily(df, column):
    df = df[["DATUM_TIJDSTIP", column]].copy()
    df['DATUM_TIJDSTIP'] = pd.to_datetime(df['DATUM_TIJDSTIP'])
    return df.resample('D', on='DATUM_TIJDSTIP').mean().dropna()

#Same as before but now daily
def quarter_to_monthly(df, column):
    df = df[["DATUM_TIJDSTIP", column]].copy()
    df['DATUM_TIJDSTIP'] = pd.to_datetime(df['DATUM_TIJDSTIP'])
    return df.resample('M', on='DATUM_TIJDSTIP').mean().dropna()


#rading a large csv file in chunks 
def get_header(file_name):
    chunk = pd.read_csv(file_name, chunksize=1000)
    df = chunk.get_chunk(10)
    return list(df.columns.values)

def read_csv_large_csv(file_name, found_item = False, ID = ""):
    found_item_n = 0
    df_ = pd.DataFrame(columns = get_header(""))
    for chunk in pd.read_csv(file_name, chunksize=1000000): #chunksizes
        if chunk['ASSETID'].isin([ID]).any(): #Note that we get info by name, so we are searching for stations
            df_ = pd.concat([df_, chunk[chunk["ASSETID"] == ID]], ignore_index = True)
            found_item_n += 1
            found_item = True #We have found a station in the current chunk
        elif found_item_n > 0:
            break
        if found_item and found_item_n == 0:
            return df_, found_item
    return df_,found_item

#Given a station type, id and year returns the baseload data for that station
def baseload_data(TYPE,ID,YEAR = 2021):
    #print(df1.head(10))

    df_alliander = pd.DataFrame(columns = get_header("alliander_stations1.csv"))
    df, found_item = read_csv_large_csv("alliander_stations1.csv",ID = ID)
    df_alliander = pd.concat([df_alliander, df ], ignore_index = True)
    df, found_item = read_csv_large_csv("alliander_stations2.csv",found_item, ID)
    df_alliander = pd.concat([df_alliander, df ], ignore_index = True)
    return df_alliander

#Given a dateframe of alliander takes the mean time
def taking_mean_of_time(df_alliander):
    unique_time = df_alliander.DATUM_TIJDSTIP.unique() #get all unique time values
    for time in unique_time:
        indices = df_alliander.index[df_alliander['DATUM_TIJDSTIP'] == time].tolist() #get index of time value, returns of the form [10, 40, 50]
        mean = 0
        for index in indices:
            mean += df_alliander._get_value(index, "BELASTING")  #add values
        mean /= len(indices) #take mean
        df_alliander.loc[indices[0],"BELASTING"]=mean #change value
        df_alliander = df_alliander.drop(indices[1:]) #remove other row entries
    return df_alliander

#Gets a alliander dataframe
def get_dataframe_alliander(station_id, station, df_knmi, year = 2022):
    df_alliander = taking_mean_of_time(
        reformat_alliander_data(baseload_data(station, station_id, year)))  # dataframe from alliander
    weather_station = lws.get_closest_valid_weather_station(station, df_knmi, station_id) #gets closest valid weather staiton
    df_knmi = df_knmi[df_knmi["STN"] == weather_station].copy()
    return lws.same_sizing(df_alliander.drop_duplicates(), df_knmi.drop_duplicates()) #samesizing is a safety check such that df_knmi and alliander have the same dates




