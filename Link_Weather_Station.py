import numpy as np
import Database_Alliander as dba
import Database_KNMI as dbk
import math
import pandas as pd


#Open the E_MSR database which contains the coordinates of the stations
def NP_E_MSR():
    return pd.read_csv("NP_E_MSR.csv")

#open the OS_INST database which contains the connection between the stations names and station locations
def NP_OS_INST():
	return pd.read_csv("NP_OS_INST.csv")

#Given Rijksdriekhoekscoordinates returns the LONGITUDE LATITUDE GPS coordinates
def RD_to_gps(X,Y):
    dX = (X - 155000) * 10**-5
    dY = (Y - 463000) * 10**-5

    SomN = (3235.65389 * dY) + (-32.58297 * dX**2) + (-0.2475 * dY**2) + (-0.84978 * dX**2 * dY) + (
                -0.0655 * dY**3) + (-0.01709 * dX**2 * dY**2) + (-0.00738 * dX) + (0.0053 * dX**4) + (
                       -0.00039 * dX**2 * dY**3) + (0.00033 * dX**4 * dY) + (-0.00012 * dX * dY)
    SomE = (5260.52916 * dX) + (105.94684 * dX * dY) + (2.45656 * dX * dY**2) + (-0.81885 * dX**3) + (
                0.05594 * dX * dY**3) + (-0.05607 * dX**3 * dY) + (0.01199 * dY) + (-0.00256 * dX**3 * dY**2) + (
                       0.00128 * dX * dY**4) + (0.00022 * dY**2) + (-0.00022 * dX**2) + (0.00026 * dX**5)

    LAT = (52.15517 + (SomN / 3600))
    LON = (5.387206 + (SomE / 3600))
    return LON, LAT

def dist(coord1, coord2):
    #Given 2 coordinates of the form
    #coord1 = [longitude1,latitude1]
    #coord2 = [longitude2,latitude2]
    #Returns the distance between coord1 and coord2 using a lat long formula
    if coord1 == coord2:
        return 0
    cos1 = math.cos(coord1[1]*math.pi/180)
    cos2 = math.cos(coord2[1]*math.pi/180)
    cos3 = math.cos((coord1[0]-coord2[0])*math.pi/180)
    sin1 = math.sin(coord1[1]*math.pi/180)
    sin2 = math.sin(coord2[1]*math.pi/180)
    return 6370*math.acos(cos1*cos2*cos3+sin1*sin2)

#Given a station gps-coords returns the distance to every station as a zip of the form:
#[[station1, dist_to_station1],[station_2, dist_to_station2]]
def LookUpList(df, station_LON, station_LAT):
    lst = np.zeros((len(df.index),2))
    LON = df.LON.tolist()
    LAT = df.LAT.tolist()
    weather_stations = df["STN"].tolist()
    for i in range(len(df.index)):
        coord1 = [LON[i],LAT[i]]
        coord2 = [station_LON, station_LAT]
        lst[i,1] = dist(coord1,coord2)
        lst[i,0] = weather_stations[i]
    return lst

#returns a look up list of all distances
def get_weather_station_lookuplist(station, station_id):
    df_e_msr = NP_E_MSR() #open database
    df_os_inst = NP_OS_INST() #open database
    # Given a specialized station name (like OS SNEEK 10-1i) gives the standardized location name (OS SNEEK in the case of the example)
    if station == "OS": #OS special
        station_naam = df_os_inst[df_os_inst["STATION_INSTALLATIE_NAAM"] == station_id]["STATION_NAAM"].tolist()[0]
        X_COORDINAAT = df_e_msr[df_e_msr["NAAM"] == station_naam]["X_COORDINAAT"].tolist()[0] #x coordinate of that station
        Y_COORDINAAT = df_e_msr[df_e_msr["NAAM"] == station_naam]["Y_COORDINAAT"].tolist()[0] #y coordinate of that station
    else:
        X_COORDINAAT = df_e_msr[df_e_msr["NUMMER"] == station_id]["X_COORDINAAT"].tolist()[
            0]  # x coordinate of that station
        Y_COORDINAAT = df_e_msr[df_e_msr["NUMMER"] == station_id]["Y_COORDINAAT"].tolist()[
            0]  # y coordinate of that station
    LON, LAT = RD_to_gps(float(X_COORDINAAT),float(Y_COORDINAAT)) #RD to GPS - Safety cast to float
    df_weather_station = dbk.get_station_names_KNMI()
    return LookUpList(df_weather_station,LON,LAT)

#Returns the closest weather station that has valid data, if too many columns are required, might return nothing :|
def get_closest_valid_weather_station(station, df_knmi, station_id):
    # Check if the columns contain nan values for a certain station
    def is_weather_station_legit(w_station, df_knmi):
        columns = ["T", "FH", "Q"]
        df_knmi = df_knmi[df_knmi["STN"] == w_station]  # select our weather station
        for column in columns:
            if np.isnan(df_knmi[column].tolist()).any() or df_knmi.empty:
                return False
        return True

    weather_station = 0
    dist_to_station = sorted(get_weather_station_lookuplist(station, station_id),
                             key=lambda l: l[1])  # get distances to station and sort on the distance

    for possible_weather_station in dist_to_station:  # Checks for the closest weather station with valid values
        if is_weather_station_legit(possible_weather_station[0], df_knmi):
            weather_station = possible_weather_station[0]
            break
    return weather_station

#Returns the closest weather station that has valid data, if too many columns are required, might return nothing :|
def get_closest_valid_weather_station_temp(station, df_knmi, station_id,var):
    # Check if the columns contain nan values for a certain station
    def is_weather_station_legit(w_station, df_knmi):
        columns = [var]
        df = df_knmi[df_knmi["STN"] == w_station]  # select our weather station
        for column in columns:
            if np.isnan(df[column].tolist()).any() or df.empty:
                return False
        return True

    weather_station = 0
    dist_to_station = sorted(get_weather_station_lookuplist(station, station_id),
                             key=lambda l: l[1])  # get distances to station and sort on the distance

    for possible_weather_station in dist_to_station:  # Checks for the closest weather station with valid values
        if is_weather_station_legit(possible_weather_station[0], df_knmi):
            weather_station = possible_weather_station[0]
            break
    return weather_station


#Given two dataframes and a column name, makes sure that the two dataframes are the same size (important for lots of functions)
def same_sizing(df1, df2, column = "DATUM_TIJDSTIP"):
    df1 = df1[df1[column].isin(df2[column].tolist())]  # df now have the same length
    df2 = df2[df2[column].isin(df1[column].tolist())]  # df now have the same length
    return df1, df2
