from knmy import knmy
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import Python_Plots
from datetime import datetime
def reformat_date(df):
    #Refomat our date type, from YYYYMMDD, H to "YYYY-MM-DD HH:MM" this way its the same as alliander
    def reformat(date):
        date = list(str(date))
        date.insert(6,"-")
        date.insert(4,"-")
        return ''.join(date)

    def pad_zeros(hour):
        hour = str(hour-1) #We subtract 1 hour from the knmi range, sine we want it to range from 0..23 not 1..24
        hour = hour.replace(" ","")
        hour = " " + hour.zfill(2)
        hour += ":00"
        return hour

    newymd = list(map(reformat,df.YYYYMMDD.tolist()))
    newh = list(map(pad_zeros, df.H.tolist()))
    new_date = list(map(lambda a,b: a+b, newymd,newh))
    return df.assign(DATUM_TIJDSTIP = new_date)

def reformat_temp(df):
    def reformat_temp(temp):
        return float(temp)/10

    def reformat_tempff(temp_ff):
        return float(temp_ff)/10

    def reformat_tempfh(temp_fh):
        return float(temp_fh)/10

    df["T"] = list(map(reformat_temp,df["T"].tolist())) #Reformat temp, knmi records in *10 (so 5 degrees celcius is 50)
    df["FF"] = list(map(reformat_tempff, df["FF"].tolist()))
    df["FH"] = list(map(reformat_tempfh, df["FH"].tolist()))
    return df



#Given a list with years as input generates KNMI data for those years as:
#years = [year1, year2, year3]
def get_dataframe_KNMI_(years = [2021,2022]):
    #Has to be split up, cause KNMI doesnt allow files to be too large :(
    stations = get_station_names_KNMI()["STN"].tolist()
    df_knmi = pd.DataFrame(columns = list(get_column_names_KNMI()))
    for station in stations:
        for year in years:
            start = int(str(year) + "010101")
            end = int(str(year) + "123024")
            disclaimer, stations, variables, data = knmy.get_hourly_data(stations=[station], start=start, end=end, inseason=True, parse = True)
            if len(str(data["STN"].iloc[0])) == len(str(station)) and not data.empty: #safety check, some stations f up
                df_knmi = pd.concat([df_knmi,data], ignore_index = True)
    df_knmi = reformat_date(df_knmi)
    df_knmi = reformat_temp(df_knmi)
    return df_knmi

def get_dataframe_KNMI(years = [2021,2022]):
    df_knmi = pd.read_csv("\knmi_stations.csv")
    years = [str(year) for year in years]
    df_knmi = df_knmi[df_knmi['DATUM_TIJDSTIP'].astype(str).str.contains("|".join(years))]
    df_knmi = reformat_date(df_knmi)
    df_knmi = reformat_temp(df_knmi)
    return df_knmi
#Returns all station names
def get_station_names_KNMI():
    #Returns all stations names as string list
    stations = pd.read_csv("KNMI_names.csv")

    #disclaimer, stations, variables, data = knmy.get_hourly_data(start=2021010101, end=2021020202, inseason=True, parse=True)
    #stations["STN"] = stations.index
    return stations.rename(columns = {"name" : "NAME", "latitude" : "LAT", "longitude" : "LON", "altitude" : "ALT"})

#Returns column names
def get_column_names_KNMI():
    #Get all column names from knmi timetable
    df_knmi = pd.read_csv("knmi_stations.csv")
    #disclaimer, stations, variables, data = knmy.get_hourly_data(start=2021010101, end=2021020202, inseason=True,parse=True)
    return list(data.columns)

#Given a list with years & stations as input generates KNMI data for those years (with stations) as:
#Input: [year1, year2, year3], [station1, station2, station3]
def get_data_KNMI_single_station(years,stations):
    df_knmi = pd.DataFrame(columns=list(get_column_names_KNMI()))
    for station in stations:
        for year in years:
            start = int(str(year) + "010101")
            end = int(str(year) + "123024")
            disclaimer, stations, variables, data = knmy.get_hourly_data(stations=[station], start=start, end=end,
                                                                         inseason=True, parse=True)
            if len(str(data["STN"].iloc[0])) == len(str(station)):  # safety check, some stations f up
                df_knmi = pd.concat([df_knmi, data], ignore_index=True)
    df_knmi = reformat_date(df_knmi)
    df_knmi = reformat_temp(df_knmi)
    return df_knmi


#Critical regions for temperature, sun and wind - Only temp is used atm
#TODO: Rename this lol
def heat_map_list_temp(temperatures):
    def temperature_boundaries(temp):
        EXTREME_LOWER = 0
        LOW = 8
        NORMAL = 15
        EXTREME_HIGH = 25
        if temp <= EXTREME_LOWER:
            return 1
        elif temp <= LOW:
            return 2
        elif temp <= NORMAL:
            return 3
        elif temp <= EXTREME_HIGH:
            return 4
        return 5
    return list(map(temperature_boundaries,temperatures))

#Flags critical date by the functions above
#Then remove the non-critical areas and returns the dataframe
def get_critical_region(df_knmi):
    df_knmi = df_knmi.dropna(subset = ["T", "Q", "FH"]) #safety check
    temperature = np.array([float(t) for t in df_knmi["T"].tolist()])
    #temperature_F = heat_map_list_temp(temperature)
    #df_knmi = df_knmi.assign(TF = temperature_F) #temp flagged
    df_knmi = df_knmi[(df_knmi["T"] <= 0)]
   # df_knmi = df_knmi.drop(["TF"], axis = 1)  #Als dit niet werkt deze line weghalen, heeft verder geen effect
    return df_knmi

def hours(dates):
    def HOURS(date):
        lst = np.zeros(24)
        HOUR = int(date[11:13])
        lst[HOUR] = 1
        return lst
    return list(map(HOURS, dates))

def get_working_hours(df_knmi):
    datetimes = df_knmi["DATUM_TIJDSTIP"].tolist()
    def GET_HOURS(date):
        return int(date[11:13])
    HOURS = list(map(GET_HOURS, datetimes))
    pd.set_option('display.max_columns', None)
    df_knmi = df_knmi.assign(HOURS=HOURS)  # temp flagged
    df_knmi = df_knmi[(df_knmi["HOURS"]>7) & (df_knmi["HOURS"]<17)]
    df_knmi = df_knmi.drop(["HOURS"], axis=1)  # Als dit niet werkt deze line weghalen, heeft verder geen effect
    return df_knmi

def get_evening_hours(df_knmi):
    datetimes = df_knmi["DATUM_TIJDSTIP"].tolist()
    def GET_HOURS(date):
        return int(date[11:13])
    HOURS = list(map(GET_HOURS, datetimes))
    pd.set_option('display.max_columns', None)
    df_knmi = df_knmi.assign(HOURS=HOURS)  # temp flagged
    df_knmi = df_knmi[(df_knmi["HOURS"]<8) | (df_knmi["HOURS"]>16)]
    df_knmi = df_knmi.drop(["HOURS"], axis=1)  # Als dit niet werkt deze line weghalen, heeft verder geen effect
    return df_knmi

def get_holidays(): #Returns a list with dates of all holidays
    holidays = ["2022-1-%s" %str(i) for i in range(1,10)] #kerstvakantie
    holidays += ["2022-2-%s" %str(i) for i in range(19,28)] #voorjaarsvakantie
    holidays += ["2022-4-30"] #meivakantie
    holidays += ["2022-5-%s" %str(i) for i in range(1,9)] # meivakantie
    holidays += ["2022-7-%s" % str(i) for i in range(16, 32)] #zomervakantie
    holidays += ["2022-8-%s" % str(i) for i in range(1, 29)]  # zomervakantie
    holidays += ["2022-10-%s" % str(i) for i in range(16, 25)]  # herfstvakantie
    holidays += ["2022-12-%s" % str(i) for i in range(25, 32)]  # kerstvakantie

    holidays = [datetime.strptime(holiday, "%Y-%m-%d").date() for holiday in holidays]
    return holidays

def get_bank_holidays():#Returns a list with official bank_holidays for 2022
    holidays = ["2022-1-1", "2022-4-15", "2022-4-17", "2022-4-18", "2022-4-27",
                "2022-5-5", "2022-5-26", "2022-6-6", "2022-12-25", "2022-12-26"]
    return [datetime.strptime(holiday, "%Y-%m-%d").date() for holiday in holidays]

def get_unofficial_bank_holidays(): #Returns a list with unofficial bank_holidays for 2022
    holidays = ["2022-1-6", "2022-2-14", "2022-2-26","2022-2-27","2022-2-28", "2022-3-1",
                "2022-3-27","2022-4-21","2022-5-4","2022-5-8", "2022-5-19", "2022-9-20", "2022-10-4",
                "2022-10-30","2022-10-31", "2022-11-11", "2022-12-5","2022-12-15"]
    return [datetime.strptime(holiday, "%Y-%m-%d").date() for holiday in holidays]

def DATUM_TIJDSTIP_to_date(date):
    return datetime.strptime(date, "%Y-%m-%d %H:%M").date()

def get_table_extreme_errors(df): #Gets a table for all extreme errors 
    holidays = get_holidays()
    bank_holidays = get_bank_holidays()
    unofficial_bank_holidays = get_unofficial_bank_holidays()
    table = np.zeros((6,2))

    for i in range(len(df.index)):
        if float(df.iloc[i]["PREDICTED_VALUES"]) < float(df.iloc[i]["OBSERVED_VALUES"]):
            under = 1
        else:
            under = 0
        date = DATUM_TIJDSTIP_to_date(str(df.iloc[i]["DATUM_TIJDSTIP"]))
        is_holiday = False
        if date in holidays:
            table[1,under] += 1
            table[4, under] += 1
            is_holiday = True
        if date in bank_holidays:
            table[2,under] += 1
            table[4, under] += 1
            is_holiday = True
        if date in unofficial_bank_holidays:
            table[3,under] += 1
            table[4, under] += 1
            is_holiday = True
        if not is_holiday:
            table[0, under] += 1
            table[4, under] += 1
        table[5, under] += float(df.iloc[i]["ERROR_VALUES"])
    return [[float(elt) for elt in row ] for row in  table]

def get_df_reduced_temp(year, df):
    temp_values = [-17.8, -15.3, -16.8, -17.9, -19, -19.2, -18.7, -18.9, -15.6, -13.6, -11.6, -0.6,
                   -8, -7.2, -7.5, 8.8, -10.3, -13.6, -15.6, -15.7, -17.3, -17.2, -11.7, -8.9] #new_temp_profile
    #Months to be evaluated: jan, feb, sept, okt, nov, dec. So index: 01,02,09,10,11,12
    YYYYMM = [str(year)+n for n in ["01","02", "09", "10", "11", "12"]]
    for i in range(1, len(temp_values)+1):
        df.loc[(df["H"] == i) & (df['YYYYMMDD'].astype(str).str.contains("|".join(YYYYMM))), "T"] = temp_values[i-1]
        # changes the value of every hour to that of the new temperature value. This it the df["H"] condition.
        #The other condition also makes sure that we only do it for specific months chosen in advance
    return df

