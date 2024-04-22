import numpy as np
import Controller as cll
import Database_Alliander as dba
import Link_Weather_Station as lws
import Database_KNMI as dbk
import Regression_Model as rm
import Python_Plots as pp
import Test_Stations as ts
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages
from scipy import polyval,polyfit
import pandas as pd
from copy import deepcopy
pd.options.mode.chained_assignment = None  # default='warn' so that the annoying error messages can go away



#############################################################################################################################################





#Interpolate data curves, can be moved to another file
def interpolating_data():
    STATION_TYPE = "OS"
    STATION_ID = "OS WESTZAANSTRAAT 10-1i"
    train_year = [2022]
    print("starting linear regression")
    y_predicted, y_observed, X,date, df_knmi, OLS_complete = cll.get_model_1(STATION_ID = STATION_ID,
                                                                           train_year = train_year, STATION_TYPE = STATION_TYPE)
    
    df_knmi_21 = dbk.get_dataframe_KNMI(years = [2021])
    
    df_alliander_21,df_knmi_21 = dba.get_dataframe_alliander(STATION_ID,STATION_TYPE, df_knmi_21, 2021)
    df_knmi_21["FH"] = np.mean(df_knmi_21["FH"].tolist())
    df_knmi_21["Q"] = np.mean(df_knmi_21["Q"].tolist())
    df_knmi_21_0 = df_knmi_21.copy()
    df_knmi_21["T"] = -20
    df_knmi_21_0["T"] = 0

    dates_interpolation_curve = ["2021-12-06 14:00","2021-12-06 23:00","2021-12-07 14:00","2021-12-07 23:00",
                                 "2021-12-08 14:00","2021-12-08 23:00","2021-12-09 14:00","2021-12-09 23:00",
                                 "2021-12-10 14:00","2021-12-10 23:00","2021-12-11 14:00","2021-12-11 23:00",
                                 "2021-12-12 14:00","2021-12-12 23:00"]
    dates_interpolation_curve_legend = ["monday 2021-12-06 14:00", "monday 2021-12-06 23:00", "tuesday 2021-12-07 14:00", "tuesday 2021-12-07 23:00",
                                 "wednesday 2021-12-08 14:00", "wednesday 2021-12-08 23:00", "thursday 2021-12-09 14:00", "thursday 2021-12-09 23:00",
                                 "friday 2021-12-10 14:00", "friday 2021-12-10 23:00", "saturday 2021-12-11 14:00", "saturday 2021-12-11 23:00",
                                 "sunday 2021-12-12 14:00", "sunday 2021-12-12 23:00"]

    df_alliander_21 = df_alliander_21[df_alliander_21['DATUM_TIJDSTIP'].astype(str).str.contains("|".join(dates_interpolation_curve))]
    df_knmi_21 = df_knmi_21[df_knmi_21['DATUM_TIJDSTIP'].astype(str).str.contains("|".join(dates_interpolation_curve))]
    df_knmi_21_0 = df_knmi_21_0[df_knmi_21_0['DATUM_TIJDSTIP'].astype(str).str.contains("|".join(dates_interpolation_curve))]
    

        
    

    
    # Model 1 results

    #model 2

    

    yobs_21, X_21, date_21 = rm.linear_regression_model_data_controller(df_alliander_21,df_knmi_21)
    yobs_21, X_21_0, date_21 = rm.linear_regression_model_data_controller(df_alliander_21,df_knmi_21_0)
    prediction_21 = OLS_complete.predict(X_21)
    prediction_21_0 = OLS_complete.predict(X_21_0)
    line_plot_values = []
    for i in range(len(prediction_21)):
        line_plot_values.append([prediction_21[i],prediction_21_0[i]])    
    
    pp.multiple_line_plots([-20,0],line_plot_values,dates_interpolation_curve_legend,
                           xaxis = "Temperature[Celcius]", yaxis = "Model values[kW]", png = "Interpolatie_curves.png")


    return


#Test to forecasting
def Extreme_data_points():
    STATION_TYPE = "OS"
    STATION_ID = "OS WESTZAANSTRAAT 10-1i"
    train_year = [2022]
    test_year = []
    print("getting dataframes")
    df_knmi = dbk.get_dataframe_KNMI(years = test_year +train_year)  # Get KNMI df- this way we only have to call this functions one time
    #df_alliander, df_knmi = dba.get_dataframe_alliander(STATION_ID, STATION_TYPE,
    #                                                 df_knmi)  # Get df for the test station and id (deep copy for knmi dataframe)

    print("starting linear regression")
    print("model 1")
    #Model 1
    y_predicted_OLS, y_observed_OLS,X, date, df_knmi, OLS_complete = cll.get_model_1(df_knmi_train= df_knmi, STATION_ID = STATION_ID, STATION_TYPE = STATION_TYPE, train_year = train_year, test_year = test_year)
    
    y_predicted_reduced, y_observed_reduced, X_reduced, date_reduced, df_knmi_extended_model, OLS_critical_reduced,OLS_complete = cll.get_model_OLS_extended(df_knmi_train=df_knmi, STATION_ID=STATION_ID,
                                                                                  STATION_TYPE=STATION_TYPE, train_year = train_year, test_year = test_year)

    print("model 3")
    y_predicted_GLS, y_observed_GLS,X, date, df_knmi, GLS_complete = cll.get_model_3(df_knmi_train= df_knmi, STATION_ID = STATION_ID, STATION_TYPE = STATION_TYPE, train_year = train_year, test_year = test_year)
    print("making predictions")
    #Making predictions
    #parameters_OLS = OLS_complete.params
    #df_alliander,df_knmi = dba.get_dataframe_alliander(STATION_ID,STATION_TYPE,df_knmi,year = train_year[0])
    #y_observed_test, X_reduced, date_test = rm.linear_regression_model_data_controller_reduced(df_alliander, df_knmi, parameters_OLS)
    df_knmi.to_csv("df1_test.csv", index = False)

    df_knmi = dbk.get_df_reduced_temp(train_year[0], df_knmi)
    df_knmi.to_csv("df2_test.csv", index = False)

    new_temp = df_knmi["T"].tolist()
    X[:,1] = new_temp
    X[:,2] = np.zeros(len(X[:,2]))
    df_knmi_extended_model = dbk.get_df_reduced_temp(train_year[0], df_knmi_extended_model)
    new_temp = df_knmi_extended_model["T"].tolist()
    X_reduced[:,1] = new_temp
    

    #ayooooo ff converten
    y_predicted_OLS = np.array(rm.decimal_to_float(y_predicted_OLS))
    y_observed_OLS = np.array(rm.decimal_to_float(y_observed_OLS))
    y_predicted_reduced = np.array(rm.decimal_to_float(y_predicted_reduced))
    y_observed_reduced = np.array(rm.decimal_to_float(y_observed_reduced))
    y_predicted_GLS = np.array(rm.decimal_to_float(y_predicted_GLS))
    y_observed_GLS = np.array(rm.decimal_to_float(y_observed_GLS))


    sigma = np.var(OLS_complete.resid)**0.5
    random_list = np.random.normal(0,sigma, len(OLS_complete.resid))
    standard_dev_list = [3*sigma] * len(OLS_complete.resid)
    #max_value_OLS_list = np.add(OLS_complete.predict(X) , random_list)
    #max_value_OLS_list = np.add(OLS_complete.predict(X), np.array(standard_dev_list))
    max_value_OLS_list = OLS_complete.predict(X)
    max_value_OLS = max(max_value_OLS_list)
    index = np.where(max_value_OLS_list == max_value_OLS)[0][0]
    max_value_OLS_date = list(df_knmi["DATUM_TIJDSTIP"])[index]

    sigma = np.var(OLS_critical_reduced.resid) ** 0.5
    random_list = np.random.normal(0,sigma, len(X_reduced))
    standard_dev_list = [3 * sigma] * len(X_reduced)
    max_value_OLS_reduced_list = np.add(OLS_critical_reduced.predict(X_reduced) , random_list)
    max_value_OLS_reduced_list = np.add(OLS_critical_reduced.predict(X_reduced), np.array(standard_dev_list))
    max_value_OLS_reduced_list = OLS_critical_reduced.predict(X_reduced)
    max_value_OLS_reduced = max(max_value_OLS_reduced_list)
    index = np.where(max_value_OLS_reduced_list == max_value_OLS_reduced)[0][0]
    max_value_OLS_reduced_date = list(df_knmi_extended_model["DATUM_TIJDSTIP"])[index]

    sigma = np.var(GLS_complete.resid) ** 0.5
    random_list = np.random.normal(0,sigma, len(GLS_complete.resid))
    standard_dev_list = [3 * sigma] * len(GLS_complete.resid)
    prediction_GLS = GLS_complete.predict(X)
    max_value_GLS_list = np.add(GLS_complete.predict(X) , random_list)
    max_value_GLS_list = np.add(GLS_complete.predict(X), np.array(standard_dev_list))
    max_value_GLS_list = GLS_complete.predict(X)
    max_value_GLS = max(max_value_GLS_list)
    index = np.where(max_value_GLS_list == max_value_GLS)[0][0]
    max_value_GLS_date = list(df_knmi["DATUM_TIJDSTIP"])[index]

  #  max_value_OLS_reduced = max(OLS_critical_reduced.predict(X_reduced) + np.absolute(y_predicted_reduced - y_observed_reduced))
  #  max_value_GLS = max(GLS_complete.predict(X) + np.absolute(y_predicted_GLS - y_observed_GLS))
    index = np.where(y_observed_OLS == max(y_observed_OLS))[0][0]
    date_observed_max_value = list(df_knmi["DATUM_TIJDSTIP"])[index]
    txt = STATION_ID + " - predictions \n"
    txt += "Model name | Model predicted score (kW) | Model date\n"
    txt +=("OLS: " + str(max_value_OLS) + " | " + max_value_OLS_date +"\n")
    txt += ("EXTENDED : " + str(max_value_OLS_reduced) + " | " + max_value_OLS_reduced_date+"\n")
    txt += ("GLS: " + str(max_value_GLS) + " | " + max_value_GLS_date )
    print(txt)

    pp.plot_vs_date(date, value = max_value_OLS_list,
                         xaxis="date",yaxis = "predicted value[kW]", 
                         png = "OLS_prediction.png")
    pp.plot_vs_date(date_reduced, value=max_value_OLS_reduced_list,xaxis="date",
                         yaxis="predicted value[kW]", 
                         png = "OLS_extended_prediction.png")
    pp.plot_vs_date(date, value=max_value_GLS_list,xaxis="date",
                         yaxis="predicted value[kW]", 
                         png = "GLS_prediction.png")

    return

def start_testing():
    import Testing as testing
    testing.temp_plots()
    return
start_testing()
