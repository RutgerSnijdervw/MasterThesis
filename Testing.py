import Database_KNMI as dbk
import Python_Plots as pp
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import Controller as cll
import Test_Stations as ts
import Regression_Model as rm
from scipy.stats import t
import Database_Alliander as dba
from copy import deepcopy
import Link_Weather_Station as lws
from datetime import datetime

######################################
# This file contains all functions used for testing or obtaining results
# Most often its the usage of ols/GLS/extended model and creating plots
########################################



def get_multiple_stations_cor_wind_temp():
    stations = [235,240,251,380]
    df_knmi = dbk.get_dataframe_KNMI([2022])
    all_station = np.array(df_knmi["STN"].tolist())
    all_station = np.unique(all_station)
    all_stations = []
    for x in all_station:
        if not np.isnan(df_knmi[df_knmi["STN"] == x]["T"].tolist()).any():
            if not np.isnan(df_knmi[df_knmi["STN"] == x]["FH"].tolist()).any():
                if not np.isnan(df_knmi[df_knmi["STN"] == x]["Q"].tolist()).any():
                    all_stations.append(x)
    print(all_stations)
    for station in all_stations:
        correlation_wind_temp(df_knmi,station)
    return

def standard_error(r,n):
    return np.sqrt((1-r**2)/(n-2))

def t_value(r,s):
    return r/s

def correlation_wind_temp(df_knmi,station = 251):
      print(station)
    #working weather stations
    # 235 (De kooy)
    # 240 (schiphol)
    # 251 Hoorn (goede relatie)
    # 380 Maastricht
    df_knmi = df_knmi[df_knmi["STN"] == station]
    wind_speed = list(df_knmi["FH"])
    temperature  = list(df_knmi["T"])
    solar_radiation  = list(df_knmi["Q"])
    n = len(temperature)

    
    wind_coeff = rm.pearson_correlation_coeff(wind_speed,temperature)
    wind_intercept = rm.simple_linear_regression_b_known(temperature,wind_speed,wind_coeff)
    se_temp = standard_error(wind_coeff,n)
    t_temp = wind_coeff/se_temp
    print("relation wind speed (x axis), temp (y axis) (Pearson)): ", round(wind_coeff,4))
    #print("wind_speed (standard error): ",se_temp)
    #print("wind_speed (t-value): ",t_temp)
    print("wind_speed (p - value): ", round(t.sf(np.abs(t_temp), n-2)*2,4))
    solar_coeff = rm.pearson_correlation_coeff(solar_radiation, temperature)
    solar_intercept = rm.simple_linear_regression_b_known(temperature, solar_radiation, solar_coeff)
    se_solar = standard_error(solar_coeff,n)
    t_solar = solar_coeff/se_solar
    print("relation solar radiation (x axis) and temp (y axis) (Pearson): ", round(solar_coeff,4))
    #print("solar (standard error): ",se_solar)
    #print("solar (t-value): ",t_solar)
    print("solar (p - value): ", round(t.sf(np.abs(t_solar), n-2)*2,4))

    
    
    pp.scatterplot_with_corr_line(wind_speed,temperature,
                   xaxis= "Wind speed[m/s]", yaxis= "Temperature[Celcius]" ,
                   png = "wind_temp_cor%d.png" %station)



    pp.scatterplot_with_corr_line(solar_radiation, temperature,
                   xaxis = "Solar radiation[J/cm2]", yaxis = "                   ",
                   png = "solar_temp_cor%d.png"%station)


    X_wind_speed = np.column_stack((np.ones((len(wind_speed), 1)), wind_speed))
    OLS_wind_speed = rm.linear_regression_statsmodels(temperature, X_wind_speed)
   # print(rm.simple_linear_regression(wind_speed,temperature))
  #  print(OLS_wind_speed.summary())

    X_solar = np.column_stack((np.ones((len(solar_radiation), 1)), solar_radiation))
    OLS_solar= rm.linear_regression_statsmodels(temperature, X_solar)
  #  print(rm.simple_linear_regression(solar_radiation,temperature))
  #  print(OLS_solar.summary())

   # pdf.close()
    return

def align_list_ljust(lst):
    longest_length = max([len(str(s)) for s in lst])
    return [str(ele).ljust(longest_length) for ele in lst]

def align_matrix_ljust(mtx):
    size_matrix = len(mtx[0])
    for i in range(size_matrix):
        mtx[:, i] = align_list_ljust(mtx[:, i])
    return mtx

def autocorrelation_test():
    STATION_TYPE = "OS"  # "MSR"#
    STATION_ID = "OS WESTZAANSTRAAT 10-1i"  # "1 002 001"#
    train_year = [2022]
    test_year = []
    df_knmi = dbk.get_dataframe_KNMI(
        years=test_year + train_year)  # Get KNMI df- this way we only have to call this functions one time
    # df_alliander, df_knmi = dba.get_dataframe_alliander(STATION_ID, STATION_TYPE,
    #                                                 df_knmi)  # Get df for the test station and id (deep copy for knmi dataframe)

    print("starting linear regression")
    print("model 1")
    # Model 1
    y_predicted_OLS, y_observed_OLS, X, date, df_knmi, OLS_complete = cll.get_model_1(df_knmi_train=df_knmi,
                                                                                      STATION_ID=STATION_ID,
                                                                                      STATION_TYPE=STATION_TYPE,
                                                                                      train_year=train_year,
                                                                                      test_year=test_year)

    y_predicted_GLS, y_observed_GLS, X, date, df_knmi, GLS_complete = cll.get_model_3(df_knmi_train=df_knmi,
                                                                                      STATION_ID=STATION_ID,
                                                                                      STATION_TYPE=STATION_TYPE,
                                                                                      train_year=train_year,
                                                                                      test_year=test_year)
    y_observed = rm.decimal_to_float(y_observed_OLS)
    y_predicted = rm.decimal_to_float(y_predicted_OLS)
    residual = (y_observed - y_predicted)
    d = rm.durbin_watson(residual)

    pp.ACF_plot(residual, lags = 24*4,png = "ACF_OLS.png")

    y_observed = rm.decimal_to_float(y_observed_GLS)
    y_predicted = rm.decimal_to_float(y_predicted_GLS)
    residual = (y_observed - y_predicted)
    d = rm.durbin_watson(residual)

    pp.ACF_plot(residual, lags=24 * 4, png="ACF_GLS.png")
    return


def variables_selection():
    STATION_TYPE = "OS"  # "MSR"#
    STATION_ID = "OS MIDDENMEER 20-1i"  # "1 002 001"#
    train_year = [2022]
    test_year = []
    # df_alliander, df_knmi = dba.get_dataframe_alliander(STATION_ID, STATION_TYPE,
    #                                                 df_knmi)  # Get df for the test station and id (deep copy for knmi dataframe)

    print("starting linear regression")
    print("model 1")
    # Model 1
    OLS_complete = cll.get_model_OLS_variables_test(STATION_ID=STATION_ID,STATION_TYPE=STATION_TYPE,train_year=train_year)
    return

def extreme_points_detail():
    STATION_TYPE = "OS"#"MSR"#
    STATION_ID = "OS WESTZAANSTRAAT 10-1i"#"1 002 001"#
    train_year = [2022]
    test_year = []
    print("getting dataframes")
    df_knmi = dbk.get_dataframe_KNMI(
        years=test_year + train_year)  # Get KNMI df- this way we only have to call this functions one time
    # df_alliander, df_knmi = dba.get_dataframe_alliander(STATION_ID, STATION_TYPE,
    #                                                 df_knmi)  # Get df for the test station and id (deep copy for knmi dataframe)

    print("starting linear regression")
    print("model 1")
    # Model 1
    y_predicted_OLS, y_observed_OLS, X, date, df_knmi, OLS_complete = cll.get_model_1(df_knmi_train=df_knmi,
                                                                                      STATION_ID=STATION_ID,
                                                                                      STATION_TYPE=STATION_TYPE,
                                                                                      train_year=train_year,
                                                                                      test_year=test_year)
    mape_score = [round(rm.MAPE([y_predicted_OLS[i]],[y_observed_OLS[i]])*100,2) for i in range(len(y_predicted_OLS))]
    df_knmi["MAPE_SCORES"] = mape_score
    df_knmi["PREDICTED_VALUES"] = y_predicted_OLS
    df_knmi["OBSERVED_VALUES"] = y_observed_OLS
    
    abs_error = np.absolute(np.array(y_predicted_OLS) - np.array(y_observed_OLS))
    #abs_error = [abs_error[i]/y_observed_OLS[i] for i in range(len(abs_error))]
    df_knmi["ERROR_VALUES"] = abs_error
    
    

    
    df_knmi_2 = df_knmi[["YYYYMMDD","H","DATUM_TIJDSTIP","MAPE_SCORES","PREDICTED_VALUES","OBSERVED_VALUES","ERROR_VALUES"]]
    df_knmi_3 = df_knmi[["YYYYMMDD","H","DATUM_TIJDSTIP","MAPE_SCORES","PREDICTED_VALUES","OBSERVED_VALUES","ERROR_VALUES"]]
    df_knmi_2 = df_knmi_2.sort_values(by = ["ERROR_VALUES"], ascending = False)
    df_knmi_3 = df_knmi_3.sort_values(by = ["MAPE_SCORES"], ascending = False)
    df_knmi_2 = df_knmi_2.head(200)
    df_knmi_3 = df_knmi_3.head(200)
    
    pd.set_option('display.max_columns', None)



        
    data_df_2 = [[str(ele) for ele in sub] for sub in df_knmi_2.values]
    df_knmi_values_2 = np.matrix(df_knmi_2.columns)
    df_knmi_values_2 = np.append(df_knmi_values_2, np.matrix(data_df_2), axis=0)
    df_knmi_values_2 = np.array(df_knmi_values_2)
    for i in range(len(df_knmi_2.columns)):
        df_knmi_values_2[:, i] = align_list_ljust(df_knmi_values_2[:, i])

    data_df_3 = [[str(ele) for ele in sub] for sub in df_knmi_3.values]
    df_knmi_values_3 = np.matrix(df_knmi_3.columns)
    df_knmi_values_3 = np.append(df_knmi_values_3, np.matrix(data_df_3), axis=0)
    df_knmi_values_3 = np.array(df_knmi_values_3)
    for i in range(len(df_knmi_3.columns)):
        df_knmi_values_3[:, i] = align_list_ljust(df_knmi_values_3[:, i])

    
    np.savetxt(r'error_analysis_error_values.txt', df_knmi_values_2, fmt='%s')

    np.savetxt(r'error_analysis_mape_values.txt', df_knmi_values_3, fmt='%s')

    table = dbk.get_table_extreme_errors(df_knmi_3)
    table_txt = [["","overfit", "underfit"]]
    columns = ["Normal day", "Holiday", "Bank holiday", "Unofficial bank holiday", "Amount of points", "Total error all data points"]
    for i in range(len(table)):
        table_txt.append([columns[i]] + [str(int(round(elt))) for elt in table[i]])
    table_txt = align_matrix_ljust(np.array(table_txt))
    np.savetxt(r'table_holidays_over_under_prediction_mape.txt', table_txt,
               fmt='%s')

    table = dbk.get_table_extreme_errors(df_knmi_2)
    table_txt = [["","overfit", "underfit"]]
    columns = ["Normal day", "Holiday", "Bank holiday", "Unofficial bank holiday", "Amount of points", "Total error all data points"]
    for i in range(len(table)):
        table_txt.append([columns[i]] + [str(int(round(elt))) for elt in table[i]])
    table_txt = align_matrix_ljust(np.array(table_txt))
    np.savetxt(r'table_holidays_over_under_prediction_error.txt', table_txt,
               fmt='%s')
    return

def normal_distribution_residual():
    STATION_TYPE = "OS"
    STATION_ID = "OS WESTZAANSTRAAT 10-1i"
    test_year = [2022]
    train_year = [2021]

    print("getting dataframes")
    df_knmi = dbk.get_dataframe_KNMI(
        years= train_year)  # Get KNMI df- this way we only have to call this functions one time
    # df_alliander, df_knmi = dba.get_dataframe_alliander(STATION_ID, STATION_TYPE,
    #                                                 df_knmi)  # Get df for the test station and id (deep copy for knmi dataframe)

    print("starting linear regression")
    print("model 1")
    # Model 1
    y_predicted_OLS, y_observed_OLS, X, date, df_knmi, OLS_complete = cll.get_model_1(df_knmi=df_knmi,
                                                                                      STATION_ID=STATION_ID,
                                                                                      STATION_TYPE=STATION_TYPE,
                                                                                      train_year=train_year,
                                                                                      test_year = test_year
                                                                                      )
    residual = OLS_complete.resid
    print(residual)
    from matplotlib import pyplot as plt
    fig,ax = plt.subplots(figsize = (10,7))
    ax.hist(residual)
    plt.show()
    print(np.mean(residual))
    print(np.var(residual))
    return

def observe_external_events():
    STATION_TYPE = "OS"
    STATION_ID = "OS WESTZAANSTRAAT 10-1i"
    test_year = []
    train_year = [2022]

    print("getting dataframes")
    df_knmi = dbk.get_dataframe_KNMI(
        years= train_year)  # Get KNMI df- this way we only have to call this functions one time
    # df_alliander, df_knmi = dba.get_dataframe_alliander(STATION_ID, STATION_TYPE,
    #                                                 df_knmi)  # Get df for the test station and id (deep copy for knmi dataframe)

    print("starting linear regression")
    print("model 1")
    # Model 1
    y_predicted_OLS, y_observed_OLS, X, date, df_knmi, OLS_complete = cll.get_model_1(df_knmi_test=df_knmi,
                                                                                      STATION_ID=STATION_ID,
                                                                                      STATION_TYPE=STATION_TYPE,
                                                                                      train_year=train_year,
                                                                                      test_year = test_year
                                                                                      )

    mape_score = [rm.MAPE([y_predicted_OLS[i]],[y_observed_OLS[i]])*100 for i in range(len(y_predicted_OLS))]
    pp.plot_vs_date(date,mape_score, xaxis= "Date", yaxis= "Mape score[%]",
                   png = "mape_events_%s.png"%str(STATION_ID))
    return


def train_test_station(STATION_ID):
    STATION_TYPE = "MSR"
    train_year = [2021]
    test_year = [2022]
    df_knmi = dbk.get_dataframe_KNMI(
        years=test_year + train_year)  # Get KNMI df- this way we only have to call this functions one time
    # df_alliander, df_knmi = dba.get_dataframe_alliander(STATION_ID, STATION_TYPE,
    #                                                 df_knmi)  # Get df for the test station and id (deep copy for knmi dataframe)

    # Model 1
    y_predicted_OLS, y_observed_OLS, X, date, df_knmi, OLS_complete = cll.get_model_1(STATION_ID=STATION_ID,
                                                                                      STATION_TYPE=STATION_TYPE,
                                                                                      train_year=train_year,
                                                                                      test_year=test_year)
    return y_predicted_OLS, y_observed_OLS,OLS_complete


def create_csv_file():
    header2 = ["station", "OLS_TEST_R2", "OLS_TEST_MSE", "OLS_TEST_MAE","OLS_TEST_MAPE", "OLS_TEST_SDE", "OLS_TEST_DB",
                "OLS_TRAIN_R2", "OLS_TRAIN_MSE","OLS_TRAIN_MAE","OLS_TRAIN_MAPE" ,"OLS_TRAIN_SDE", "OLS_TRAIN_DB",
                "GLS_TEST_R2", "GLS_TEST_MSE", "GLS_TEST_MAE","GLS_TEST_MAPE", "GLS_TEST_SDE", "GLS_TEST_DB",
                "GLS_TRAIN_R2", "GLS_TRAIN_MSE", "GLS_TRAIN_MAE","GLS_TRAIN_MAPE", "GLS_TRAIN_SDE", "GLS_TRAIN_DB"]
    df = pd.DataFrame(columns = header2)
    df.to_csv("train_test_results_22_21.csv", index = False)
    return

def create_csv_file_extended():
    header2 = ["station", "R2", "MSE", "MAE","MAPE", "SDE", "DB",]
    df = pd.DataFrame(columns = header2)
    df.to_csv("train_test_results_22_21_extended.csv", index = False)
    return

def create_csv_file_AIC_BIC():
    header2 = ["station", "OLS_AIC", "OLS_BIC", "GLS_AIC", "GLS_BIC", "Extended_AIC", "Extended_BIC"]
    df = pd.DataFrame(columns = header2)
    df.to_csv("train_test_results_22_21_AIC_BIC.csv", index = False)
    return
def create_csv_file_outlier():
    header2 = ["station", "OLS_TEST_R2", "OLS_TEST_MSE", "OLS_TEST_MAE","OLS_TEST_MAPE", "OLS_TEST_SDE", "OLS_TEST_DB",
                "OLS_TRAIN_R2", "OLS_TRAIN_MSE","OLS_TRAIN_MAE","OLS_TRAIN_MAPE" ,"OLS_TRAIN_SDE", "OLS_TRAIN_DB",
                "GLS_TEST_R2", "GLS_TEST_MSE", "GLS_TEST_MAE","GLS_TEST_MAPE", "GLS_TEST_SDE", "GLS_TEST_DB",
                "GLS_TRAIN_R2", "GLS_TRAIN_MSE", "GLS_TRAIN_MAE","GLS_TRAIN_MAPE", "GLS_TRAIN_SDE", "GLS_TRAIN_DB"]
    df = pd.DataFrame(columns = header2)
    df.to_csv("train_test_results_22_21_outlier.csv", index = False)
    return


def process_results(STATION_ID,y_predicted_train_OLS,y_predicted_test_OLS,y_observed_train_OLS,y_observed_test_OLS,OLS_complete,
                                                 y_predicted_train_GLS,y_predicted_test_GLS,y_observed_train_GLS,y_observed_test_GLS,GLS_complete, doTest = True):
    results_single_station = []
    results_single_station.append(STATION_ID)
    
    if rm.Rsquared(y_observed_test_OLS, y_predicted_test_OLS) < 0 and doTest:
        return []
    
    residual_OLS_test = (rm.decimal_to_float(y_observed_test_OLS) - rm.decimal_to_float(y_predicted_test_OLS))
    results_single_station.append(round(rm.Rsquared(y_observed_test_OLS, y_predicted_test_OLS), 3))
    results_single_station.append(round(rm.MSE(y_observed_test_OLS, y_predicted_test_OLS), 3))
    results_single_station.append(round(rm.MAE(y_observed_test_OLS, y_predicted_test_OLS), 3))
    results_single_station.append(round(rm.MAPE(y_observed_test_OLS, y_predicted_test_OLS), 3))
    results_single_station.append(round(np.var(OLS_complete.resid) ** 0.5, 3))
    results_single_station.append(round(rm.durbin_watson(residual_OLS_test),3))
    
    
    if rm.Rsquared(y_observed_train_OLS, y_predicted_train_OLS) <0:
        return []
    residual_OLS_train = (rm.decimal_to_float(y_observed_train_OLS) - rm.decimal_to_float(y_predicted_train_OLS))
    results_single_station.append(round(rm.Rsquared(y_observed_train_OLS, y_predicted_train_OLS), 3))
    results_single_station.append(round(rm.MSE(y_observed_train_OLS, y_predicted_train_OLS), 3))
    results_single_station.append(round(rm.MAE(y_observed_train_OLS, y_predicted_train_OLS), 3))
    results_single_station.append(round(rm.MAPE(y_observed_train_OLS, y_predicted_train_OLS), 3))
    results_single_station.append(round(np.var(OLS_complete.resid) ** 0.5, 3))
    results_single_station.append(round(rm.durbin_watson(residual_OLS_train),3))

    if rm.Rsquared(y_predicted_test_GLS, y_observed_test_GLS)< 0:
        return []
    residual_GLS_test = (rm.decimal_to_float(y_observed_test_GLS) - rm.decimal_to_float(y_predicted_test_GLS))
    results_single_station.append(round(rm.Rsquared(y_predicted_test_GLS, y_observed_test_GLS), 3))
    results_single_station.append(round(rm.MSE(y_predicted_test_GLS, y_observed_test_GLS), 3))
    results_single_station.append(round(rm.MAE(y_predicted_test_GLS, y_observed_test_GLS), 3))
    results_single_station.append(round(rm.MAPE(y_predicted_test_GLS, y_observed_test_GLS), 3))
    results_single_station.append(round(np.var(GLS_complete.resid) ** 0.5, 3))
    results_single_station.append(round(rm.durbin_watson(residual_GLS_test),3))

    if rm.Rsquared(y_predicted_train_GLS, y_observed_train_GLS)< 0:
        return []
    residual_GLS_train = (rm.decimal_to_float(y_observed_train_GLS) - rm.decimal_to_float(y_predicted_train_GLS))
    results_single_station.append(round(rm.Rsquared(y_predicted_train_GLS, y_observed_train_GLS), 3))
    results_single_station.append(round(rm.MSE(y_predicted_train_GLS, y_observed_train_GLS), 3))
    results_single_station.append(round(rm.MAE(y_predicted_train_GLS, y_observed_train_GLS), 3))
    results_single_station.append(round(rm.MAPE(y_predicted_train_GLS, y_observed_train_GLS), 3))
    results_single_station.append(round(np.var(GLS_complete.resid) ** 0.5, 3))
    results_single_station.append(round(rm.durbin_watson(residual_GLS_train),3))
    return results_single_station


def train_test_data_set():
    STATION_TYPE = "MSR"
    df_MSR_stations = pd.read_excel("MSR_small_changes.xlsx")
    
    STATIONS_IDS = df_MSR_stations["MSR"].tolist()
    train_year = [2022]
    test_year = [2021]
   # df_knmi_train = dbk.get_dataframe_KNMI(
   #     years=train_year)
   # df_knmi_test = dbk.get_dataframe_KNMI(
   #     years=test_year)
   
    header2 = ["station", "OLS_TEST_R2", "OLS_TEST_MSE", "OLS_TEST_MAE","OLS_TEST_MAPE", "OLS_TEST_SDE", "OLS_TEST_DB",
                "OLS_TRAIN_R2", "OLS_TRAIN_MSE","OLS_TRAIN_MAE","OLS_TRAIN_MAPE" ,"OLS_TRAIN_SDE", "OLS_TRAIN_DB",
                "GLS_TEST_R2", "GLS_TEST_MSE", "GLS_TEST_MAE","GLS_TEST_MAPE", "GLS_TEST_SDE", "GLS_TEST_DB",
                "GLS_TRAIN_R2", "GLS_TRAIN_MSE", "GLS_TRAIN_MAE","GLS_TRAIN_MAPE", "GLS_TRAIN_SDE", "GLS_TRAIN_DB"]
    results = []
    df_knmi_train = dbk.get_dataframe_KNMI(years = train_year)
    df_knmi_test = dbk.get_dataframe_KNMI(years=test_year)
    df_train_test_ = pd.read_csv("train_test_results_22_21.csv")
    stations_to_be_removed = df_train_test_["station"].tolist()
    for station in stations_to_be_removed:
        STATIONS_IDS = STATIONS_IDS
        STATIONS_IDS = [x for x in STATIONS_IDS if x != station]
    for STATION_ID in STATIONS_IDS:
        print(STATION_ID)
        y_predicted_train_OLS, y_predicted_test_OLS, y_observed_train_OLS, y_observed_test_OLS, OLS_complete = cll.get_model_1_special_case(df_knmi_train=df_knmi_train,
                                        df_knmi_test=df_knmi_test,
                                        STATION_ID=STATION_ID,
                                        STATION_TYPE=STATION_TYPE,
                                        train_year=train_year,
                                        test_year=test_year)
        y_predicted_train_GLS, y_predicted_test_GLS, y_observed_train_GLS, y_observed_test_GLS, GLS_complete = cll.get_model_3_special_case(df_knmi_train=df_knmi_train,
                                        df_knmi_test=df_knmi_test,
                                        STATION_ID=STATION_ID,
                                        STATION_TYPE=STATION_TYPE,
                                        train_year=train_year,
                                        test_year=test_year)
           # y_predicted_OLS, y_observed_OLS, X, date, df_knmi, OLS_complete = cll.get_model_1(df_knmi_train= df_knmi_train,
           #                                                                                       df_knmi_test=df_knmi_test,
           #                                                                                         STATION_ID=STATION_ID,
           #                                                                                       STATION_TYPE=STATION_TYPE,
           #                                                                                       train_year=train_year,
           #                                                                                       test_year=test_year)
           # y_predicted_GLS, y_observed_GLS, X, date, df_knmi, GLS_complete = cll.get_model_3(
           #    # df_knmi_train= df_knmi_train,
           #    # df_knmi_test=df_knmi_test,
           #     STATION_ID=STATION_ID,
           #     STATION_TYPE=STATION_TYPE,
           #     train_year=train_year,
           #     test_year=test_year)
        results_single_station = process_results(STATION_ID,y_predicted_train_OLS,y_predicted_test_OLS,y_observed_train_OLS,y_observed_test_OLS,OLS_complete,
                                                 y_predicted_train_GLS,y_predicted_test_GLS,y_observed_train_GLS,y_observed_test_GLS,GLS_complete)
        if results_single_station != []:
            df_train_test = pd.read_csv("train_test_results_22_21.csv")
            df_train_test.loc[len(df_train_test.index)] = results_single_station
            df_train_test.to_csv("train_test_results_22_21.csv", index = False)
            
            

    #results_combined = np.append(np.matrix(header1), np.matrix(header2), axis=0)
    #results_combined = np.append(np.matrix(results_combined), np.matrix(results), axis=0)
    #results_combined = align_matrix_ljust(np.array(results_combined))
    #np.savetxt(r'/home/as2-streaming-user/MyFiles/TemporaryFiles/train_test_results.txt', results_combined,
    #           fmt='%s')

    return

def train_test_data_set_extended():
    STATION_TYPE = "MSR"
    df_MSR_stations = pd.read_excel("MSR_small_changes.xlsx")
    
    STATIONS_IDS = df_MSR_stations["MSR"].tolist()
    train_year = [2022]
    test_year = [2021]
   # df_knmi_train = dbk.get_dataframe_KNMI(
   #     years=train_year)
   # df_knmi_test = dbk.get_dataframe_KNMI(
   #     years=test_year)
   
    header2 = ["station", "R2", "MSE", "MAE","MAPE", "SDE", "DB"]
    results = []
    df_knmi_train = dbk.get_dataframe_KNMI(years = train_year)
    df_knmi_test = dbk.get_dataframe_KNMI(years=test_year)
    df_train_test_ = pd.read_csv("train_test_results_22_21_extended.csv")
    stations_to_be_removed = df_train_test_["station"].tolist()
    for station in stations_to_be_removed:
        STATIONS_IDS = STATIONS_IDS
        STATIONS_IDS = [x for x in STATIONS_IDS if x != station]
    for STATION_ID in STATIONS_IDS:
        print(STATION_ID)
        y_predicted, y_observed,  X, date, df_knmi, OLS_reduced, OLS_complete = cll.get_model_OLS_extended(df_knmi_train=df_knmi_train,
                                        STATION_ID=STATION_ID,
                                        STATION_TYPE=STATION_TYPE,
                                        train_year=train_year)
        
        
        results_single_station = []
        if rm.Rsquared(y_predicted, y_observed)>= 0 :
            residual = (rm.decimal_to_float(y_observed) - rm.decimal_to_float(y_predicted))
            results_single_station.append(STATION_ID)
            results_single_station.append(round(rm.Rsquared(y_predicted, y_observed), 3))
            results_single_station.append(round(rm.MSE(y_predicted, y_observed), 3))
            results_single_station.append(round(rm.MAE(y_predicted, y_observed), 3))
            results_single_station.append(round(rm.MAPE(y_predicted, y_observed), 3))
            results_single_station.append(round(np.var(OLS_reduced.resid) ** 0.5, 3))
            results_single_station.append(round(rm.durbin_watson(residual),3))
            
        if results_single_station != []:
            df_train_test = pd.read_csv("\train_test_results_22_21_extended.csv")
            df_train_test.loc[len(df_train_test.index)] = results_single_station
            df_train_test.to_csv("train_test_results_22_21_extended.csv", index = False)
            
            

    #results_combined = np.append(np.matrix(header1), np.matrix(header2), axis=0)
    #results_combined = np.append(np.matrix(results_combined), np.matrix(results), axis=0)
    #results_combined = align_matrix_ljust(np.array(results_combined))
    #np.savetxt(r'/home/as2-streaming-user/MyFiles/TemporaryFiles/train_test_results.txt', results_combined,
    #           fmt='%s')

    return

def train_test_data_set_AIC_BIC():
    STATION_TYPE = "MSR"
    df_MSR_stations = pd.read_excel("MSR_small_changes.xlsx")
    
    STATIONS_IDS = df_MSR_stations["MSR"].tolist()
    train_year = [2022]
    test_year = [2021]
   # df_knmi_train = dbk.get_dataframe_KNMI(
   #     years=train_year)
   # df_knmi_test = dbk.get_dataframe_KNMI(
   #     years=test_year)
   
    header2 = ["station", "OLS_AIC", "OLS_BIC", "GLS_AIC", "GLS_BIC", "Extended_AIC", "Extended_BIC"]
    results = []
    df_knmi_train = dbk.get_dataframe_KNMI(years = train_year)
    df_knmi_test = dbk.get_dataframe_KNMI(years=test_year)
    df_train_test_ = pd.read_csv("train_test_results_22_21_AIC_BIC.csv")
    stations_to_be_removed = df_train_test_["station"].tolist()
    for station in stations_to_be_removed:
        STATIONS_IDS = STATIONS_IDS
        STATIONS_IDS = [x for x in STATIONS_IDS if x != station]
    for STATION_ID in STATIONS_IDS:
        print(STATION_ID)
        y_predicted, y_observed,  X, date, df_knmi, OLS_reduced, OLS_extended = cll.get_model_OLS_extended(df_knmi_train=df_knmi_train,
                                        STATION_ID=STATION_ID,
                                        STATION_TYPE=STATION_TYPE,
                                        train_year=train_year)
        y_predicted_train_OLS, y_predicted_test_OLS, y_observed_train_OLS, y_observed_test_OLS, OLS_complete = cll.get_model_1_special_case(df_knmi_train=df_knmi_train,
                                        df_knmi_test=df_knmi_test,
                                        STATION_ID=STATION_ID,
                                        STATION_TYPE=STATION_TYPE,
                                        train_year=train_year,
                                        test_year=test_year)
        y_predicted_train_GLS, y_predicted_test_GLS, y_observed_train_GLS, y_observed_test_GLS, GLS_complete = cll.get_model_3_special_case(df_knmi_train=df_knmi_train,
                                        df_knmi_test=df_knmi_test,
                                        STATION_ID=STATION_ID,
                                        STATION_TYPE=STATION_TYPE,
                                        train_year=train_year,
                                        test_year=test_year)
        
        results_single_station = []
        
        results_single_station.append(STATION_ID)
        results_single_station.append(round(OLS_complete.aic, 3))
        results_single_station.append(round(OLS_complete.bic, 3))
        results_single_station.append(round(GLS_complete.aic, 3))
        results_single_station.append(round(GLS_complete.bic, 3))
        results_single_station.append(round(OLS_reduced.aic, 3))
        results_single_station.append(round(OLS_reduced.bic,3))
            
        if results_single_station != []:
            df_train_test = pd.read_csv("train_test_results_22_21_AIC_BIC.csv")
            df_train_test.loc[len(df_train_test.index)] = results_single_station
            df_train_test.to_csv("train_test_results_22_21_AIC_BIC.csv", index = False)
    return


def train_test_data_set_outliers():
    STATION_TYPE = "MSR"
    df_MSR_stations = pd.read_excel("MSR_small_changes.xlsx")
    
    STATIONS_IDS = df_MSR_stations["MSR"].tolist()
    train_year = [2022]
    test_year = [2021]
   
    header2 = ["station", "OLS_TEST_R2", "OLS_TEST_MSE", "OLS_TEST_MAE","OLS_TEST_MAPE", "OLS_TEST_SDE", "OLS_TEST_DB",
                "OLS_TRAIN_R2", "OLS_TRAIN_MSE","OLS_TRAIN_MAE","OLS_TRAIN_MAPE" ,"OLS_TRAIN_SDE", "OLS_TRAIN_DB",
                "GLS_TEST_R2", "GLS_TEST_MSE", "GLS_TEST_MAE","GLS_TEST_MAPE", "GLS_TEST_SDE", "GLS_TEST_DB",
                "GLS_TRAIN_R2", "GLS_TRAIN_MSE", "GLS_TRAIN_MAE","GLS_TRAIN_MAPE", "GLS_TRAIN_SDE", "GLS_TRAIN_DB"]
    results = []
    df_knmi_train = dbk.get_dataframe_KNMI(years = train_year)
    df_knmi_test = dbk.get_dataframe_KNMI(years=test_year)
    df_train_test_ = pd.read_csv("train_test_results_22_21.csv")
    stations_to_be_removed = df_train_test_["station"].tolist()
    for station in stations_to_be_removed:
        STATIONS_IDS = STATIONS_IDS
        STATIONS_IDS = [x for x in STATIONS_IDS if (x != station and x < stations_to_be_removed[-1])]
    for STATION_ID in STATIONS_IDS:
        print(STATION_ID)
        y_predicted_train_OLS, y_predicted_test_OLS, y_observed_train_OLS, y_observed_test_OLS, OLS_complete = cll.get_model_1_special_case(df_knmi_train=df_knmi_train,
                                        df_knmi_test=df_knmi_test,
                                        STATION_ID=STATION_ID,
                                        STATION_TYPE=STATION_TYPE,
                                        train_year=train_year,
                                        test_year=test_year)
        y_predicted_train_GLS, y_predicted_test_GLS, y_observed_train_GLS, y_observed_test_GLS, GLS_complete = cll.get_model_3_special_case(df_knmi_train=df_knmi_train,
                                        df_knmi_test=df_knmi_test,
                                        STATION_ID=STATION_ID,
                                        STATION_TYPE=STATION_TYPE,
                                        train_year=train_year,
                                        test_year=test_year)

        results_single_station = process_results(STATION_ID,y_predicted_train_OLS,y_predicted_test_OLS,y_observed_train_OLS,y_observed_test_OLS,OLS_complete,
                                                 y_predicted_train_GLS,y_predicted_test_GLS,y_observed_train_GLS,y_observed_test_GLS,GLS_complete, doTest = False)
        df_train_test = pd.read_csv("train_test_results_22_21_outlier.csv")
        df_train_test.loc[len(df_train_test.index)] = results_single_station
        df_train_test.to_csv("train_test_results_22_21_outlier.csv", index = False)

    return

def compare_outliers():
    STATION_TYPE = "MSR"
    df_MSR_stations = pd.read_excel("MSR_small_changes.xlsx")
    
    STATIONS_IDS = df_MSR_stations["MSR"].tolist()
    train_year = [2022]
    test_year = [2021]
    df_train_test = pd.read_csv("train_test_results_22_21.csv")["station"].tolist()
    df_train_test_outlier = pd.read_csv("train_test_results_22_21_outlier.csv")["station"].tolist()

    STATION_IDS = [df_train_test_outlier[0],df_train_test[0]]


    
    for STATION_ID in STATION_IDS:
        print(STATION_ID)
        train_year = [2022]
        test_year = [2021] 
        y_predicted_GLS, y_observed_GLS,X, date, df_knmi, GLS_complete = cll.get_model_3(STATION_ID = STATION_ID, STATION_TYPE = STATION_TYPE, train_year = train_year, test_year = test_year)
        y_predicted_OLS, y_observed_OLS,X, date, df_knmi, OLS_complete = cll.get_model_1(STATION_ID = STATION_ID, STATION_TYPE = STATION_TYPE, train_year = train_year, test_year = test_year)
       # y_predicted_OLS_extended, y_observed_OLS_extended,X, date, df_knmi_, OLS_extended,OLS_complete = cll.get_model_OLS_extended(STATION_ID = STATION_ID, STATION_TYPE = STATION_TYPE, train_year = train_year, test_year = test_year)


        residual_GLS = (rm.decimal_to_float(y_observed_GLS) - rm.decimal_to_float(y_predicted_GLS))
        residual_OLS = (rm.decimal_to_float(y_observed_OLS) - rm.decimal_to_float(y_predicted_OLS))
      #  residual_OLS_extended = (rm.decimal_to_float(y_observed_OLS_extended) - rm.decimal_to_float(y_predicted_OLS_extended))
        
        y_predicted_OLS_ALL_VALUES.append(y_predicted_OLS)
        
        print("GLS: SSE,Rsquared,MSE,MAE,MAPE,durbinwatson")
        print(rm.SSE(y_observed_GLS,y_predicted_GLS))
        print(rm.Rsquared(y_observed_GLS,y_predicted_GLS))
        print(rm.MSE(y_observed_GLS,y_predicted_GLS))
        print(rm.MAE(y_observed_GLS,y_predicted_GLS))
        print(rm.MAPE(y_observed_GLS,y_predicted_GLS))
        print(rm.durbin_watson(residual_GLS))
        print(GLS_complete.summary(xname=cll.get_model_1_coefficient_names()))

        print("OLS: SSE,Rsquared,MSE,MAE,MAPE,durbinwatson")
        print(rm.SSE(y_observed_OLS,y_predicted_OLS))
        print(rm.Rsquared(y_observed_OLS,y_predicted_OLS))
        print(rm.MSE(y_observed_OLS,y_predicted_OLS))
        print(rm.MAE(y_observed_OLS,y_predicted_OLS))
        print(rm.MAPE(y_observed_OLS,y_predicted_OLS))
        print(rm.durbin_watson(residual_OLS))
        print(OLS_complete.summary(xname=cll.get_model_1_coefficient_names()))

       # print("OLS extended: SSE,Rsquared,MSE,MAE,MAPE,durbinwatson")
       # print(rm.SSE(y_predicted_OLS_extended,y_observed_OLS_extended))
       # print(rm.Rsquared(y_predicted_OLS_extended,y_observed_OLS_extended))
        #print(rm.MSE(y_predicted_OLS_extended,y_observed_OLS_extended))
        #print(rm.MAE(y_predicted_OLS_extended,y_observed_OLS_extended))
        #print(rm.MAPE(y_predicted_OLS_extended,y_observed_OLS_extended))
        #print(rm.durbin_watson(residual_OLS_extended))
        #print(OLS_extended.summary(xname=cll.get_model_2_coefficient_names()))

        
        mape_score = [rm.MAPE([y_predicted_OLS[i]],[y_observed_OLS[i]])*100 for i in range(len(y_predicted_OLS))]
        
        pp.plot_vs_date(df_knmi["DATUM_TIJDSTIP"].tolist(),y_predicted_OLS, xaxis= "Date", yaxis= "Electricity load[kW]",
                   png = "outlier_predicted_value_%s.png"%str(STATION_ID))
        pp.plot_vs_date(df_knmi["DATUM_TIJDSTIP"].tolist(),y_predicted_GLS, xaxis= "Date", yaxis= "Electricity load[kW]",
                   png = "outlier_predicted_valueGLS_%s.png"%str(STATION_ID))
        pp.plot_vs_date(df_knmi["DATUM_TIJDSTIP"].tolist(),mape_score, xaxis= "Date", yaxis= "Mape score[%]",
                   png = "outlier_mape_%s.png"%str(STATION_ID))
        pp.plot_vs_date(df_knmi["T"].tolist(),mape_score, xaxis= "Temp[Celcius]", yaxis= "Mape score[%]",
                   png = "outlier_mape_temp_%s.png"%str(STATION_ID))
        pp.plot_vs_date(df_knmi["FH"].tolist(),mape_score, xaxis= "Wind speed[m/s]", yaxis= "Mape score[%]",
                   png = "outlier_mape_wind_%s.png"%str(STATION_ID))
        pp.heat_map_scatterplot(y_observed_OLS, y_predicted_OLS, df_knmi["T"].tolist(), ranges_heat=[20, 10, 0, -5],
                            yaxis="Observed value[kW]", xaxis="Model Value[kW]",  align_ax=True,
                            png = "outlier_scatter_%s.png"%str(STATION_ID))


    
    
    return

def beta_coefficient_model_extreme():
    STATION_TYPE = "OS"
    STATION_ID = "OS WESTZAANSTRAAT 10-1i"
    train_year = []
    test_year = [2022]
    print("getting dataframes")
    df_knmi = dbk.get_dataframe_KNMI(years = test_year +train_year)  # Get KNMI df- this way we only have to call this functions one time
    #df_alliander, df_knmi = dba.get_dataframe_alliander(STATION_ID, STATION_TYPE,
    #                                                 df_knmi)  # Get df for the test station and id (deep copy for knmi dataframe)

    print("starting linear regression")
    print("model 1")
    #Model 1
    y_predicted_OLS, y_observed_OLS,X, date, df_knmi, OLS_complete = cll.get_model_1(df_knmi_train= df_knmi, STATION_ID = STATION_ID, STATION_TYPE = STATION_TYPE, train_year = train_year, test_year = test_year)

    pdf = PdfPages('%s.pdf' % STATION_ID)

    coeff_names = cll.get_model_1_coefficient_names()

    # Model 1 results
    pp.text_to_pdf(OLS_complete.summary(xname=coeff_names), pdf)
    pp.heat_map_scatterplot(y_observed_OLS, y_predicted_OLS, df_knmi["T"].tolist(), ranges_heat=[20, 10, 0, -5],
                            title="Observed value vs predicted value (model 1) - %s" % STATION_ID, xaxis="Observed value",
                            yaxis="Predicted Value", pdf=pdf, align_ax=True)
    print("model 2")
    #Model 2
    y_predicted_reduced, y_observed_reduced, X_reduced, date, df_knmi_critical, OLS_critical_reduced = cll.get_model_OLS_extended(df_knmi_train=df_knmi, STATION_ID=STATION_ID,
                                                                                  STATION_TYPE=STATION_TYPE, train_year = train_year, test_year = test_year)
    coeff_names_short_list = cll.get_model_2_coefficient_names()
    pp.text_to_pdf(OLS_critical_reduced.summary(xname=coeff_names_short_list), pdf)
    pp.heat_map_scatterplot(y_observed_reduced, y_predicted_reduced, df_knmi_critical["T"].tolist(), ranges_heat=[-2, -5, -8, -10],
                            title="Observed value vs predicted value (model 2) - %s" % STATION_ID, xaxis="Observed value",
                            yaxis="Predicted Value", pdf=pdf, align_ax=True)

    print("model 2 Adjusted")
    #Model 2
    y_predicted_reduced, y_observed_reduced, X_reduced, dates_reduced, df_knmi_critical, OLS_critical_reduced = cll.get_model_OLS_extended(df_knmi_train=df_knmi, STATION_ID=STATION_ID,
                                                                                  STATION_TYPE=STATION_TYPE, train_year = train_year, test_year = test_year)
    list_data_values = rm.regression_model_coefficient_date_values(OLS_complete.params,dates_reduced)
    new_y_observed_reduced = np.array(y_observed_reduced) - np.array(list_data_values)
    OLS_critical_reduced_without_date = rm.linear_regression_statsmodels(new_y_observed_reduced, X_reduced[:,:-1])
    OLS_critical_reduced_without_date.predict()
    coeff_names_short_list = cll.get_model_2_coefficient_names()[:-1]
    pp.text_to_pdf(OLS_critical_reduced_without_date.summary(xname=coeff_names_short_list), pdf)
    y_predicted_reduced = np.array(y_predicted_reduced) + np.array(list_data_values)
    pp.heat_map_scatterplot(y_observed_reduced, y_predicted_reduced, df_knmi_critical["T"].tolist(), ranges_heat=[-2, -5, -8, -10],
                            title="Observed value vs predicted value (model 2 with data coefficient removed before fit) - %s" % STATION_ID, xaxis="Observed value",
                            yaxis="Predicted Value", pdf=pdf, align_ax=True)
    print(rm.Rsquared(y_observed_reduced, y_predicted_reduced))
    params_temp = OLS_critical_reduced.params
    params_temp[-1] = 1
    predicted_values = rm.predict_values(X_reduced, params_temp)
    print(rm.Rsquared(y_observed_reduced,predicted_values))
    pp.heat_map_scatterplot(y_observed_reduced, predicted_values, df_knmi_critical["T"].tolist(), ranges_heat=[-2, -5, -8, -10],
                            title="Observed value vs predicted value (model 2 with parameter set to 1) - %s" % STATION_ID, xaxis="Observed value",
                            yaxis="Predicted Value", pdf=pdf, align_ax=True)
    pdf.close()
    return



def holidays(train_year = [2022], test_year = []):
    STATION_TYPE = "OS"
    STATION_ID = "OS WESTZAANSTRAAT 10-1i"
    #train_year = [2022]
    df_knmi = dbk.get_dataframe_KNMI(years = test_year +train_year)  
    print("starting linear regression")
    print("OLS")
    #Model 1
    y_predicted_OLS, y_observed_OLS,X, date, df_knmi, OLS_complete = cll.get_model_1(df_knmi_train= df_knmi, STATION_ID = STATION_ID, STATION_TYPE = STATION_TYPE, train_year = train_year, test_year = test_year)
    y_predicted_GLS, y_observed_GLS,X, date, df_knmi, GLS_complete = cll.get_model_3(df_knmi_train= df_knmi, STATION_ID = STATION_ID, STATION_TYPE = STATION_TYPE, train_year = train_year, test_year = test_year)
    MAPE_OLS = [rm.MAPE([y_predicted_OLS[i]],[y_observed_OLS[i]])*100 for i in range(len(y_predicted_OLS))]
    MAPE_GLS = [rm.MAPE([y_predicted_GLS[i]],[y_observed_GLS[i]])*100 for i in range(len(y_predicted_GLS))]
    MAE_OLS = [rm.MAE([y_predicted_OLS[i]],[y_observed_OLS[i]])*100 for i in range(len(y_predicted_OLS))]
    MAE_GLS = [rm.MAE([y_predicted_GLS[i]],[y_observed_GLS[i]])*100 for i in range(len(y_predicted_GLS))]
    print("MAPE PLOT")
    date1 = rm.get_date_month_day(date)
    date2 = rm.get_date_month_day_hour(date)
    pp.multiplot_vs_date(date, MAPE_OLS,MAPE_GLS,xaxis = "Date", yaxis1 = "MAPE score[%]", png = "holidays_MAPE_scores.png" ,
                         legend = ["OLS", "GLS"], show_plot = True)

    return


def diff_west_middenmeer():
    STATION_TYPE = "OS"
    STATION_IDS = ["OS WESTZAANSTRAAT 10-1i","OS MIDDENMEER 20-1i"]
    for STATION_ID in STATION_IDS:
        print(STATION_ID)
        train_year = [2022]
        test_year = [2021] 
        y_predicted_GLS, y_observed_GLS,X, date, df_knmi, GLS_complete = cll.get_model_3(STATION_ID = STATION_ID, STATION_TYPE = STATION_TYPE, train_year = train_year, test_year = test_year)
        y_predicted_OLS, y_observed_OLS,X, date, df_knmi, OLS_complete = cll.get_model_1(STATION_ID = STATION_ID, STATION_TYPE = STATION_TYPE, train_year = train_year, test_year = test_year)
        y_predicted_OLS_extended, y_observed_OLS_extended,X, date, df_knmi, OLS_extended,OLS_complete = cll.get_model_OLS_extended(STATION_ID = STATION_ID, STATION_TYPE = STATION_TYPE, train_year = train_year, test_year = test_year)


        residual_GLS = (rm.decimal_to_float(y_observed_GLS) - rm.decimal_to_float(y_predicted_GLS))
        residual_OLS = (rm.decimal_to_float(y_observed_OLS) - rm.decimal_to_float(y_predicted_OLS))
        residual_OLS_extended = (rm.decimal_to_float(y_observed_OLS_extended) - rm.decimal_to_float(y_predicted_OLS_extended))

        
        print("GLS: SSE,Rsquared,MSE,MAE,MAPE,durbinwatson")
        print(rm.SSE(y_observed_GLS,y_predicted_GLS))
        print(rm.Rsquared(y_observed_GLS,y_predicted_GLS))
        print(rm.MSE(y_observed_GLS,y_predicted_GLS))
        print(rm.MAE(y_observed_GLS,y_predicted_GLS))
        print(rm.MAPE(y_observed_GLS,y_predicted_GLS))
        print(rm.durbin_watson(residual_GLS))
        print(GLS_complete.summary(xname=cll.get_model_1_coefficient_names()))

        print("OLS: SSE,Rsquared,MSE,MAE,MAPE,durbinwatson")
        print(rm.SSE(y_observed_OLS,y_predicted_OLS))
        print(rm.Rsquared(y_observed_OLS,y_predicted_OLS))
        print(rm.MSE(y_observed_OLS,y_predicted_OLS))
        print(rm.MAE(y_observed_OLS,y_predicted_OLS))
        print(rm.MAPE(y_observed_OLS,y_predicted_OLS))
        print(rm.durbin_watson(residual_OLS))
        print(OLS_complete.summary(xname=cll.get_model_1_coefficient_names()))

        print("OLS extended: SSE,Rsquared,MSE,MAE,MAPE,durbinwatson")
        print(rm.SSE(y_predicted_OLS_extended,y_observed_OLS_extended))
        print(rm.Rsquared(y_predicted_OLS_extended,y_observed_OLS_extended))
        print(rm.MSE(y_predicted_OLS_extended,y_observed_OLS_extended))
        print(rm.MAE(y_predicted_OLS_extended,y_observed_OLS_extended))
        print(rm.MAPE(y_predicted_OLS_extended,y_observed_OLS_extended))
        print(rm.durbin_watson(residual_OLS_extended))
        print(OLS_extended.summary(xname=cll.get_model_2_coefficient_names()))
    return

def week_vs_weekend():
    STATION_TYPE = "OS"
    STATION_ID = "OS WESTZAANSTRAAT 10-1i"
    train_year = [2022]
    test_year = []
    df_knmi = dbk.get_dataframe_KNMI(years = test_year +train_year)  
    legend = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    #days = ["2022-07-04","2022-07-05","2022-07-06","2022-07-07", "2022-07-08","2022-07-09","2022-07-10"]
    days = ["2022-11-07","2022-11-08","2022-11-09","2022-11-10", "2022-11-11","2022-11-12","2022-11-13"]
    df_knmi = df_knmi[df_knmi['DATUM_TIJDSTIP'].astype(str).str.contains("|".join(days))]
    df_alliander,df_knmi = dba.get_dataframe_alliander(STATION_ID,STATION_TYPE,df_knmi, 2022)
    y_values = [df_alliander[df_alliander['DATUM_TIJDSTIP'].astype(str).str.contains(days[i])]["BELASTING"].tolist() for i in range(len(days))]
    date = [str(i) for i in range(1,25)]
    pp.multiple_line_plots(date,y_values,legends = legend,  xaxis = "Hours",title = "Weekday vs Weekenday - OS WESTZAANSTRAAT 10-1i",
                        yaxis = "Electricity load[kW]",png = "week_weekend_compared.png")


    df_knmi = dbk.get_dataframe_KNMI(years = test_year +train_year)  
    y_predicted_GLS, y_observed_GLS,X, date, df_knmi, GLS_complete = cll.get_model_3(df_knmi_train= df_knmi, STATION_ID = STATION_ID, STATION_TYPE = STATION_TYPE, train_year = train_year, test_year = test_year)
    y_predicted_GLS_old, y_observed_GLS,X, date, df_knmi, GLS_complete = cll.get_model_3_temp(df_knmi_train= df_knmi, STATION_ID = STATION_ID, STATION_TYPE = STATION_TYPE, train_year = train_year, test_year = test_year)
    MAPE_GLS = [rm.MAPE([y_predicted_GLS[i]],[y_observed_GLS[i]])*100 for i in range(len(y_predicted_GLS))]
    MAPE_GLS_OLD = [rm.MAPE([y_predicted_GLS_old[i]],[y_observed_GLS[i]])*100 for i in range(len(y_predicted_GLS))]
    df_knmi["MAPE_GLS"]  = MAPE_GLS
    df_knmi["MAPE_GLS_OLD"]  = MAPE_GLS_OLD
    df_knmi["OBSERVED"] = y_observed_GLS
    df_knmi_week = df_knmi[df_knmi['DATUM_TIJDSTIP'].astype(str).str.contains("|".join(days[0:5]))]
    df_knmi_weekend = df_knmi[df_knmi['DATUM_TIJDSTIP'].astype(str).str.contains("|".join(days[5:7]))]

    
    
    #pp.multiplot_vs_date(rm.get_day_of_week(df_knmi_week["DATUM_TIJDSTIP"]), df_knmi_week["MAPE_GLS"],df_knmi_week["MAPE_GLS_OLD"],
    pp.multiplot_vs_date(df_knmi_week["DATUM_TIJDSTIP"], df_knmi_week["MAPE_GLS"],df_knmi_week["MAPE_GLS_OLD"],
                         xaxis = "Day of week", yaxis1 = "MAPE score[%]", png = "MAPE_weekday.png",
                         legend = ["new model", "old model"])
    
    #pp.multiplot_vs_date(rm.get_day_of_week(df_knmi_weekend["DATUM_TIJDSTIP"]), df_knmi_weekend["MAPE_GLS"],df_knmi_weekend["MAPE_GLS_OLD"],
    pp.multiplot_vs_date(df_knmi_weekend["DATUM_TIJDSTIP"], df_knmi_weekend["MAPE_GLS"],df_knmi_weekend["MAPE_GLS_OLD"],
                         xaxis = "Day of week", yaxis1 = "MAPE score[%]", png = "MAPE_weekendday.png",
                         legend = ["new model", "old model"])
    return

def week_vs_weekend():
    STATION_TYPE = "OS"
    STATION_ID = "OS WESTZAANSTRAAT 10-1i"
    train_year = [2022]
    test_year = []
    df_knmi = dbk.get_dataframe_KNMI(years = test_year +train_year)  
    legend = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    #days = ["2022-07-04","2022-07-05","2022-07-06","2022-07-07", "2022-07-08","2022-07-09","2022-07-10"]
    days = ["2022-11-07","2022-11-08","2022-11-09","2022-11-10", "2022-11-11","2022-11-12","2022-11-13"]
    df_knmi = df_knmi[df_knmi['DATUM_TIJDSTIP'].astype(str).str.contains("|".join(days))]
    df_alliander,df_knmi = dba.get_dataframe_alliander(STATION_ID,STATION_TYPE,df_knmi, 2022)
    y_values = [df_alliander[df_alliander['DATUM_TIJDSTIP'].astype(str).str.contains(days[i])]["BELASTING"].tolist() for i in range(len(days))]
    date = [str(i) for i in range(1,25)]
    pp.multiple_line_plots(date,y_values,legends = legend,  xaxis = "Hours",title = "Weekday vs Weekenday - OS WESTZAANSTRAAT 10-1i",
                        yaxis = "Electricity load[kW]",png = "week_weekend_compared.png")


    df_knmi = dbk.get_dataframe_KNMI(years = test_year +train_year)  
    y_predicted_GLS, y_observed_GLS,X, date, df_knmi, GLS_complete = cll.get_model_3(df_knmi_train= df_knmi, STATION_ID = STATION_ID, STATION_TYPE = STATION_TYPE, train_year = train_year, test_year = test_year)
    y_predicted_GLS_old, y_observed_GLS,X, date, df_knmi, GLS_complete = cll.get_model_3_temp(df_knmi_train= df_knmi, STATION_ID = STATION_ID, STATION_TYPE = STATION_TYPE, train_year = train_year, test_year = test_year)
    MAPE_GLS = [rm.MAPE([y_predicted_GLS[i]],[y_observed_GLS[i]])*100 for i in range(len(y_predicted_GLS))]
    MAPE_GLS_OLD = [rm.MAPE([y_predicted_GLS_old[i]],[y_observed_GLS[i]])*100 for i in range(len(y_predicted_GLS))]
    df_knmi["MAPE_GLS"]  = MAPE_GLS
    df_knmi["MAPE_GLS_OLD"]  = MAPE_GLS_OLD
    df_knmi["OBSERVED"] = y_observed_GLS
    df_knmi_week = df_knmi[df_knmi['DATUM_TIJDSTIP'].astype(str).str.contains("|".join(days[0:5]))]
    df_knmi_weekend = df_knmi[df_knmi['DATUM_TIJDSTIP'].astype(str).str.contains("|".join(days[5:7]))]

    
    
    #pp.multiplot_vs_date(rm.get_day_of_week(df_knmi_week["DATUM_TIJDSTIP"]), df_knmi_week["MAPE_GLS"],df_knmi_week["MAPE_GLS_OLD"],
    pp.multiplot_vs_date(df_knmi_week["DATUM_TIJDSTIP"], df_knmi_week["MAPE_GLS"],df_knmi_week["MAPE_GLS_OLD"],
                         xaxis = "Day of week", yaxis1 = "MAPE score[%]", png = "MAPE_weekday.png",
                         legend = ["new model", "old model"])
    
    #pp.multiplot_vs_date(rm.get_day_of_week(df_knmi_weekend["DATUM_TIJDSTIP"]), df_knmi_weekend["MAPE_GLS"],df_knmi_weekend["MAPE_GLS_OLD"],
    pp.multiplot_vs_date(df_knmi_weekend["DATUM_TIJDSTIP"], df_knmi_weekend["MAPE_GLS"],df_knmi_weekend["MAPE_GLS_OLD"],
                         xaxis = "Day of week", yaxis1 = "MAPE score[%]", png = "MAPE_weekendday.png",
                         legend = ["new model", "old model"])
    return

def plots():
    STATION_TYPE = "OS"
    STATION_ID = "OS WESTZAANSTRAAT 10-1i"
    train_year = [2022]
    test_year = [2021] 
    y_predicted_OLS, y_observed_OLS,X, date, df_knmi, OLS_complete = cll.get_model_1(STATION_ID = STATION_ID, STATION_TYPE = STATION_TYPE, train_year = train_year, test_year = test_year)
    df_knmi["LOAD"] = y_observed_OLS

    df_knmi_low = df_knmi[df_knmi["T"] <= 0]
    df_knmi_high = df_knmi[df_knmi["T"] >= 0]
    pp.scatterplot_temp(df_knmi_low["T"].tolist(),df_knmi_low["LOAD"].tolist(),df_knmi_high["T"].tolist(),df_knmi_high["LOAD"].tolist(), xaxis= "Temperature[Celcius]", yaxis= "Electricity load[kW]", align_ax = False ,
                   png = "extreme_temp.png")
    pp.scatterplot(df_knmi["T"].tolist(),y_observed_OLS, xaxis= "Temperature[Celcius]", yaxis= "Electricity load[kW]", align_ax = False ,
                   png = "temp_vs_load.png")

    return

def results_per_model2():
    STATION_TYPE = "OS"
    STATION_ID = "OS WESTZAANSTRAAT 10-1i"
    train_year = [2021]
    test_year = [2022] 
    y_predicted_GLS, y_observed_GLS,X, date, df_knmi, GLS_complete = cll.get_model_3(STATION_ID = STATION_ID, STATION_TYPE = STATION_TYPE, train_year = train_year, test_year = test_year)
    y_predicted_OLS, y_observed_OLS,X, date, df_knmi, OLS_complete = cll.get_model_1(STATION_ID = STATION_ID, STATION_TYPE = STATION_TYPE, train_year = train_year, test_year = test_year)
    y_predicted_OLS_extended, y_observed_OLS_extended,X, date, df_knmi, OLS_extended,OLS_complete = cll.get_model_OLS_extended(STATION_ID = STATION_ID, STATION_TYPE = STATION_TYPE, train_year = train_year, test_year = test_year)
    
    residual_GLS = (rm.decimal_to_float(y_observed_GLS) - rm.decimal_to_float(y_predicted_GLS))
    residual_OLS = (rm.decimal_to_float(y_observed_OLS) - rm.decimal_to_float(y_predicted_OLS))
    residual_OLS_extended = (rm.decimal_to_float(y_observed_OLS_extended) - rm.decimal_to_float(y_predicted_OLS_extended))

    
    print("GLS: SSE,Rsquared,MSE,MAE,MAPE,durbinwatson")
    print(rm.SSE(y_observed_GLS,y_predicted_GLS))
    print(rm.Rsquared(y_observed_GLS,y_predicted_GLS))
    print(rm.MSE(y_observed_GLS,y_predicted_GLS))
    print(rm.MAE(y_observed_GLS,y_predicted_GLS))
    print(rm.MAPE(y_observed_GLS,y_predicted_GLS))
    print(rm.durbin_watson(residual_GLS))
    print(GLS_complete.summary(xname=cll.get_model_1_coefficient_names()))

    print("OLS: SSE,Rsquared,MSE,MAE,MAPE,durbinwatson")
    print(rm.SSE(y_observed_OLS,y_predicted_OLS))
    print(rm.Rsquared(y_observed_OLS,y_predicted_OLS))
    print(rm.MSE(y_observed_OLS,y_predicted_OLS))
    print(rm.MAE(y_observed_OLS,y_predicted_OLS))
    print(rm.MAPE(y_observed_OLS,y_predicted_OLS))
    print(rm.durbin_watson(residual_OLS))
    print(OLS_complete.summary(xname=cll.get_model_1_coefficient_names()))

    print("OLS extended: SSE,Rsquared,MSE,MAE,MAPE,durbinwatson")
    print(rm.SSE(y_predicted_OLS_extended,y_observed_OLS_extended))
    print(rm.Rsquared(y_predicted_OLS_extended,y_observed_OLS_extended))
    print(rm.MSE(y_predicted_OLS_extended,y_observed_OLS_extended))
    print(rm.MAE(y_predicted_OLS_extended,y_observed_OLS_extended))
    print(rm.MAPE(y_predicted_OLS_extended,y_observed_OLS_extended))
    print(rm.durbin_watson(residual_OLS_extended))
    print(OLS_extended.summary(xname=cll.get_model_2_coefficient_names()))
    
    pp.scatterplot(y_predicted_GLS,y_observed_GLS, xaxis= "Model values[kW]", yaxis= "Actual values[kW]", align_ax = True ,
                   png = "GLS_scatter_1b.png", color = "blue")
    pp.scatterplot(y_observed_GLS,y_predicted_GLS, xaxis= "Actual values[kW]", yaxis= "Model values[kW]", align_ax = True ,
                   png = "GLS_scatter_2b.png", color = "blue")
    pp.scatterplot(y_predicted_OLS,y_observed_OLS, xaxis= "Model values[kW]", yaxis= "Actual values[kW]", align_ax = True ,
                   png = "OLS_scatter_1b.png", color = "red")
    pp.scatterplot(y_observed_OLS,y_predicted_OLS, xaxis= "Actual values[kW]", yaxis= "Model values[kW]", align_ax = True ,
                   png = "OLS_scatter_2b.png", color = "red")
    pp.scatterplot(y_predicted_OLS_extended,y_observed_OLS_extended, xaxis= "Model values[kW]", yaxis= "Actual values[kW]", align_ax = True ,
                   png = "OLS_scatter_extended_1b.png", color = "red")
    pp.scatterplot(y_observed_OLS_extended,y_predicted_OLS_extended, xaxis= "Actual values[kW]", yaxis= "Model values[kW]", align_ax = True ,
                   png = "OLS_scatter_extended_2b.png", color = "red")

    return

def making_predictions():
    STATION_TYPE = "OS"
    STATION_ID = "OS WESTZAANSTRAAT 10-1i"
    train_year = [2022]
    test_year = []
    df_knmi = dbk.get_dataframe_KNMI(years = test_year +train_year)
    y_predicted_OLS_extended, y_observed_OLS_extended, X_reduced, date_reduced, df_knmi_critical, OLS_extended, OLS_complete = cll.get_model_OLS_extended(df_knmi_train=df_knmi, STATION_ID=STATION_ID,
                                                                                  STATION_TYPE=STATION_TYPE, train_year = train_year, test_year = test_year)
    y_predicted_GLS_extended, y_observed_GLS_extended, X_reduced, date_reduced, df_knmi_critical, GLS_extended, GLS_complete = cll.get_model_GLS_extended(df_knmi_train=df_knmi, STATION_ID=STATION_ID,
                                                                                  STATION_TYPE=STATION_TYPE, train_year = train_year, test_year = test_year)

    
   # print(OLS_complete.summary(xname=cll.get_model_1_coefficient_names()))
   # print(GLS_complete.summary(xname=cll.get_model_1_coefficient_names()))
   # print(OLS_extended.summary(xname=cll.get_model_2_coefficient_names()))
    #print(GLS_extended.summary(xname=cll.get_model_2_coefficient_names()))

    extrapolated_temp = [int("%s" %i) for i in range(-15,1)]
   # print(extrapolated_temp)
    OLS_date = "2022-11-03 12:00"
    GLS_date = "2022-12-08 12:00"
    df_alliander, df_knmi = dba.get_dataframe_alliander(STATION_ID,STATION_TYPE, df_knmi,2022)
    OLS_wind = df_knmi[df_knmi["DATUM_TIJDSTIP"] == OLS_date]["FH"].tolist()[0]
    OLS_solar = df_knmi[df_knmi["DATUM_TIJDSTIP"] == OLS_date]["Q"].tolist()[0]

    GLS_wind = df_knmi[df_knmi["DATUM_TIJDSTIP"] == GLS_date]["FH"].tolist()[0]
    GLS_solar = df_knmi[df_knmi["DATUM_TIJDSTIP"] == GLS_date]["Q"].tolist()[0]

    X_OLS = [[1,float(temp),float(OLS_wind),float(OLS_solar),rm.regression_model_coefficient_date_values(OLS_complete.params,[OLS_date])[0]] for temp in extrapolated_temp]
    X_GLS = [[1,float(temp),float(GLS_wind),float(GLS_solar),rm.regression_model_coefficient_date_values(GLS_complete.params,[GLS_date])[0]] for temp in extrapolated_temp]
 
    OLS_predicted = [rm.predict_values(X_OLS,OLS_extended.params)]
    GLS_predicted = [rm.predict_values(X_GLS,GLS_extended.params)]

    print(OLS_predicted)
    print(GLS_predicted)
    
    return



def temp_plots():
    STATION_TYPE = "OS"
    STATION_ID = "OS WESTZAANSTRAAT 10-1i"
    train_year = [2022]
    test_year = []
    y_predicted, y_observed,X, date, df_knmi,OLS_complete_Westzaanstraat = cll.get_model_1(train_year = train_year, test_year = test_year,
                                                                                                 STATION_ID = STATION_ID, STATION_TYPE = STATION_TYPE)
    print(rm.Rsquared(y_observed, y_predicted))

    y_predicted, y_observed,X, date, df_knmi,GLS_complete = cll.get_model_3(train_year = train_year, test_year = test_year,
                                                                                                 STATION_ID = STATION_ID, STATION_TYPE = STATION_TYPE)

    
    print(rm.Rsquared(y_observed, y_predicted))
    return

def temp_plots2():
    STATION_TYPE = "OS"
    STATION_ID = "OS WESTZAANSTRAAT 10-1i"
    train_year = [2022]
    test_year = [2021]
    y_predicted, y_observed,X, date, df_knmi,OLS_complete_Westzaanstraat = cll.get_model_1(train_year = train_year, test_year = test_year,
                                                                                                 STATION_ID = STATION_ID, STATION_TYPE = STATION_TYPE)
    print(rm.Rsquared(y_observed, y_predicted))

    y_predicted, y_observed,X, date, df_knmi,GLS_complete = cll.get_model_3(train_year = train_year, test_year = test_year,
                                                                                                 STATION_ID = STATION_ID, STATION_TYPE = STATION_TYPE)

    
    print(rm.Rsquared(y_observed, y_predicted))
    return
temp_plots()
temp_plots2()
