import Regression_Model as rm
import Database_Alliander as dba
import Database_KNMI as dbk
import Link_Weather_Station as lws
import pandas as pd
#Using the ols model to make predictions
#Two options, one with test data and one without
#If no test_year is given then the train_year is used for predictions on itself
#Giving a knmi/alliander df is optional (this is to avoid having to call df's several times)
#OLS
def get_model_1(df_knmi_train = pd.DataFrame({'A' : []}), df_knmi_test = pd.DataFrame({'A' : []}), train_year = [2021], test_year = [], STATION_ID = "", STATION_TYPE = ""): #OLS model
    #pd.set_option('display.max_columns', None) #display options for dataframe
    if df_knmi_train.empty:  #No train dataframe is given
        df_knmi_train = dbk.get_dataframe_KNMI(years = train_year) #Get knmi df
    df_alliander_train, df_knmi_train = dba.get_dataframe_alliander(STATION_ID, STATION_TYPE,
                                                           df_knmi_train, year = train_year[0])  # Get df for the test station and id (deep copy for knmi dataframe)
    if test_year != []: #We want a different test set than train set
        if df_knmi_test.empty:
            df_knmi_test = dbk.get_dataframe_KNMI(years = test_year)
        df_alliander_test, df_knmi_test = dba.get_dataframe_alliander(STATION_ID, STATION_TYPE,
                                                           df_knmi_test, year = test_year[0])
    y_observed_train, X_train, date_train = rm.linear_regression_model_data_controller(df_alliander_train, df_knmi_train) #Get training data
    OLS_complete = rm.linear_regression_statsmodels(y_observed_train, X_train) #Use linear regression to train model using OLS
    if test_year != []:
        y_observed_test, X_test, date_test = rm.linear_regression_model_data_controller(df_alliander_test, df_knmi_test) #Get testing data
        y_predicted = OLS_complete.predict(X_test)  #Use model on testing data
        return rm.decimal_to_float(y_predicted), y_observed_test,X_test, date_test, df_knmi_test, OLS_complete #return the data
    y_predicted = OLS_complete.predict(X_train) #Predict training data
    return y_predicted, y_observed_train,X_train, date_train, df_knmi_train, OLS_complete #Return training data (doesnt need else statement here)


#Using the ols model to make predictions on the critical region (when temp is below 0)
#The parameter results from the first model is used for the date on the second model
#Two options, one with test data and one without
#If no test_year is given then the train_year is used for predictions on itself
#Giving a knmi/alliander df is optional (this is to avoid having to call df's several times)
def get_model_OLS_extended(df_knmi_train = pd.DataFrame({'A' : []}), df_knmi_test = pd.DataFrame({'A' : []}), train_year = [2021], test_year = [ ], STATION_ID = "", STATION_TYPE = ""): #OLS Model critical region
    if df_knmi_train.empty:
        df_knmi_train = dbk.get_dataframe_KNMI(years = train_year) #Get knmi df
    df_alliander_train, df_knmi_train = dba.get_dataframe_alliander(STATION_ID, STATION_TYPE,
                                                           df_knmi_train, year = train_year[0])  # Get df for the test station and id (deep copy for knmi dataframe)
    if test_year != []:
        df_knmi_test = dbk.get_dataframe_KNMI(years = test_year)
        df_alliander_test, df_knmi_test = dba.get_dataframe_alliander(STATION_ID, STATION_TYPE,
                                                           df_knmi_train, year = test_year[0])
        df_knmi_test_critical = dbk.get_critical_region(df_knmi_test)
        df_alliander_test_critical, df_knmi_test_critical = lws.same_sizing(df_alliander_test, df_knmi_test_critical)

    y_observed_train, X_train, date_train = rm.linear_regression_model_data_controller(df_alliander_train, df_knmi_train) #Get training data
    OLS_complete = rm.linear_regression_statsmodels(y_observed_train, X_train) #Use linear regression to train model using OLS
    df_knmi_train_critical = dbk.get_critical_region(df_knmi_train)

    df_alliander_train_critical, df_knmi_train_critical = lws.same_sizing(df_alliander_train, df_knmi_train_critical)

    y_observed_train, X_train, date_train = rm.linear_regression_model_data_controller_reduced(df_alliander_train_critical, df_knmi_train_critical,
                                                                             OLS_complete.params)
    OLS_critical_reduced = rm.linear_regression_statsmodels(y_observed_train, X_train)

    if test_year != []:
        y_observed_test, X_test, date_test = rm.linear_regression_model_data_controller_reduced(
            df_alliander_test_critical, df_knmi_test_critical, OLS_complete.params) #Get testing data
        y_predicted = OLS_critical_reduced.predict(X_test)  #Use model on testing data
        return rm.decimal_to_float(y_predicted), rm.decimal_to_float(y_observed_test),X_test, date_test, df_knmi_test_critical, OLS_critical_reduced,OLS_complete #return the data
    y_predicted = OLS_critical_reduced.predict(X_train) #Predict training data
    return rm.decimal_to_float(y_predicted), rm.decimal_to_float(y_observed_train),X_train, date_train, df_knmi_train_critical, OLS_critical_reduced,OLS_complete #Return training data (doesnt need else statement here)

#GLS
def get_model_3(df_knmi_train = pd.DataFrame({'A' : []}), df_knmi_test = pd.DataFrame({'A' : []}),  train_year = [2021], test_year = [ ], STATION_ID = "", STATION_TYPE = ""): #GLS model
    if df_knmi_train.empty:
        df_knmi_train = dbk.get_dataframe_KNMI(years=train_year)  # Get knmi df
    df_alliander_train, df_knmi_train = dba.get_dataframe_alliander(STATION_ID, STATION_TYPE,
                                                                        df_knmi_train,
                                                                        year=train_year[0])  # Get df for the test station and id (deep copy for knmi dataframe)
    if test_year != []:
        if df_knmi_test.empty:
            df_knmi_test = dbk.get_dataframe_KNMI(years=test_year)
        df_alliander_test, df_knmi_test = dba.get_dataframe_alliander(STATION_ID, STATION_TYPE,
                                                                      df_knmi_test, year=test_year[0])

    y_observed_train, X_train, date_train = rm.linear_regression_model_data_controller(df_alliander_train,
                                                                                       df_knmi_train)  # Get training data

    GLS_complete = rm.linear_regression_GLS(y_observed_train,
                                                    X_train)  # Use linear regression to train model using OLS
    if test_year != []:
        y_observed_test, X_test, date_test = rm.linear_regression_model_data_controller(df_alliander_test,
                                                                                        df_knmi_test)  # Get testing data
        y_predicted = GLS_complete.predict(X_test)  # Use model on testing data
        return y_predicted, y_observed_test,X_test, date_test, df_knmi_test, GLS_complete  # return the data
    y_predicted = GLS_complete.predict(X_train)  # Predict training data
    return y_predicted, y_observed_train,X_train, date_train, df_knmi_train, GLS_complete  # Return training data (doesn't need else statement here)

def get_model_GLS_extended(df_knmi_train = pd.DataFrame({'A' : []}), df_knmi_test = pd.DataFrame({'A' : []}), train_year = [2021], test_year = [ ], STATION_ID = "", STATION_TYPE = ""): #OLS Model critical region
    if df_knmi_train.empty:
        df_knmi_train = dbk.get_dataframe_KNMI(years = train_year) #Get knmi df
    df_alliander_train, df_knmi_train = dba.get_dataframe_alliander(STATION_ID, STATION_TYPE,
                                                           df_knmi_train, year = train_year[0])  # Get df for the test station and id (deep copy for knmi dataframe)
    if test_year != []:
        df_knmi_test = dbk.get_dataframe_KNMI(years = test_year)
        df_alliander_test, df_knmi_test = dba.get_dataframe_alliander(STATION_ID, STATION_TYPE,
                                                           df_knmi_train, year = test_year[0])
        df_knmi_test_critical = dbk.get_critical_region(df_knmi_test)
        df_alliander_test_critical, df_knmi_test_critical = lws.same_sizing(df_alliander_test, df_knmi_test_critical)

    y_observed_train, X_train, date_train = rm.linear_regression_model_data_controller(df_alliander_train, df_knmi_train) #Get training data
    GLS_model = rm.linear_regression_GLS(y_observed_train, X_train) #Use linear regression to train model using OLS

    df_knmi_train_critical = dbk.get_critical_region(df_knmi_train)

    df_alliander_train_critical, df_knmi_train_critical = lws.same_sizing(df_alliander_train, df_knmi_train_critical)

    y_observed_train, X_train, date_train = rm.linear_regression_model_data_controller_reduced(df_alliander_train_critical, df_knmi_train_critical,
                                                                             GLS_model.params)
    GLS_model_reduced = rm.linear_regression_GLS(rm.decimal_to_float(y_observed_train), X_train)

    if test_year != []:
        y_observed_test, X_test, date_test = rm.linear_regression_model_data_controller_reduced(
            df_alliander_test_critical, df_knmi_test_critical, GLS_model.params) #Get testing data
        y_predicted = GLS_model_reduced.predict(X_test)  #Use model on testing data
        return rm.decimal_to_float(y_predicted), rm.decimal_to_float(y_observed_test),X_test, date_test, df_knmi_test, GLS_model_reduced, GLS_model #return the data
    y_predicted = GLS_model_reduced.predict(X_train) #Predict training data
    return rm.decimal_to_float(y_predicted), rm.decimal_to_float(y_observed_train),X_train, date_train, df_knmi_train, GLS_model_reduced, GLS_model#Return training data (doesnt need else statement here)





#Returns all coefficients for the OLS regression model as a list
def get_model_1_coefficient_names():
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    coeff_names = ["Intercept", "HDD", "CDD", "Sun", "Wind"] + months + days + \
                  [("Uur weekday " + str(i)) for i in range(1, 25)] + [("Uur weekendday " + str(i)) for i in
                                                                 range(1, 25)]
    return coeff_names

#Returns all coefficients for the extended regression model as a list
def get_model_2_coefficient_names():
    return ["Intercept", "CDD", "Sun", "Wind", "Date"]

#Returns all coefficients for the GLS regression model as a list
def get_model_3_coefficient_names():
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    coeff_names = ["Intercept", "HDD", "CDD", "Sun", "Wind"] + months + days + \
                  [("Uur weekday " + str(i)) for i in range(1, 25)] + [("Uur weekendday " + str(i)) for i in
                                                                 range(1, 25)]
    return coeff_names


#Special case that returns a different output
def get_model_1_special_case(df_knmi_train = pd.DataFrame({'A' : []}), df_knmi_test = pd.DataFrame({'A' : []}), train_year = [2021], test_year = [], STATION_ID = "", STATION_TYPE = ""): #OLS model
    pd.set_option('display.max_columns', None)
    if df_knmi_train.empty:
        df_knmi_train = dbk.get_dataframe_KNMI(years = train_year) #Get knmi df)
    df_alliander_train, df_knmi_train = dba.get_dataframe_alliander(STATION_ID, STATION_TYPE,
                                                           df_knmi_train, year = train_year[0])  # Get df for the test station and id (deep copy for knmi dataframe)
    if test_year != []:
        if df_knmi_test.empty:
            df_knmi_test = dbk.get_dataframe_KNMI(years = test_year)
        df_alliander_test, df_knmi_test = dba.get_dataframe_alliander(STATION_ID, STATION_TYPE,
                                                           df_knmi_test, year = test_year[0])
    y_observed_train, X_train, date_train = rm.linear_regression_model_data_controller(df_alliander_train, df_knmi_train) #Get training data

    OLS_complete = rm.linear_regression_statsmodels(y_observed_train, X_train) #Use linear regression to train model using OLS

    y_observed_test, X_test, date_test = rm.linear_regression_model_data_controller(df_alliander_test, df_knmi_test) #Get testing data
    y_predicted_test = OLS_complete.predict(X_test)  #Use model on testing data
    y_predicted_train = OLS_complete.predict(X_train) #Predict training data
    
    return y_predicted_train, y_predicted_test, y_observed_train, y_observed_test, OLS_complete

#Special case that returns a different output
def get_model_3_special_case(df_knmi_train = pd.DataFrame({'A' : []}), df_knmi_test = pd.DataFrame({'A' : []}),  train_year = [2021], test_year = [ ], STATION_ID = "", STATION_TYPE = ""): #GLS model
    if df_knmi_train.empty:
        df_knmi_train = dbk.get_dataframe_KNMI(years=train_year)  # Get knmi df
    df_alliander_train, df_knmi_train = dba.get_dataframe_alliander(STATION_ID, STATION_TYPE,
                                                                        df_knmi_train,
                                                                        year=train_year[0])  # Get df for the test station and id (deep copy for knmi dataframe)
    if test_year != []:
        if df_knmi_test.empty:
            df_knmi_test = dbk.get_dataframe_KNMI(years=test_year)
        df_alliander_test, df_knmi_test = dba.get_dataframe_alliander(STATION_ID, STATION_TYPE,
                                                                      df_knmi_test, year=test_year[0])

    y_observed_train, X_train, date_train = rm.linear_regression_model_data_controller(df_alliander_train,
                                                                                       df_knmi_train)  # Get training data

    GLS_complete = rm.linear_regression_GLS(y_observed_train,
                                                    X_train)  # Use linear regression to train model using OLS
    y_observed_test, X_test, date_test = rm.linear_regression_model_data_controller(df_alliander_test,
                                                                                    df_knmi_test)  # Get testing data
    y_predicted_test = GLS_complete.predict(X_test)  # Use model on testing data
    y_predicted_train = GLS_complete.predict(X_train)  # Predict training data
    return y_predicted_train, y_predicted_test, y_observed_train, y_observed_test, GLS_complete

