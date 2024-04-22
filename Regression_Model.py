
import datetime
import calendar

import math
import numpy as np
import pandas as pd
import holidays
import statsmodels.api as sm
from scipy.linalg import toeplitz
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Given:
# x = [x1,x2,x3,..,xt] vector
# Z = [z1, z2, z3,..., zt] matrix of vectors
# zi = [zi1,zi2,...,zik] vector
# returns the function that approximates x
# (alternative?) gives betas that can approx x where
# xt = b1*zt1 + b2*zt2 + ... + bk*ztk

# Simple linear regression model using sklearn
# Given a matrix Z [[],...,[]] and an array x [,...,]
# calculates the coefficients beta [,...,] s.t. the R2 score is maximized
# Returns the prediction on the same Z (Could differentiate between test and train sets - but this is for later)


# N = sample size (number of entries in database)
# p = number of columns + 1 (number of coefficients + 1)
# y_hat are the predicted values
# y_values are the actual values
# X_intercept is te standard intercept from sklearn package
def linear_regression_statsmodels(y_values, X):
    y_values = np.array([float(y_value) for y_value in y_values])
    return sm.OLS(y_values, X).fit()

#Given a simple linear regression of the form Y = a + bX
#returns a,b
def simple_linear_regression(Y,X):
    mean_y = np.mean(Y)
    mean_x = np.mean(X)
    numenator = sum([(X[i]-mean_x)*(Y[i]-mean_y) for i in range(len(X))])
    denominator = sum([(x-mean_x)**2 for x in X])
    b = numenator/denominator
    a = mean_y - b*mean_x
    return a,b

#Given simple linear regression of the form Y = a + bX where b is known
#returns a
def simple_linear_regression_b_known(Y,X,b):
    return np.mean(Y)-b*np.mean(X)

def linear_regression_GLS(y_values,X):
    ols_resid = linear_regression_statsmodels(y_values, X).resid
    resid_fit = sm.OLS(np.asarray(ols_resid)[1:],sm.add_constant(np.asarray(ols_resid)[:-1])).fit()
    rho = resid_fit.params[1]
    order = toeplitz(range(len(ols_resid)))
    sigma = rho ** order
    gls_model = sm.GLS(y_values, X, sigma= sigma)
    return gls_model.fit()


def SSE(y_observed,y_hat):
    total_error = 0
    for i in range(len(y_observed)):
        total_error += (y_observed[i]-y_hat[i])**2
    return total_error

def SSE1(y):
    avg = np.mean(y)
    total_error = 0
    for i in range(len(y)):
        total_error += (y[i]-avg)**2
    return total_error

def Rsquared(y_observed,y_hat):
    return (SSE1(y_observed) - SSE(y_observed,y_hat))/SSE1(y_observed)

def MSE(y_observed, y_hat):
    return mean_squared_error(y_observed, y_hat)

def MAE(y_observed, y_hat):
    return mean_absolute_error(y_observed, y_hat)

def MAPE(y_observed, y_hat):
    return mean_absolute_percentage_error(y_observed, y_hat)



#Returns the Pearsons correlation coefficient between two vectors x and y
def pearson_correlation_coeff(X,Y):
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    denominator_x = sum([(x-mean_x)**2 for x in X])
    denominator_y = sum([(y-mean_y)**2 for y in Y])
    denominator = (denominator_x)**0.5*(denominator_y)**0.5
    numenator = 0
    for i in range(len(X)):
        numenator += (X[i]-mean_x)*(Y[i] - mean_y)
    return numenator/denominator

#returns dubin watson test
def durbin_watson(eps):
    d = sum([(eps[i]-eps[i-1])**2 for i in range(1, len(eps))])
    d /= sum([(eps[i])**2 for i in range(len(eps))])
    return d

def predict_values(X, parameters):
    y = np.zeros(len(X))
    for i in range(len(X)):
        for j in X[i]:
            y[i] = np.inner(X[i],parameters)
    return np.array(y)

#Type cast a list of decimals (or any other format) to a list of floats, in a np array
#Using vectorization for faster performance
def decimal_to_float(lst):
    def dec_to_float(l):
        return float(l)
    return np.array(list(map(dec_to_float,lst)))

# Given a list of dates of the form YYYY-MM-DD HH:MM
# returns a list of lists of each month
# [[1,1,1,1,...,1,0,0,....,0],[0,0,...,0,1,...,1,0,...,0],...,[0,0,...,0,1,...,1]]
# so it has length 12xlen(dates) (12 months 365 days) january has 1 when it is january and 0 otherwise
def months_binary(dates):
    def reformat(date):
        MONTH = int(date[5:7])
        lst = np.zeros(12)
        lst[MONTH - 1] = 1
        return lst
    return list(map(reformat, dates))


# Given a list of dates of the form YYYY-MM-DD HH:MM
# returns a list of lists of all days as binary i.e.
# if 2021-1-1 is a friday it gives [0,0,0,0,1,0,0] on this day
# Then 2021-1-1 gives [0,0,0,0,0,1,0] (its a saturday)
# So it has length 7xlen(dates)
def days_binary(dates):
    def reformat(date):
        YYYYMMDD = date[0:10]
        lst = np.zeros(7)
        d = pd.Timestamp(YYYYMMDD)
        lst[d.dayofweek] = 1
        return lst

    return list(map(reformat, dates))


# Hours of daylight
# calculates per day how much hour of daylight there is
def HDL(dates):
    def HDL_DAYS(date):
        YEAR = int(date[0:4])
        MONTH = int(date[5:7])
        DAY = int(date[8:10])
        day_of_year = datetime.date(YEAR, MONTH, DAY).timetuple().tm_yday
        lambda_t = 0.4102 * math.sin(2 * math.pi / 365 * (day_of_year - 80.25))
        HDLt = 7.722 * math.acos(-math.tan(2 * math.pi * 52 / 360 * math.tan(lambda_t)))
        return [HDLt]
    return list(map(HDL_DAYS, dates))


# Binary Dummy for the hours of the day
# if the date is 2021-1-1 01:00 returns [1,0,...,0] (len24)
# If the date is 2021-1-1 2:15 returns [0,1,0,...,0]
def hours(dates):
    def HOURS(date):
        lst = np.zeros(24)
        HOUR = int(date[11:13])
        lst[HOUR] = 1
        return lst
    return list(map(HOURS, dates))


def hours_extracted(dates):
    def HOURS(date):
        return date[11:13]
    return list(map(HOURS, dates))

def get_date_month_day(dates):
    def YEAR_REMOVED(date):
        return date[5:10]
    

def get_day_of_week(dates):
    def day_of_week(date):
        my_date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M')
        return calendar.day_name[my_date.weekday()]
    return list(map(day_of_week, dates))

def get_date_month_day_hour(dates):
    def YEAR_REMOVED(date):
        return date[5:13]
    return list(map(YEAR_REMOVED, dates))

def hours_weekday(dates):
    def HOURS(date):
        lst = np.zeros(24)
        YEAR = int(date[0:4])
        MONTH = int(date[5:7])
        DAY = int(date[8:10])
        HOUR = int(date[11:13])

        x_date = datetime.date(YEAR,MONTH,DAY)
        no = x_date.weekday()
        if no < 5:
            lst[HOUR] = 1
        return lst
    return list(map(HOURS, dates))

def hours_weekendday(dates):
    def HOURS(date):
        lst = np.zeros(24)
        YEAR = int(date[0:4])
        MONTH = int(date[5:7])
        DAY = int(date[8:10])
        HOUR = int(date[11:13])
        x_date = datetime.date(YEAR, MONTH, DAY)
        no = x_date.weekday()

        if no >= 5:
            lst[HOUR] = 1
        return lst
    return list(map(HOURS, dates))
#Given coefficient values for the dates and the date itself re-calculates the date value of the regression model
def regression_model_coefficient_date_values(coefficients, dates):
    coefficent_months = coefficients[5:17]
    coefficient_days = coefficients[17:24]
    coefficient_hours = coefficients[24:72]
    def datum_tijdstip_waarde(date):
        YYYYMMDD = date[0:10]
        YEAR = int(date[0:4])
        MONTH = int(date[5:7])
        DAY = int(date[8:10])
        HOUR = int(date[11:13])
        d = pd.Timestamp(YYYYMMDD)


        lst_hours = np.zeros(48)
        x_date = datetime.date(YEAR, MONTH, DAY)
        no = x_date.weekday()
        if no < 5:
            lst_hours[HOUR] = 1
        else:
            lst_hours[HOUR+24] = 1
        lst_days = np.zeros(7)
        lst_days[d.dayofweek] = 1
        lst_months = np.zeros(12)
        lst_months[MONTH - 1] = 1
        return np.inner(coefficent_months, lst_months) + np.inner(coefficient_days, lst_days) + np.inner(
            coefficient_hours, lst_hours)
    return list(map(datum_tijdstip_waarde, dates))


# Dummy variables for the holidays
# I.e. returns 1 if we have a certain holiday
# Holidays are obtains from the holiday package for the Netherlands
def holiday_time(dates):
    def HOLIDAY(date):
        YEAR = int(date[0:4])
        holiday_dates = sorted([key for key in holidays.Netherlands(years=int(YEAR))])
        lst = np.zeros(len(holiday_dates))
        YYYYMMDD = date[0:10]
        for i in range(len(lst)):
            if str(holiday_dates[i]) == YYYYMMDD:
                lst[i] = 1
        return lst

    return list(map(HOLIDAY, dates))


# HEAT DEGREE DAYS
def HDD(temperatures):
    Tipping_point = 18
    def HDD_DAYS(temp):
        return [max(temp[0] - Tipping_point, 0)]
    return list(map(HDD_DAYS, temperatures))


# COLD DEGREE DAYS
def CDD(temperatures):
    Tipping_point = 18
    def CDD_DAYS(temp):
        return [max(Tipping_point - temp[0], 0)]
    return list(map(CDD_DAYS, temperatures))


#TODO: Fix wind functions - W2 needs a better curve
#TODO: Remove W3? Has no effect, maybe hinders other func's

# LOWER BOUNDARY WIND TURBINES: 4 m/s
# UPPER BOUNDARY WIND TURBINES: 20 m/s
def W1(wind_speeds):
    def W1_(wind_speed):
        return [int(wind_speed[0] < 4)]
    return list(map(W1_, wind_speeds))


def W2(wind_speeds):
    vc = 4
    vr = 11
    vf = 25
    def W2_(wind_speed):
        if wind_speed[0] < vc:
            return [0]
        elif wind_speed[0] > vf:
            return [0]
        elif wind_speed[0] > vr:
            return [1]#[10**3]
        else:
            return [(wind_speed[0]**3 - vc**3)/(vr**3-vc**3)] #[wind_speed[0]**3]
    return list(map(W2_, wind_speeds))


def W3(wind_speeds):
    def W3_(wind_speed):
        return [wind_speed[0]]
    return list(map(W3_, wind_speeds))




#Take the intersection of two dataframes based on a column


def linear_regression_model_data_controller(df_alliander, df_knmi):
    y_observed = [float(y) for y in df_alliander["BELASTING"].tolist()]  # observed value from alliander
    date = df_alliander["DATUM_TIJDSTIP"].tolist()  # dates, same for both df

    xtemp = df_knmi["T"].tolist()  # Temperature from the knmi
    xtemp = np.array([y for y in xtemp]).reshape(
        (len(xtemp), 1))  # Rescale it (kmni works with 100 for 10 degrees)
    xHDD = HDD(xtemp)
    xCDD = CDD(xtemp)
    xFF = np.array([[float(FF_value) ] for FF_value in df_knmi["FH"].tolist()])  # Windspeed from the knmi
    xW2 = W2(xFF)
    xSQ = np.array([[float(SQ_value)] for SQ_value in df_knmi["Q"].tolist()])  # Sunshine from knmi
    months_bin = months_binary(date)  # months bin
    days_bin = days_binary(date)  # days binary
    HOURS_WEEKDAY_t = hours_weekday(date)  # hours binary
    HOURS_WEEKENDDAY_t = hours_weekendday(date)  # hours binary
    X = np.column_stack(
        (np.ones((len(xtemp), 1)),xHDD,xCDD, xSQ, xW2, months_bin, days_bin, HOURS_WEEKDAY_t,HOURS_WEEKENDDAY_t))  # combine them into coefficient matrix
    return y_observed, X, date

def linear_regression_model_data_controller_old(df_alliander, df_knmi):
    y_observed = [float(y) for y in df_alliander["BELASTING"].tolist()]  # observed value from alliander
    date = df_alliander["DATUM_TIJDSTIP"].tolist()  # dates, same for both df

    xtemp = df_knmi["T"].tolist()  # Temperature from the knmi
    xtemp = np.array([y for y in xtemp]).reshape(
        (len(xtemp), 1))  # Rescale it (kmni works with 100 for 10 degrees)
    xHDD = HDD(xtemp)
    xCDD = CDD(xtemp)
    xFF = np.array([[float(FF_value) ] for FF_value in df_knmi["FH"].tolist()])  # Windspeed from the knmi
    xW2 = W2(xFF)
    xSQ = np.array([[float(SQ_value)] for SQ_value in df_knmi["Q"].tolist()])  # Sunshine from knmi
    months_bin = months_binary(date)  # months bin
    days_bin = days_binary(date)  # days binary
    HOURS_t = hours(date)  # hours binary
    X = np.column_stack(
        (np.ones((len(xtemp), 1)),xHDD,xCDD, xSQ, xW2, months_bin, days_bin, HOURS_t))  # combine them into coefficient matrix
    return y_observed, X, date


def linear_regression_model_data_controller_reduced(df_alliander, df_knmi,coefficients):
    y_observed = df_alliander["BELASTING"].tolist()  # observed value from alliander
    date = df_alliander["DATUM_TIJDSTIP"].tolist()  # dates, same for both df

    xtemp = df_knmi["T"].tolist()  # Temperature from the knmi
    xtemp = np.array([y for y in xtemp]).reshape(
        (len(xtemp), 1))
    xHDD = HDD(xtemp)
    xCDD = CDD(xtemp)
    xFF = np.array([[float(FF_value)] for FF_value in df_knmi["FH"].tolist()])  # Windspeed from the knmi
    xW1 = W1(xFF)
    xW2 = W2(xFF)
    xW3 = W3(xFF)
    xSQ = np.array([[float(SQ_value)] for SQ_value in df_knmi["Q"].tolist()])  # Sunshine from knmi
    xdate = regression_model_coefficient_date_values(coefficients, date)
    X = np.column_stack(
        (np.ones((len(xtemp), 1)),xCDD, xSQ, xW2, xdate))  # combine them into coefficient matrix
    return y_observed, X, date
