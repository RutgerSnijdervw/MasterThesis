import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Database_KNMI as dbk
#import seaborn
#seaborn.set(style='ticks')
from collections import Counter

def histogram_heat_pumps():
    np.random.seed(0)
    df = pandas.DataFrame(np.random.normal(size=(37,2)), columns=['A', 'B'])
    fig, ax = plt.subplots()
    a_heights, a_bins = np.histogram(df['A'])
    b_heights, b_bins = np.histogram(df['B'], bins=a_bins)

    a_heights = [108804, 162161,375764,362648,413635]
    b_heights = [570041,732202,1107966,1470614,1884249]
    a_bins = [2017,2018,2019,2020,2021,2022]
    b_bins = [2017, 2018,2019,2020,2021,2022]
    width = (a_bins[1] - a_bins[0])/3
    ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue')
    ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen')
    fig.show()
    #seaborn.despine(ax=ax, offset=10)
    return

def sin_cos_plot():
    x_values = np.arange(1, 101) / 10
    sin_values = np.random.normal(size=100) + np.sin(2 * np.pi * x_values)
    cos_values = np.random.normal(size=100) + np.cos(2 * np.pi * x_values)
    plt.figure(figsize=(12,6))
    plt.plot(x_values,sin_values, 'ro-')
    plt.plot(x_values, cos_values, 'bo-')
    plt.xlabel("x values", fontsize = 40)
    plt.ylabel("y values", fontsize = 40)
    #plt.title("sin & cos plot")
    plt.legend(["sin(x)", "cos(x)"],fontsize=20)
    plt.savefig("Sin_Cos_ex.pdf", dpi=300,format = "pdf", bbox_inches='tight')
    plt.close()

    x_values_more_points = np.arange(1, 1000) / 100
    sin_values_normal = np.sin(2 * np.pi * x_values_more_points)
    cos_values_normal = np.cos(2 * np.pi * x_values_more_points)
    plt.figure(figsize=(12, 6))
    plt.plot(x_values_more_points, sin_values_normal, 'r')
    plt.plot(x_values_more_points, cos_values_normal, 'b')
    plt.xlabel("x values", fontsize = 40)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylabel("y values", fontsize = 40)
   # plt.title("sin & cos plot", fontsize = 18)
    plt.legend(["sin(x)", "cos(x)"],fontsize=20)
    plt.savefig("Sin_Cos_ex_normal.pdf", dpi=300,format = "pdf", bbox_inches='tight')
    
    plt.close()

    plt.figure(figsize=(12,6))
    plt.plot(x_values, sin_values, 'ro-')
    plt.xlabel("x values", fontsize = 40)
    plt.ylabel("y values", fontsize = 40)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.savefig("Sin_ex.pdf", dpi=300,format = "pdf", bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12,6))
    acf, ci = sm.tsa.acf(sin_values, nlags = 99, alpha = 0.05)
    for i in range(len(acf)):
        plt.vlines(i,0,acf[i])
    plt.xlabel("Lag", fontsize = 40)
    plt.ylabel("ACF", fontsize = 40)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    #plt.title("sin ACF plot")
    x_values = list(range(-2, 103, 1))
    zero_line = [0 for x in x_values]
    plt.plot(x_values, zero_line, "k")
    plt.xlim([x_values[0], x_values[-1]])
    plt.savefig("ACF_Sin.pdf", dpi=300,format = "pdf", bbox_inches='tight')

    plt.close()

    backwards = smt.ccf(sin_values[::-1], cos_values[::-1], adjusted=False)[::-1]
    forwards = smt.ccf(sin_values, cos_values, adjusted=False)
    ccf_output = np.r_[backwards[:-1], forwards]
    plt.figure(figsize=(12,6))
    x_values = list(range(-99,100,1))
    for i in range(len(x_values)):
        plt.vlines(x_values[i],0,ccf_output[i])
    plt.ylabel("CCF", fontsize = 40)
    #plt.title("Sin and Cos CCF plot")
    plt.xlabel("Lag", fontsize = 40)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    x_values = list(range(-102,103,1))
    zero_line = [0 for x in x_values]
    plt.plot(x_values, zero_line, "k")
    plt.xlim([x_values[0],x_values[-1]])
    plt.savefig("CCF.pdf", dpi=300,format = "pdf", bbox_inches='tight')
    return
#histogram_heat_pumps()

def missing_data_info():
    column_names = dbk.get_column_names_KNMI()
    
    df_knmi = dbk.get_dataframe_KNMI([2022])
    stations = df_knmi["STN"].drop_duplicates().tolist()
    print(len(stations))
    missing_data = np.zeros(len(column_names))
    for col in column_names:
        data = []
        complete_stations = 0
        col_values = df_knmi[col].tolist()
        data.append(col)
        data.append(str(df_knmi[col].isna().sum()) + " (" + str(round(df_knmi[col].isna().sum()/len(df_knmi)*100,2)) + "\%)")
        
        for station in stations:
            df_knmi_station = df_knmi[df_knmi["STN"] == station]
            if df_knmi_station[col].isna().sum() == 0:
                complete_stations += 1
        data.append(str(complete_stations)+ " ("+str(round(complete_stations/len(stations)*100,2)) + "\%)")
        print("&".join(data) + "\\")

    return  

def train_test_result_table():
    df = pd.read_csv("train_test_results_22_21.csv")
    df = df[["station","OLS_TEST_R2","OLS_TEST_MAPE","OLS_TEST_DB","GLS_TEST_R2","GLS_TEST_MAPE","GLS_TEST_DB"]]
    df_avg = df.mean().tolist() #Takes mean of df and ignores the first value (thats the station value)
    print(df_avg)
    df = pd.concat([df.head(4),df.tail(4)])
    print(df)
    for index, row in df.iterrows():
        row_ = row.tolist()
        row_[1:] = [str(round(entry,2)) for entry in row_[1:]]
        print(" & ".join(row_) + "\\")
    df_avg = [str(round(entry,2)) for entry in df_avg]
    print(" & ".join(df_avg))
    return
train_test_result_table()
