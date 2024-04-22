import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import Database_KNMI as dbk
import Database_Alliander as dba
import numpy as np
import pandas as pd
import pylab
import statsmodels.api as sm
import matplotlib as mpl
import Regression_Model as rm




####################################################################################Standardized plots###############################################################
#Type cast a list of decimals (or any other format) to a list of floats, in a np array
#Using vectorization for faster performance
def decimal_to_float(lst):
    def dec_to_float(l):
        return float(l)
    return np.array(list(map(dec_to_float,lst)))


#Plot a single value against the date
def plot_vs_date(date,value, title= "", xaxis= "", yaxis= "",pdf = "", show_plot = False, png = ""):
    value = decimal_to_float(value)
    fig, ax = plt.subplots(figsize=(11.69,8.27))
    ax.set_title(title)
    ax.xaxis.set_major_locator(plt.MaxNLocator(8))  # reduce the amount of visible ticks (1 for every time is a bit much)
    #ax.tick_params(axis='x', labelrotation=30)  # rotate them so they do not overlap
    ax.tick_params(axis='x', labelrotation=15, labelsize = 20)  # rotate them so they do not overlap
    ax.tick_params(axis='y', labelsize = 20)
    # make a plot
    ax.plot(date, value)#, color="red")

    # set x-axis label
    ax.set_xlabel(xaxis, fontsize=40)
    # set y-axis label
    ax.set_ylabel(yaxis,fontsize=40)
    if show_plot:
        plt.show()
    elif png == "":
        plt.savefig(pdf, format='pdf', bbox_inches='tight')
    else:
        plt.savefig(png, dpi = 300, bbox_inches='tight')
    plt.close()
    return

#Multi plot vs the date
#Using double axis since the value range might differ
def multiplot_vs_date(date,value1,value2, title= "", xaxis= "", yaxis1= "", yaxis2 = "",
                      pdf = "",
                      png = "", align_ax = False,
                      show_plot = False,
                      legend = []):
    value1 = decimal_to_float(value1)
    value2 = decimal_to_float(value2)
    fig, ax = plt.subplots(figsize=(11.69,8.27))
    ax.set_title(title, fontsize = 40)
    ax.xaxis.set_major_locator(plt.MaxNLocator(8))  # reduce the amount of visible ticks
    ax.tick_params(axis='x', labelrotation=15, labelsize = 20)  # rotate them so they do not overlap
    ax.tick_params(axis='y', labelsize = 20)
   # ax.tick_params(labelsize=24)
    # make a plot
    ax.plot(date, value1, color="red")
    # set x-axis label
    ax.set_xlabel(xaxis, fontsize=40)
    # set y-axis label
    ax.set_ylabel(yaxis1,fontsize=40, color = "red")#color="red",
    
                  
    
    if yaxis2 == "":
        ax.plot(date, value2, color = "blue")
    else:
        # twin object for two different y-axis on the sample plot
        ax2 = ax.twinx()
    # make a plot with different y-axis using second axis object
        ax2.plot(date, value2, color="blue")#, alpha=0.6)
        ax2.set_ylabel(yaxis2, color="blue", fontsize=40)
    if align_ax:
        concat_lsts = np.concatenate([value1,value2])
        min_value = min(concat_lsts)
        max_value = max(concat_lsts)
        ax.set_ylim([min_value, max_value])
        if yaxis2 != "":
            ax2.set_ylim([min_value, max_value])
    if legend != []:
        ax.legend(legend, loc = 1,fontsize=24)
    if show_plot:
        plt.show()
    if png == "":
        plt.savefig(pdf, format='pdf', bbox_inches='tight')
    else:
        plt.savefig(png, dpi = 300, bbox_inches='tight')
    plt.close()
    return


#Scatterplot of 2 values
#TODO: update the scatterplot to includ
def scatterplot(value1,value2, title= "", xaxis= "", yaxis= "", pdf = "", align_ax = False,png = "", color = ""):
    value1 = [float(value_point) for value_point in value1]
    value2 = [float(value_point) for value_point in value2]
    np.seterr(invalid='ignore')
    smoothed = sm.nonparametric.lowess(exog=value1, endog=value2, frac=0.2) #lowess fit - Locally Weighted Scatterplot Smoothing
    fig, ax = plt.subplots(figsize=(11.69,8.27))
    ax.tick_params(labelsize = 20)
    ax.set_title(title)
    # make a plot
    if color == "":
        ax.scatter(value1, value2, color="red", s = 10)
    else:
        ax.scatter(value1, value2, color=color, s=10)
    #ax.plot(smoothed[:, 0], smoothed[:, 1], c="k",linestyle='--') #Lowess fit
    pylab.autoscale(enable=True, axis="x", tight=True)
    # set x-axis label
    ax.set_xlabel(xaxis, fontsize=40)
    # set y-axis label

    ax.set_ylabel(yaxis, fontsize=40)
    if align_ax: #Possible allignment of axis (to make the plot a square)
        concat_lsts = np.concatenate([value1,value2])
        min_value = min(concat_lsts)
        max_value = max(concat_lsts)
       # ax.plot([min_value, max_value], [min_value, max_value], color="k")
        ax.set_ylim([min_value, max_value])
        ax.set_xlim([min_value, max_value])
    if png != "":
        plt.savefig(png, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(pdf, format='pdf', bbox_inches='tight')
    plt.close()
    return

def scatterplot_temp(value1,value2,value3,value4, title= "", xaxis= "", yaxis= "", pdf = "", align_ax = False,png = "", color = ""):
    value1 = [float(value_point) for value_point in value1]
    value2 = [float(value_point) for value_point in value2]
    value3 = [float(value_point) for value_point in value3]
    value4 = [float(value_point) for value_point in value4]
    np.seterr(invalid='ignore')
    smoothed = sm.nonparametric.lowess(exog=value1, endog=value2, frac=0.2) #lowess fit - Locally Weighted Scatterplot Smoothing
    fig, ax = plt.subplots(figsize=(11.69,8.27))
    ax.tick_params(labelsize = 20)
    ax.set_title(title)
    # make a plot
    ax.scatter(value1, value2, color="blue", s=10)
    ax.scatter(value3, value4, color="red", s=10)
    #ax.plot(smoothed[:, 0], smoothed[:, 1], c="k",linestyle='--') #Lowess fit
    pylab.autoscale(enable=True, axis="x", tight=True)
    # set x-axis label
    ax.set_xlabel(xaxis, fontsize=40)
    # set y-axis label

    ax.set_ylabel(yaxis, fontsize=40)
    if align_ax: #Possible allignment of axis (to make the plot a square)
        concat_lsts = np.concatenate([value1,value2])
        min_value = min(concat_lsts)
        max_value = max(concat_lsts)
       # ax.plot([min_value, max_value], [min_value, max_value], color="k")
        ax.set_ylim([min_value, max_value])
        ax.set_xlim([min_value, max_value])
    if png != "":
        plt.savefig(png, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(pdf, format='pdf', bbox_inches='tight')
    plt.close()
    return


#Creates a scatter plot of value1 and value2 with a line for line_value_x and line_value_y
def scatterplot_with_line(value1,value2, line_value_x, line_value_y, title= "", xaxis= "", yaxis= "", pdf = PdfPages('foo.pdf'), align_ax = False, png = ""):
    value1 = [float(value_point) for value_point in value1] #safety conversion
    value2 = [float(value_point) for value_point in value2]
    np.seterr(invalid='ignore')
    smoothed = sm.nonparametric.lowess(exog=value1, endog=value2, frac=0.2) #lowess fit - Locally Weighted Scatterplot Smoothing
    fig, ax = plt.subplots(figsize=(11.69,8.27))
    ax.set_title(title)
    # make a plot
    ax.scatter(value1, value2, color="red", s = 8)
    #ax.plot(smoothed[:, 0], smoothed[:, 1], c="k",linestyle='--') #Lowess fit
    #pylab.autoscale(enable=True, axis="x", tight=True)
    # set x-axis label
    ax.set_xlabel(xaxis, fontsize=14)
    # set y-axis label
    ax.set_ylabel(yaxis,
                  color="red",
                  fontsize=14)

    ax.plot(line_value_x, line_value_y, color='k',linestyle='dashed')
    if align_ax: #Possible allignment of axis (to make the plot a square)
        concat_lsts = np.concatenate([value1,value2])
        min_value = min(concat_lsts)
        max_value = max(concat_lsts)
        ax.plot([min_value, max_value], [min_value, max_value], color="k")
        ax.set_ylim([min_value, max_value])
        ax.set_xlim([min_value, max_value])

    if png != "":
        plt.savefig(png, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(pdf, format='pdf', bbox_inches='tight')
    plt.close()
    return


def scatterplot_with_corr_line(value1,value2, title= "", xaxis= "", yaxis= "", pdf = PdfPages('foo.pdf'), align_ax = False, png = ""):
    value1 = [float(value_point) for value_point in value1] #safety conversion
    value2 = [float(value_point) for value_point in value2]
    np.seterr(invalid='ignore')
    smoothed = sm.nonparametric.lowess(exog=value1, endog=value2, frac=0.2) #lowess fit - Locally Weighted Scatterplot Smoothing
    fig, ax = plt.subplots(figsize=(12.86,9.1))
    ax.set_title(title, fontsize=10)
    # make a plot
    ax.scatter(value1, value2, color="red", s = 8)
    #ax.plot(smoothed[:, 0], smoothed[:, 1], c="k",linestyle='--') #Lowess fit
    #pylab.autoscale(enable=True, axis="x", tight=True)
    # set x-axis label
    ax.set_xlabel(xaxis, fontsize=40)
    # set y-axis label
    ax.set_ylabel(yaxis,fontsize=40)
                  #color="red",
                  
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()

    concat_lsts = np.concatenate([value1, value2])
    min_value = min(concat_lsts)
    max_value = max(concat_lsts)
    X = np.column_stack((np.ones((len(value1), 1)), value1))
    OLS_simple = rm.linear_regression_statsmodels(value2, X)
    a_coeff,b_coeff = OLS_simple.params

   # b_coeff = rm.pearson_correlation_coeff(value1, value2) # value 2 = a + value1*b
   # a_coeff = rm.simple_linear_regression_b_known(value2, value1, b_coeff)
    X_corr_line = np.linspace(min_value-10,max_value+10,100)
    ax.plot(X_corr_line, X_corr_line*b_coeff+a_coeff, color='k', linestyle='dashed')
    ax.set_ylim([ymin, ymax])
    ax.set_xlim([xmin, xmax])
    ax.tick_params(labelsize=26)
    if align_ax: #Possible allignment of axis (to make the plot a square)

        ax.plot([min_value, max_value], [min_value, max_value], color="k")
        ax.set_ylim([min_value, max_value])
        ax.set_xlim([min_value, max_value])

    if png != "":
        plt.savefig(png, dpi=200, bbox_inches='tight')
    else:
        plt.savefig(pdf, format='pdf', bbox_inches='tight')
    plt.close()
    return
#Given a xvalue of the form [x1,...,xn] and yvalues of the form [[y1,...,yn],...,[y_1,...,y_n]]
#Creates a plot for all the different y lines with different colours (only want nice ones)
#And also different markers
def multiple_line_plots(xvalue, yvalues, legends = [],title= "", xaxis= "", yaxis= "", pdf = "", png = ""):
    colors_array = list(mpl.colors.cnames.keys())
    markers_array = ['-','--','-.',':']
    cm = plt.get_cmap('gist_rainbow')
    fig, ax = plt.subplots(figsize=(15.19,10.75))
    colors = [cm(1.*i/len(yvalues)) for i in range(len(yvalues))]
    colors = colors[::2] #remove every second element
    #colors = [ele for ele in colors for i in range(2)] #copy every element in same list
    ax.set_prop_cycle(color = colors)
    for i in range(len(yvalues)):
       # if i < 10:
       #     ax.plot(xvalue, yvalues[i], label=legends[i],linestyle = markers_array[(i%2)])#len(markers_array)#color = colors_array[i],
       ax.plot(xvalue, yvalues[i], label=legends[i], linestyle=markers_array[i%len(markers_array)])
    ax.legend(legends,fontsize=20)
    ax.set_title(title, fontsize = 40)
    ax.tick_params(labelsize=19)
    ax.set_xlabel(xaxis, fontsize=40)
    # set y-axis label
    ax.set_ylabel(yaxis,
                  fontsize=40)
    if png != "":
        plt.savefig(png, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(pdf, format='pdf', bbox_inches='tight')
    plt.close()
    return

#Creates ACF plots for list
def ACF_plot(values, lags = 1, xaxis = "Lags", yaxis = "ACF", pdf = "",
             title = "",png = ""):
    sm.graphics.tsa.plot_acf(values, lags=lags)
    plt.xlabel(xaxis, fontsize = 40)
    plt.ylabel(yaxis, fontsize = 40)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    if png != "":
        plt.savefig(png, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(pdf, format='pdf', bbox_inches='tight')
    plt.close()
    return

#Given a string of text, stores it in a pdf
def text_to_pdf(txt, pdf):
    firstPage = plt.figure(figsize=(11.69, 8.27))
    firstPage.clf()
    firstPage.text(.2, .9, txt, size=6, horizontalalignment="left", verticalalignment='top')
    pdf.savefig()
    plt.close()
    return

####################################################################################Special Plots for report###############################################################
#Scatterplot same as before, but now makes it a heatmap based on another columns values (value_heat)
#and based on the ranges from ranges_heat
def heat_map_scatterplot(value1,value2,value_heat,ranges_heat=[3,2,1,0], title= "", xaxis= "", yaxis= "", pdf = "", align_ax = False):
    value1 = [float(value_point) for value_point in value1]
    value2 = [float(value_point) for value_point in value2]
    np.seterr(invalid='ignore')
    smoothed = sm.nonparametric.lowess(exog=value1, endog=value2, frac=0.2) #lowess fit - Locally Weighted Scatterplot Smoothing

    df = pd.DataFrame(list(zip(value1,value2,value_heat)), columns = ["value1", "value2", "value_heat"])
    colours = ["red", "darkorange", "yellow", "green", "blue"]

    fig, ax = plt.subplots(figsize=(11.69,8.27))
    ax.set_title(title)
    
    # make a plot
    df_ = df[df["value_heat"] > ranges_heat[0]]
    ax.scatter(df_["value1"].tolist(), df_["value2"].tolist(), color=colours[0], s = 10)
    ax.tick_params(labelsize=19)
    for i in range(0,len(ranges_heat)-1):
        df_ = df[(df["value_heat"] < ranges_heat[i]) & (df["value_heat"] > ranges_heat[i+1])]
        ax.scatter(df_["value1"].tolist(), df_["value2"].tolist(), color=colours[i+1], s=10)
    df_ = df[df["value_heat"] < ranges_heat[-1]]
    ax.scatter(df_["value1"].tolist(), df_["value2"].tolist(), color=colours[-1], s=10)
    ax.plot(smoothed[:, 0], smoothed[:, 1], c="k",linestyle='--') #Lowess fit
    pylab.autoscale(enable=True, axis="x", tight=True)
    # set x-axis label
    ax.set_xlabel(xaxis, fontsize=40)
    # set y-axis label
    ax.set_ylabel(yaxis,
                  color="red",
                  fontsize=40)
    if align_ax:
        concat_lsts = np.concatenate([value1,value2])
        min_value = min(concat_lsts)
        max_value = max(concat_lsts)
        ax.plot([min_value, max_value], [min_value, max_value], color="k")
        ax.set_ylim([min_value, max_value])
        ax.set_xlim([min_value, max_value])
    plt.legend(["temp > %d" %ranges_heat[0],"%d > temp >  %d" %(ranges_heat[0], ranges_heat[1]),
                "%d > temp >  %d" %(ranges_heat[1], ranges_heat[2]),
                "%d > temp >  %d" %(ranges_heat[2], ranges_heat[3]),
                "%d > temp" % ranges_heat[3]], fontsize = 20)

    plt.savefig(pdf, format='pdf', bbox_inches='tight')
    plt.close()
    return

#These are special one time used plots
def plot1():
    df_alliander = dba.baseload_data("MSR", "2 001 665") #voorschoten
    pd.set_option('display.max_columns', None)
    df_knmi = dbk.get_data_KNMI_single_station([2021],[215])
    df_alliander_hourly = dba.quarter_to_hourly(df_alliander, "BELASTING")
    df_alliander_daily = dba.quarter_to_daily(df_alliander, "BELASTING")
    df_knmi_hourly = df_knmi[["DATUM_TIJDSTIP", "T"]].copy()
    df_knmi_daily = dba.quarter_to_daily(df_knmi, "T")

    dates = [ts.strftime('%Y-%m-%d') for ts in df_alliander_daily.index.tolist()]
    multiplot_vs_date(dates,df_alliander_daily["BELASTING"],df_knmi_daily["T"], "Voorschoten load and temperature",
                      "Date", "Load", "Temperature", png = "Daily_Load.png")


    sm.graphics.tsa.plot_acf(df_alliander_hourly["BELASTING"].values.tolist(), lags=168)
    plt.savefig("ACF_hourly.png", dpi = 150)

    sm.graphics.tsa.plot_acf(df_alliander_daily["BELASTING"].values.squeeze(), lags=31)
    plt.savefig("ACF_daily.png", dpi = 150)

    sm.tsa.stattools.ccf(df_alliander_hourly["BELASTING"],df_knmi_hourly["T"])
    plt.savefig("test.png")
    return

def plot2():
    pdf = PdfPages('/home/as2-streaming-user/MyFiles/TemporaryFiles/Voorschoten.pdf')
    print("Jaar 22")
    df_alliander22 = dba.baseload_data_2022("MSR", "2 001 665")  # voorschoten
    print("Jaar 21")
    df_alliander21 = dba.baseload_data_2021("MSR", "2 001 665")  # voorschoten
    print("Jaar 20")
    df_alliander20 = dba.baseload_data_2020("MSR", "2 001 665")  # voorschoten
    print("Jaar 19")
    df_alliander19 = dba.baseload_data_2019("MSR", "2 001 665")  # voorschoten
    df_alliander = pd.concat([df_alliander19,df_alliander20,df_alliander21,df_alliander22])

    df_knmi = dbk.get_data_KNMI_single_station([2019,2020,2021,2022], [215])
    df_alliander_daily = dba.quarter_to_daily(df_alliander, "BELASTING")
    df_knmi_daily = dba.quarter_to_daily(df_knmi, "T")

    dates = [ts.strftime('%Y-%m-%d') for ts in df_knmi_daily.index.tolist()]
    dates_alliander = [ts.strftime('%Y-%m-%d') for ts in df_alliander_daily.index.tolist()]
    plot_vs_date(dates, df_knmi_daily["T"].tolist(), title = "Temperature vs Date Voorschoten", xaxis="Date",yaxis = "Temperature (Celcius)", show_plot = True)
    plot_vs_date(dates_alliander, df_alliander_daily["BELASTING"].tolist(), title="Load vs Date Voorschoten", xaxis="Date",
                 yaxis="Load (MW)", show_plot=True)
    pdf.close()
    return


def temp():
    date = [i for i in range(1,25)]
    value = [-17.8,-15.3,-16.8,-17.9,-19,-19.2,-18.7,-18.9,-15.6,-13.6,-11.6,-9.6,-8,-7.2,-7.5,-8.8,-10.3,-13.6,-15.6,
              -15.7,-17.3,-17.2,-11.7,-8.9]
    xaxis = "Hours"
    yaxis = "Temperature[Celcius]"
    pdf = ""
    show_plot = False
    png = "Lelystad_profile.png"
    title = ""
    
    value = decimal_to_float(value)
    fig, ax = plt.subplots(figsize=(11.69,8.27))
    ax.set_title(title)
    ax.xaxis.set_major_locator(plt.MaxNLocator(8))  # reduce the amount of visible ticks (1 for every time is a bit much)
    #ax.tick_params(axis='x', labelrotation=30)  # rotate them so they do not overlap
    ax.tick_params(axis='x', labelrotation=15, labelsize = 20)  # rotate them so they do not overlap
    ax.tick_params(axis='y', labelsize = 20)
    # make a plot
    ax.plot(date, value, marker = 'o')#, color="red")
    ax.set_ylim([-22,0])
    # set x-axis label
    ax.set_xlabel(xaxis, fontsize=40)
    # set y-axis label
    ax.set_ylabel(yaxis,fontsize=40)
    if show_plot:
        plt.show()
    elif png == "":
        plt.savefig(pdf, format='pdf', bbox_inches='tight')
    else:
        plt.savefig(png, dpi = 300, bbox_inches='tight')
    plt.close()
    return

