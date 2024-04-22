import numpy as np
import math
import statsmodels.api as sm
import statsmodels.tsa.stattools as smt
import matplotlib.pyplot as plt
import pandas as pd
import Controller as cll

#most plots for mathematical examples
def ACF_example_sin():
    sin_values = [np.random.normal()+ math.cos(2*math.pi*x/10) for x in range(100)]
    plt.plot([x for x in range(len(sin_values))], sin_values, "ro-")
    plt.savefig("ACF_example_cos_points.png", dpi=150)
    sm.graphics.tsa.plot_acf(sin_values, lags = 99)
    plt.savefig("ACF_example_cos.png", dpi=150)
    return

def ACF_example_sin2():
    sin_values = [math.sin(2*math.pi*x) for x in range(100)]
    plt.plot([x for x in range(len(sin_values))], sin_values)
    plt.show()
    return

def ACF_example():
    dta = sm.datasets.sunspots.load_pandas().data
    dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
    del dta["YEAR"]
    print(type(dta.values.squeeze()))
    print(len(dta.values.squeeze()))
    sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40)
    plt.show()
    return

def sin_cos_plot():
    x_values = np.arange(1, 101) / 10
    sin_values = np.random.normal(size=100) + np.sin(2 * np.pi * x_values)
    cos_values = np.random.normal(size=100) + np.cos(2 * np.pi * x_values)
    plt.figure(figsize=(12,6))
    plt.plot(x_values,sin_values, 'ro-')
    plt.plot(x_values, cos_values, 'bo-')
    plt.xlabel("x values", fontsize =32)
    plt.ylabel("y values", fontsize = 32)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(["sin(x)", "cos(x)"], fontsize = 24)
    plt.savefig("\Sin_Cos_ex.png", bbox_inches = 'tight')
    plt.close()

    x_values_more_points = np.arange(1, 1000) / 100
    sin_values_normal = np.sin(2 * np.pi * x_values_more_points)
    cos_values_normal = np.cos(2 * np.pi * x_values_more_points)
    plt.figure(figsize=(12, 6))
    plt.plot(x_values_more_points, sin_values_normal, 'r')
    plt.plot(x_values_more_points, cos_values_normal, 'b')
    plt.xlabel("x values", fontsize = 32)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylabel("y values", fontsize = 32)
    plt.legend(["sin(x)", "cos(x)"], fontsize = 24)
    plt.savefig("Sin_Cos_ex_normal.png", bbox_inches = 'tight')
    plt.close()

    plt.figure(figsize=(12,6))
    plt.plot(x_values, sin_values, 'ro-')
    plt.xlabel("x values", fontsize = 32)
    plt.ylabel("y values", fontsize = 32)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.savefig("Sin_ex.png", bbox_inches = 'tight')
    plt.close()

    plt.figure(figsize=(12,6))
    acf, ci = sm.tsa.acf(sin_values, nlags = 99, alpha = 0.05)
    for i in range(len(acf)):
        plt.vlines(i,0,acf[i])
    plt.xlabel("Lag", fontsize = 32)
    plt.ylabel("ACF", fontsize = 32)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    x_values = list(range(-2, 103, 1))
    zero_line = [0 for x in x_values]
    plt.plot(x_values, zero_line, "k")
    plt.xlim([x_values[0], x_values[-1]])
    plt.savefig("ACF_Sin.png", bbox_inches='tight')
    plt.close()

    backwards = smt.ccf(sin_values[::-1], cos_values[::-1], adjusted=False)[::-1]
    forwards = smt.ccf(sin_values, cos_values, adjusted=False)
    ccf_output = np.r_[backwards[:-1], forwards]
    plt.figure(figsize=(12,6))
    x_values = list(range(-99,100,1))
    for i in range(len(x_values)):
        plt.vlines(x_values[i],0,ccf_output[i])
    plt.ylabel("CCF", fontsize = 32)
    plt.xlabel("Lag", fontsize = 32)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    x_values = list(range(-102,103,1))
    zero_line = [0 for x in x_values]
    plt.plot(x_values, zero_line, "k")
    plt.xlim([x_values[0],x_values[-1]])
    plt.savefig("CCF.png", bbox_inches='tight')
    return



