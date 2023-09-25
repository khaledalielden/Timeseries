#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 12:48:52 2023

@author: khd2
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

def readfile(file_dir):
    data = pd.DataFrame()
    
    file = open(file_dir,'r')
    instances=[]
    for i in file:
        row = i.split()
        instances.append(row)
    
    data['time'] = np.array([np.float(i) for i in np.array(instances).T[0]]).T
    data['magnitude'] = np.array([np.float(i) for i in np.array(instances).T[1]]).T
    return data

def seasonal_decompose(df,plot=False):
    decomposition = sm.tsa.seasonal_decompose(df, model='additive', freq=10)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    if plot is True:
        fig = decomposition.plot()
        fig.set_size_inches(14, 7)
        plt.show()
    
    return trend, seasonal, residual

def get_fitted_intervals(y, fitted):
        n = len(y)
        # denom = n * np.sum((y - np.mean(y))**2)
        sd_error = np.sqrt((1 / max(1, (n - 2))) * np.sum((y - np.mean(fitted))**2))
        # sd_error = np.sqrt(top / denom)
        # sd_error = np.std(y - fitted)
        t_stat = stats.t.ppf(.9, len(y))
        upper = fitted + t_stat*sd_error
        lower = fitted - t_stat*sd_error
        return upper, lower

def detectOutlier(df,variable,output='all',plot=False):
    t,s,r = seasonal_decompose(df[variable])
    df['fitted'] = t + s
    upper,lower = get_fitted_intervals(df[variable],df['fitted'] )
    df['outliers'] = (df[variable] >  upper) | (df[variable] <  lower)
    
    outliers_df = df[df['outliers'] == True]
    
    
    if plot is True:
        fig, ax = plt.subplots(figsize=None)
        ax.plot(df[variable], color='black')
        ax.plot(df['fitted'] , color='orange')
        ax.plot(upper, linestyle='dashed', alpha=.5, color='orange')
        ax.plot(lower, linestyle='dashed', alpha=.5, color='orange')
        
        ax.scatter(outliers_df.index, outliers_df[variable], marker='x', color='red')
        plt.show()

    if output == 'all':
        return df
    if output == 'outliers':
        return outliers_df

