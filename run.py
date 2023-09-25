#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 12:35:39 2023
@author: khd2
"""
import pandas as pd
import readfile as rf
import outliers as Ol
import features as fs

file_dir = '/home/khd2/Downloads/prediction_flare/data/401_deltaL_time_deriv.txt'
data = rf.readfile(file_dir)

outliers = Ol.detectOutlier(data,'magnitude', output='outliers',plot=True)    # to get outliers only, change output = 'outliers'

variable = data['magnitude'].to_numpy()
time = data['magnitude'].to_numpy()

variables = ["time", "growth_rate", "acceleration"]
features ={i:[] for i in variables}

scale = 10
time_index = int(scale/2)
n,t=0,0
for l in range(n,len(time)-scale):
    Amp_window = variable[l:scale+l]
    tim_window = time[l:scale+l]
    corresponding_time = time[time_index+t]
    
    features['time'].append( corresponding_time  )
    features['growth_rate'].append( fs.growth_rate(Amp_window)   )
    features['acceleration'].append( fs.variationSpeed(Amp_window,time_step=720)  )
    
features = pd.DataFrame(features)



