#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 12:34:46 2023
@author: khd2
"""
import pandas as pd
import numpy as np

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
