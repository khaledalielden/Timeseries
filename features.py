#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:43:00 2023
@author: khd2
"""
from scipy.fft import fft
import numpy as np

def growth_rate(VarSeries):

    # Fs = len(timeseries)         # frequency samples 
    # n = len(timeseries)          # length of the signal
    # k = np.arange(n)
    # T = n/Fs
    # frq = k/T                    # two sides frequency range
    
    Nom_Amp = abs( fft(VarSeries)/len(VarSeries) )  # to normalize the frequencies amplitudes 
    Amp_variation = np.array([Nom_Amp[v] for v in np.argsort(Nom_Amp)[::-1][:6] ])    # get the maximum amplitudes
    
    return np.sqrt(np.sum(Amp_variation) * Amp_variation[0])
    
def variationSpeed(VarSeries,time_step=720):
    return np.mean(VarSeries[1:] - VarSeries[:-1])/time_step     # median to not be influnced by fluctuations

