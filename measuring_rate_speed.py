# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 09:31:00 2023
@author: khaled
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft#, fftfreq
from scipy import signal
from scipy import interpolate
#from scipy.interpolate import InterpolatedUnivariateSpline
import math

class System1():
    def __init__(self,timeseries, VarSeries):
        self.time_step = np.median(timeseries[1:]-timeseries[:-1])

        # Fs = len(timeseries)         # frequency samples 
        # n = len(timeseries)          # length of the signal
        # k = np.arange(n)
        # T = n/Fs
        # frq = k/T                    # two sides frequency range
        
        Nom_Amp = abs( fft(VarSeries)/len(VarSeries) )  # to normalize the frequencies amplitudes
        
        Amp_variation = np.array([Nom_Amp[v] for v in np.argsort(Nom_Amp)[::-1][:6] ])    # get the maximum amplitudes
        
        self.growth_rate = np.sqrt(np.sum(Amp_variation) * Amp_variation[0])
        
        self.variable_speed = np.median(VarSeries[1:] - VarSeries[:-1])/self.time_step     # median to not be influnced by fluctuations

        
class shortMemory(System1):
    def __int__(self,timeseries, VarSeries):        
        super().__init__(self,timeseries, VarSeries)

    def __call__(self):
        return [self.growth_rate, self.variable_speed], self.time_step


# class longMemory():
    
#     def __init__(self,pointsList, capacity = 10):
# #        super().__init__(self,timeseries, VarSeries)
#         self.values = pointsList

#         if len(self.values) <= capacity:
#             self.values.extend( pointsList[:] )
        
# #        self.LongtMemory = values.append(values[:])

#     def call(self):
#         self.values.append(self.values[:])
#         return LongtMemory
        
class System2():
#    def __init__(self, memorylist,startpoints=3,timestep=720):
#        super().__init__(timeseries, VarSeries, startpoints=3)
#        if (len(memorylist) >= startpoints):
#            self.Amp_acceleration = (memorylist['growth_rate'][1:] - memorylist['growth_rate'][:-1])/timestep    
    def __init__(self,Xvariblelist,Yvariblelist,startpoints=3,timestep=720, normalize=10**10):
        
        if (len(Xvariblelist) >=startpoints):
            Xvariblelist = Xvariblelist.to_numpy()
            Yvariblelist = Yvariblelist.to_numpy() / normalize
            f = interpolate.interp1d(Xvariblelist, Yvariblelist, fill_value='extrapolate')

            inclination = (f(Xvariblelist)[1:] - f(Xvariblelist)[:-1])/(Xvariblelist[1:] - Xvariblelist[:-1])
       #     print(inclination.shape)
            self.slope = math.degrees(math.atan(np.mean(inclination)))
        else:
            self.slope = 0

#        return (variablelist[1:] - variablelist[:-1])#/timestep
#        self.slope = math.tan(slope)
 #       return math.tan(slope)


# class longMemory:
#     def __init__(self,variables):
#         self.variables = variables
#     def __call__(self):
#         LMemory = {}
#         for v in self.variables:
#             LMemory[v] = []
#         return LMemory







## =========================================================================
file_dir = '/home/khd2/Downloads/prediction_flare/data/401_deltaL_time_deriv.txt'
file = open(file_dir,'r')

instances=[]
for i in file:
    row = i.split()
    instances.append(row)

time = np.array([np.float(i) for i in np.array(instances).T[0]]).T
magnitude = np.array([np.float(i) for i in np.array(instances).T[1]]).T

capacity = 10#len(magnitude)

Varbs = ["time", "growth_rate", "variable_speed"]
longMemory = []#longMemory(Varbs)()
slopes=[]
scale = 10
time_index = int(scale/2)
n,t=0,0
for l in range(n,len(time)-scale):
    SM = []
    Amp_window = magnitude[l:scale+l]
    tim_window = time[l:scale+l]
    corresponding_time = time[time_index+t]
 
    SM.append( corresponding_time  )
    measures, ts = shortMemory(tim_window, Amp_window)()
    SM.extend(measures)
    
    
#    ## fill the long memeory
#    for i,v in enumerate(Varbs):
#        LM[v].append(SM[::-1][i])

    longMemory.append(SM)
    if (len(longMemory) > capacity):
        longMemory = longMemory[1:]
    
    LM = pd.DataFrame(longMemory)
    LM.columns = Varbs


    
    S2 = System2(LM[Varbs[0]],LM[Varbs[1]],startpoints=3,timestep=ts)
    slopes.append( [S2.slope,corresponding_time] )
    
    t+=1
#LM.insert(3,'slopes',slopes)                
 #       if len(longMemory) > int(0.capacity):
  #          longMemory = longMemory[1:]
          #  longMemory.append(X)
    
#longMemory = pd.DataFrame(longMemory)
#longMemory.columns = ["time", "growth_rate", "variable_speed"]
    
"""
It is found that when longMemory capacity is short, it is sensitive to get the rapidly 
increase/peak in the variables. The slope angle become sensitive and bigger.

"""
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.interpolate import InterpolatedUnivariateSpline

# # given values
# xi = np.array([0.2, 0.5, 0.7, 0.9+1])
# #yi = np.array([0.3, -0.1, 0.2, 0.1])
# yi = np.exp(-xi/3.0)
# # positions to inter/extrapolate
# x = np.linspace(0, 1, 50)
# # spline order: 1 linear, 2 quadratic, 3 cubic ... 
# order = 1
# # do inter/extrapolation
# s = InterpolatedUnivariateSpline(xi, yi, k=order)
# y = s(x)

# # example showing the interpolation for linear, quadratic and cubic interpolation
# plt.figure()
# plt.plot(xi, yi)
# for order in range(1, 3):
#     s = InterpolatedUnivariateSpline(xi, yi, k=order)
#     y = s(x)
#     plt.plot(x, y)
# plt.show()
    