#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

from plotFunctions import plotSpatial, plotTemporal


# switch on or off the overplotting of the fig3e data
plotBench = False

# read and plot all depth profiles

spatial_files = glob.glob('solution_t_*.ascii')
for i in range(0,len(spatial_files)):
    x = (pd.read_csv(spatial_files[i], delim_whitespace=True)).shift(axis=1).iloc[:,1:]
    plotSpatial(x, spatial_files[i].rstrip('.ascii'), plotBench)


# read and plot all time series

times_files = glob.glob('solution_x_*.ascii')
for i in range(0,len(times_files)):
    x = (pd.read_csv(times_files[i],delim_whitespace=True)).shift(axis=1).iloc[:,1:]
    plotTemporal(x,times_files[i].rstrip('.ascii'))