#!/usr/bin/env python3

###  plotting output from DiagenesisModel ###

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
    
    
def plotSpatial(df, filename, benchmarkComp):
    '''
    Plot a depth profile for all solution variables at fixed time.

    Parameters
    ----------
    df : DataFrame
        Contents of ASCII file from DiagenesisModel stored as pandas df.
    filename : STR
        Name of the data file w/o extension, the plot will be saved under the same name.
    benchmarkComp : BOOL
        Switch for optionally plotting the fig3e benchmark data. 

    Returns
    -------
    None.

    '''
    
    Xs = 131.9/0.1 # depth scaling constant
    x = df.x*Xs
    
    fig = plt.figure(figsize=(12,10))
    plt.plot(x,df.AR,label='AR')
    plt.plot(x,df.CA,label='CA')
    plt.plot(x,df.phi,label='phi')
    plt.plot(x,df.ca,label='Ca')
    plt.plot(x,df.co,label='CO')
    
    plotHeaviside(x/Xs)
    
    # if benchmark, plot the Fig3e data for comparison
    if (benchmarkComp):
        plotFig3e()
    
    plt.legend(loc='lower right')
    plt.xlabel('x (cm)')
    plt.ylabel('Concentrations')
    plt.xlim(0,500)
    plt.ylim(0,1.6)
    plt.savefig('%s.png'%(filename))
    plt.clf()
    
    
def plotTemporal(df, filename):
      '''
      Plot the time series at fixed depth for all solution variables from the 
      output of lheureux.f

      Parameters
      ----------
      df : DataFrame
          Contents of amarlx output file from lheureux.f stored as pandas df.
      filename : STR
          Name of the data file w/o extension, the plot will be saved under the same name.

      Returns
      -------
      None.

      '''
      Ts = 131.9/0.1**2 # time scaling constant
      t_plot = df.x*Ts/1000
      
      fig = plt.figure(figsize=(12,10))
      plt.plot(t_plot, df.AR, label=df.columns[1])
      plt.plot(t_plot, df.CA, label=df.columns[2])
      plt.plot(t_plot, df.phi, label=df.columns[5])
      plt.plot(t_plot, df.ca, label=df.columns[3])
      plt.plot(t_plot, df.co, label=df.columns[4])
      plt.legend()
      plt.xlim(0,np.array(t_plot)[-1])
      plt.ylim(0,1.4)
      plt.xlabel('t (ka)')
      plt.ylabel('Concentrations') 
      plt.savefig('%s.png'%(filename))
      plt.clf()  


def plotFig3e():
    # plot the digitized Figure 3e vals from L'Heureux (2018)
    # as an addition to a plot from code output
    
    # use default colour sequence from matplotlib to match data
    # order should be AR, CA, Po, Ca, Co
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    bm = pd.read_csv('fig3e.csv')
    plt.scatter(bm.ARX, bm.ARY, label='bm_AR',marker='x',color=colors[0])
    plt.scatter(bm.CAX, bm.CAY, label='bm_CA',marker='x',color=colors[1])
    plt.scatter(bm.PoX, bm.PoY, label='bm_phi',marker='x',color=colors[2])
    plt.scatter(bm.CaX, bm.CaY, label='bm_ca',marker='x',color=colors[3])
    plt.scatter(bm.CoX, bm.CoY, label='bm_co',marker='x',color=colors[4])


def plotHeaviside(x):
    # plot the function used to define the ADZ
    
    # for now we hard-code these parameter values
    ADZ_top = 50
    ADZ_bot = 150
    x_scale = 131.9/0.1
    
    smoothK = 500
    
    h = 0.5**2 * ( 1 + np.tanh(smoothK*(x - (ADZ_top/x_scale))))*\
                 ( 1 + np.tanh(smoothK*((ADZ_bot/x_scale) - x)))
    plt.plot(x*x_scale, h, label='Heaviside', color='black', linestyle='--')



##############################################################################

# switch on or off the overplotting of the fig3e data
plotBench = True

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





















