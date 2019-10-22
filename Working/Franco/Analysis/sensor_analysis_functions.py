#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (14, 8)
plt.style.use('ggplot')

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
np.set_printoptions(threshold=sys.maxsize)

import math
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import matplotlib.pyplot as plt
import re
import itertools
import os
from tqdm import tqdm
from operator import itemgetter
from itertools import *

import plotly.offline as pyo
import plotly_express as px
import plotly.graph_objs as go
import glob


# In[1]:


def fault_detection(df, sensor='anem', correlation_window = 10, ratio_th = 10, correlation_th = .7, diff_th = .1):
    """ Función construir estadísticas de los datos
    
        df: dataframe
        sensor: 'anem' anemometer, 'vane' windvane
        func: 'correlation', 'ratio', 'diff', 'all'
        correlation_window: default=10. Ventana de correlación

        return: Dataframe con las stats nuevas
    """
    if sensor.lower() == 'anem':
        func = ['correlation', 'ratio']
    elif sensor.lower() == 'vane':
        func = ['diff']
    
    
    str_sensor = ['timestamp']
    str_sensor.append(sensor)
    
    ch_sensors = [x for x in df.columns if any(i.lower() in x.lower() for i in str_sensor)]
    df_sensors = df.loc[:, ch_sensors]

    if sensor == 'vane':
        func_column = '_deg_cos'
        for col in df_sensors.columns[~df_sensors.columns.str.contains('Timestamp')]:
            df_sensors[str(col) + '-cos*'] = np.sin(np.deg2rad(df_sensors[col])) 
    
    
    for x in tqdm(list(itertools.combinations(df_sensors.columns[~df_sensors.columns.str.contains('Timestamp')], 2))):
        
        # Fractions
        if ('ratio' in func) or ('all' in func):
            ratio_col_name = str(x[0] + 'VS' + x[1] + '_ratio*')
            df_sensors[ratio_col_name] = df_sensors[x[0]] / df_sensors[x[1]]
            
            df_sensors[ratio_col_name + '_anomaly'] = ''
            df_sensors.loc[df_sensors[ratio_col_name] >= ratio_th, ratio_col_name + '_anomaly'] = 1
            df_sensors.loc[df_sensors[ratio_col_name] < ratio_th, ratio_col_name + '_anomaly'] = 0
            #df_sensors.loc[df_sensors[ratio_col_name] < 0.4, ratio_col_name + '_anomaly'] = 1
            
        # Rolling correlations           
        if ('correlation' in func) or ('all' in func):
            correlation_col_name = str(x[0] + 'VS' + x[1] + '_correlation*')
            
            #rolling_correlation = df_sensors[x[0]].rolling(correlation_window).corr(df_sensors[x[1]].rolling(correlation_window)).shift(2)
            rolling_correlation = [np.corrcoef(df_sensors.loc[idx-correlation_window:idx, x[0]].values, df_sensors.loc[idx-correlation_window:idx, x[1]].values)[0][1] for idx, _ in df_sensors.iterrows()]
            df_sensors[correlation_col_name] = rolling_correlation
            df_sensors[correlation_col_name + '_anomaly'] = ''
            df_sensors.loc[df_sensors[correlation_col_name] <= correlation_th, correlation_col_name + '_anomaly'] = 1
            df_sensors.loc[df_sensors[correlation_col_name] > correlation_th, correlation_col_name + '_anomaly'] = 0
            
            
            
            

        # Difference
        if ('diff' in func) or ('all' in func):
            if '-cos' in x[0] and '-cos' in x[1]:
                diff_col_name = str(x[0] + 'VS' + x[1] + '_diff*')
                df_sensors[diff_col_name] = df_sensors[x[0]] - df_sensors[x[1]]
                df_sensors[diff_col_name + '_anomaly'] = ''
                df_sensors.loc[df_sensors[diff_col_name] >= diff_th, diff_col_name + '_anomaly'] = 1
                df_sensors.loc[df_sensors[diff_col_name] < diff_th, diff_col_name + '_anomaly'] = 0
                df_sensors.loc[df_sensors[diff_col_name] <= -diff_th, diff_col_name + '_anomaly'] = 1                
    
    return df_sensors


# In[7]:


def anemometer_identification(df_stats, redundant_sensors, ratio_th = 10, correlation_th = .7, window = 10):
    
    df_to_return = df_stats.copy()
    df_stast = df_stats.copy()
    

    df_stats.replace('', 0, inplace=True)
    
    df_anomaly_column = df_stats[df_stats.columns[df_stats.columns.str.contains('anomaly')]]
    df_anomaly = df_anomaly_column[(df_anomaly_column > 0).any(axis=1)]
    
    for idx, row in tqdm(df_anomaly.iterrows(), total=df_anomaly.shape[0]):

        keys = np.array(list(row[row>0].keys().str.split('VS|_')))
        keys = keys.T[:2].ravel()

        ch_anem_columns = np.unique(keys)
        keys = list(set([x for x in keys if list(keys).count(x) > 2]))
        
        ch_anem_to_join = []
        for col in ch_anem_columns:
            if df_to_return.loc[idx, col] > 70:
                ch_anem_to_join.append(col)
        
        def_keys = []
        for k in keys:
            try:
                if (df_to_return.loc[idx, k+'VS'+redundant_sensors[k]+'_correlation*_anomaly'] == 1 |
                    df_to_return.loc[idx, k+'VS'+redundant_sensors[k]+'_ratio*_anomaly'] == 1):
                    def_keys.append(k)
            except:
                pass
            
        df_to_return.loc[idx, 'broken?'] = ','.join(np.unique(def_keys+ch_anem_to_join))    


    
    
    df_to_return.loc[df_to_return['broken?'] == '', 'broken?'] = 'None'
    df_to_return.fillna('None', inplace=True)
    return df_to_return.loc[:, ['Timestamp', 'Ch1Anem', 'Ch2Anem', 'Ch3Anem', 'Ch4Anem',
                                'Ch5Anem', 'Ch6Anem', 'broken?']]


# In[3]:


def plotting_anem_parameter_tunning(df, ratio_range=30, correlation_range=10):
    
    df_to_plot = df[df.columns[df.columns.str.contains('Anem')]]
    
    df_intersection = pd.DataFrame()
    df_ratio = pd.DataFrame()
    df_correlation = pd.DataFrame()
    
    
    idx = 0
    for ratio in tqdm(range(2, ratio_range)):
        for correlation in range(0, correlation_range):
            correlation = round(correlation *.1, 2)
            df_to_plot_stats = fault_detection(df_to_plot, sensor='anem', correlation_window=10, ratio_th = ratio, correlation_th = correlation)
        
            ratio_cols = df_to_plot_stats.columns[df_to_plot_stats.columns.str.contains('_ratio\*_anomaly')]
            correlation_cols = df_to_plot_stats.columns[df_to_plot_stats.columns.str.contains('_correlation\*_anomaly')]
            
            
            count_intersection = []
            for x in range(len(ratio_cols)):
                count_intersection.append(len(df_to_plot_stats.loc[(df_to_plot_stats[ratio_cols[x]] == 1) & 
                                                                   (df_to_plot_stats[correlation_cols[x]] == 1), :]))
    
            count_intersection = sum(count_intersection)         
            df_intersection.loc[idx, 'cant'] = count_intersection
            df_intersection.loc[idx, 'ratio'] = ratio
            df_intersection.loc[idx, 'correlation'] = correlation
            df_intersection.name = 'Intersection'
        
            count_ratio = []
            count_ratio.append([len(df_to_plot_stats.loc[df_to_plot_stats[col] == 1, col]) for col in ratio_cols])
            count_ratio = sum(*count_ratio)
            df_ratio.loc[idx, 'cant'] = count_ratio
            df_ratio.loc[idx, 'ratio'] = ratio
            df_ratio.loc[idx, 'correlation'] = correlation
            df_ratio.name = 'Ratio'
            
            count_correlation = []
            count_correlation.append([len(df_to_plot_stats.loc[df_to_plot_stats[col] == 1, col]) for col in correlation_cols])
            count_correlation = sum(*count_correlation)
            df_correlation.loc[idx, 'cant'] = count_correlation
            df_correlation.loc[idx, 'ratio'] = ratio
            df_correlation.loc[idx, 'correlation'] = correlation
            df_correlation.name = 'Correlation'
            
            idx = idx+1
            
    dfs = [df_intersection, df_ratio, df_correlation]
    
    for df in dfs:
        try:
            z = np.log10(df['cant'])
        except:
            z = 0
        y = df['ratio']
        x = df['correlation']

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')


        ax.scatter(x, y, z, c='r', marker='o')

        ax.set_xlabel('correlation')
        ax.set_ylabel('ratio')
        ax.set_zlabel('cant')

        plt.title(df.name)

        plt.show()


# In[5]:


def plot_sensors(df, index_start, index_finish, sensors=[]):
    
    time_col = df.columns[df.columns.str.contains('Time')]
    
    df_to_plot = df.loc[index_start - 20:index_finish+50, sensors]
    df_to_plot['step'] = df.loc[index_start - 20:index_finish+50, time_col]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_to_plot['step'], y=df_to_plot[sensors[0]],
                        mode='lines+markers',
                        name=str(sensors[0])))

    fig.add_trace(go.Scatter(x=df_to_plot['step'], y=df_to_plot[sensors[1]],
                        mode='lines+markers',
                        name=str(sensors[1])))

    fig.layout.update(
        title=go.layout.Title(
            text='Start: '+ str(df_to_plot.loc[index_start, 'step']) +' Ends: '+ str(df_to_plot.loc[index_finish, 'step']),
    ))
    
    #fig.show(=)    
    return fig


# In[4]:


def get_index_list(df, sensor):
    df.loc[df['broken?'].str.contains(sensor), sensor + '_index'] = 1
    df.loc[~df['broken?'].str.contains(sensor), sensor + '_index'] = 0

    indexes = df.loc[df[sensor + '_index'] == 1, :].index

    return indexes


# In[5]:


def split_kneighbor_indexes(number_list, k):
    
    groups = []
    start = 0
    
    for idx, val in enumerate(number_list[1:], start = 1):
        if number_list[idx] - number_list[idx-1]  > k:
            stop = idx
            groups.append(number_list[start:stop])
            start = stop + 1
        else:
            pass
        
    groups = [x for x in groups if x.size > 5]
    return groups


# In[2]:


def generate_dashboard(df, redundance_dict={}):
    anem_columns = df.columns[df.columns.str.contains('Anem')]
    
    anem_dict = {}
    for col in anem_columns:
        indexes = get_index_list(df, col) 
        anem_dict[col] = split_kneighbor_indexes(indexes, 12)
    
    
    dir_name = str(df.name)
    try:
        os.makedirs(dir_name)
    except:
        print("Already exists...")
        
    for key, values in anem_dict.items():
        
        html = '<html><body>' 
        try:
            os.makedirs(f'anomaly_html/{dir_name}/{key}')
        except:
            print("Already exists...")
            
        for value in values:
            fig = plot_sensors(df, value[0], value[-1], sensors=[key, redundance_dict[key]])
            fig.write_html(f'anomaly_html/{dir_name}/{key}/{key}-{value[0]}.html', full_html=False)   
            
        cont = 0
        for file in tqdm(sorted(glob.glob(f"anomaly_html/{dir_name}/{key}/*.html"), key=os.path.getmtime)):
            html += '<div><h3> Anomalia: '+ str(cont+1) +' en el sensor: '+ key +'</h3></div>'
            f = open(file, 'r', encoding='utf-8')
            f = f.read()
            html += str(f)
            cont +=1
        html += '</body></html>'

        Html_file = open(f"{dir_name}/filename_{key}.html","w")
        Html_file.write(html)
        Html_file.close()


# In[ ]:


def anem_parameter_tunning(df, ch_anem, ratio_range=30, correlation_range=10):
    
    df_to_tunning = df[df.columns[df.columns.str.contains('Anem')]].copy()
    
    
    
    idx = 0
    
    for ratio in tqdm(range(2, ratio_range)):
        for correlation in range(1, correlation_range):
            correlation = round(correlation *.1, 2)
            
            df_fault_detection = fault_detection(df_to_tunning, sensor='anem', correlation_window=10, ratio_th=ratio, correlation_th=correlation)
            
            df_anem_filtered = anemometer_identification(df_vaquerias_fault_detection)
            
            indexes = get_index_list(df_anem_filtered, ch_anem)
            
            anem_dict = {}
            #anem_dict[ch_anem] = split_kneighbor_indexes(indexes, 12)
            print(len(split_kneighbor_indexes(indexes, 12)))           


# In[ ]:


def anem_parameter_tunning(df, ratio_range=30, correlation_range=10):
    
    df_to_plot = df[df.columns[df.columns.str.contains('Anem')]]
    
    df_intersection = pd.DataFrame()
    df_ratio = pd.DataFrame()
    df_correlation = pd.DataFrame()
    
    
    idx = 0
    for ratio in tqdm(range(2, ratio_range)):
        for correlation in range(0, correlation_range):
            correlation = round(correlation *.1, 2)
            df_to_plot_stats = fault_detection(df_to_plot, sensor='anem', correlation_window=10, ratio_th = ratio, correlation_th = correlation)
        
            ratio_cols = df_to_plot_stats.columns[df_to_plot_stats.columns.str.contains('_ratio\*_anomaly')]
            correlation_cols = df_to_plot_stats.columns[df_to_plot_stats.columns.str.contains('_correlation\*_anomaly')]
            
            
            count_intersection = []
            for x in range(len(ratio_cols)):
                count_intersection.append(len(df_to_plot_stats.loc[(df_to_plot_stats[ratio_cols[x]] == 1) & 
                                                                   (df_to_plot_stats[correlation_cols[x]] == 1), :]))
    
            count_intersection = sum(count_intersection)         
            df_intersection.loc[idx, 'cant'] = count_intersection
            df_intersection.loc[idx, 'ratio'] = ratio
            df_intersection.loc[idx, 'correlation'] = correlation
            df_intersection.name = 'Intersection'
        
            count_ratio = []
            count_ratio.append([len(df_to_plot_stats.loc[df_to_plot_stats[col] == 1, col]) for col in ratio_cols])
            count_ratio = sum(*count_ratio)
            df_ratio.loc[idx, 'cant'] = count_ratio
            df_ratio.loc[idx, 'ratio'] = ratio
            df_ratio.loc[idx, 'correlation'] = correlation
            df_ratio.name = 'Ratio'
            
            count_correlation = []
            count_correlation.append([len(df_to_plot_stats.loc[df_to_plot_stats[col] == 1, col]) for col in correlation_cols])
            count_correlation = sum(*count_correlation)
            df_correlation.loc[idx, 'cant'] = count_correlation
            df_correlation.loc[idx, 'ratio'] = ratio
            df_correlation.loc[idx, 'correlation'] = correlation
            df_correlation.name = 'Correlation'
            
            idx = idx+1
            


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




