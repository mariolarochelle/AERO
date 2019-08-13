#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import sys
import os


# In[107]:


#raw_data = pd.read_csv("../../../Data/GRUPO DRAGON 2018/ech2/data/000960_Vaquerias_Jalisco_meas_2017.09.04-2017.10.03.txt", encoding = "utf8")
#raw_data = pd.read_csv("../../../Data/GRUPO DRAGON 2018/ech1 18092018/ech1 18092018.txt", encoding = "utf8")
#raw_data = pd.read_csv("../../../Data/GRUPO DRAGON 2018/ech2/data/001082__meas_2017.10.03-2018.01.23.txt", encoding = "utf8")

#raw_data = pd.read_csv("C:/Users/Franco/Documents/AnacondaProjects/Github/AERO/Data/Datos a limpiar/000960_Vaquerias,_Jalisco_meas_2017.09.04-2017.10.03.txt", encoding = "utf8")


# In[108]:


searchfor = ['timestamp', 'otherwordtomatch']

def search_timestamp(dataframe):
    
    index_matched = dataframe[dataframe[dataframe.columns[0]].str.replace(' ','').str.lower().str.contains('|'.join(searchfor))].index[0]
    
    columns_names = np.array(*dataframe[dataframe[dataframe.columns[0]].str.replace(' ','').str.lower().str.contains('|'.join(searchfor))][dataframe.columns[0]].str.split('\t'))
    #columns_names = np.array([col for col in columns_names if ('avg' in col.lower()) | ('time' in col.lower())])
    header_info = pd.DataFrame(dataframe[:index_matched])
    
    dataframe_splitted = dataframe[dataframe.columns[0]].str.split('\t', expand = True)
    
    final_dataframe = pd.DataFrame(dataframe_splitted[dataframe_splitted[dataframe_splitted[0].str.replace(' ','').str.lower().str.contains('timestamp')].index[0] +1:])
    final_dataframe.columns = columns_names
    final_dataframe.reset_index(drop=True, inplace=True)
    final_dataframe.iloc[:, 0] = pd.to_datetime(final_dataframe.iloc[:, 0])
    final_dataframe.replace('', np.NaN, inplace=True)
    final_dataframe.dropna(axis='columns', inplace = True)
    
    final_dataframe.iloc[:, 1:] = final_dataframe.iloc[:, 1:].astype(float)

    final_dataframe = final_dataframe.loc[:, [col for col in final_dataframe.columns if ('avg' in col.lower()) | ('time' in col.lower())]]
    
    final_dataframe.columns = [''.join(x.split('_', 2)[:2]) for x in final_dataframe.columns]
    
    return final_dataframe, header_info


# In[109]:


#a, b = search_timestamp(raw_data)


# In[27]:


#b.to_csv(r'pandas.txt', header=None, index=None, sep=' ', mode='a')


# In[111]:


if __name__ == "__main__":
    
    csv = sys.argv[1]
    file_name = os.path.basename(csv)
    print(csv)
    df = pd.read_csv(csv, header=None)
    df.head()
    df, header_df = search_timestamp(df)
    
    df.to_csv(f'data/{file_name}_data_cleaned.csv')
    header_df.to_csv(f'headers/{file_name}_header_info.txt', header=None, index=None, sep=' ', mode='a')
    


# In[ ]:




