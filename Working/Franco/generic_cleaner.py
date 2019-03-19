#!/usr/bin/env python
# coding: utf-8

# In[116]:


import pandas as pd
import numpy as np
import sys
import os


# In[95]:


def search_timestamp(dataframe):
    dataframe[dataframe[dataframe.columns[0]].str.replace(' ','').str.lower().str.contains('timestamp')]
    columns_names = np.array(*dataframe[dataframe[dataframe.columns[0]].str.replace(' ','').str.lower().str.contains('timestamp')][dataframe.columns[0]].str.split('\t'))

    dataframe_splitted = dataframe[dataframe.columns[0]].str.split('\t', expand = True)
    
    
    final_dataframe = pd.DataFrame(dataframe_splitted[dataframe_splitted[dataframe_splitted[0].str.replace(' ','')                                                                         .str.lower()                                                                         .str.contains('timestamp')].index[0] +1:])
    final_dataframe.columns = columns_names
    final_dataframe.reset_index(drop=True, inplace=True)

    final_dataframe.iloc[:, 0] = pd.to_datetime(final_dataframe.iloc[:, 0])
    final_dataframe.set_index(final_dataframe.iloc[:, 0], inplace=True)
    final_dataframe.replace('', np.NaN, inplace=True)
    final_dataframe.dropna(axis='columns', inplace = True)
    
    return final_dataframe


# In[115]:


if __name__ == "__main__":
    
    csv = sys.argv[1]
    file_name = os.path.basename(csv)
    print(csv)
    df = pd.read_csv(csv)
    df = search_timestamp(df)
    
    df.to_csv(f'{file_name}.csv')


# In[ ]:




