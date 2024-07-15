import matplotlib.pylab as plt
import numpy as np
import pandas as pd

def get_agri_daily_dataset(test_percent = 20):
    res = {}
    df = pd.read_csv('data/met_data_Kismacs_2013.csv').iloc[:-1]    
    df = df.astype({'év': 'int32','hónap': 'int32','nap': 'int32'})
    df2= df.groupby(['év','hónap','nap']).mean()    

    df2= df2.drop('óra',axis=1) 
    df2= df2.drop('perc',axis=1) 
    
    split_pos = int((len(df2.index)*(100-test_percent))/100)
    
    for col in df2.columns:
        traintests = (pd.Series(df2[col].iloc[:split_pos], dtype='float32'),pd.Series(df2[col].iloc[split_pos:], dtype='float32')) 
        res[col] = traintests
    return res

def get_agri_daily_dataset_whole():    
    df = pd.read_csv('data/met_data_Kismacs_2013.csv').iloc[:-1] 
    
    df = df.astype({'év': 'int32','hónap': 'int32','nap': 'int32'})
    df2= df.groupby(['év','hónap','nap']).mean()    

    df2= df2.drop('óra',axis=1) 
    df2= df2.drop('perc',axis=1) 

    return df2
    
