import pandas as pd 
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt 

import streamlit as st 
st.set_page_config('Boston housing')

st.markdown('# Prediction of boston housing')
import streamlit as st
import pandas as pd
# pipreqs --encoding=utf8
# git remote add origin https://github.com/AdbulrhmanEldeeb/Boston_housing_1


df = pd.read_csv('Housing.csv')
x=df.values[:,:-1]
y=df.values[:,-1]
st.sidebar.title("enter the information")
def inputs(): 
    crim = st.sidebar.slider("CRIM", df['CRIM'].min(), df['CRIM'].max())
    ZN = st.sidebar.slider("ZN", df['ZN'].min(), df['ZN'].max())
    INDUS = st.sidebar.slider("INDUS", df['INDUS'].min(), df['INDUS'].max())
    CHAS = st.sidebar.slider("CHAS", df['CHAS'].min(), df['CHAS'].max())
    NOX = st.sidebar.slider("NOX", df['NOX'].min(), df['NOX'].max())
    RM = st.sidebar.slider("RM", df['RM'].min(), df['RM'].max())
    AGE = st.sidebar.slider("AGE", df['AGE'].min(), df['AGE'].max())
    DIS = st.sidebar.slider("DIS", df['DIS'].min(), df['DIS'].max())
    RAD = st.sidebar.slider("RAD", df['RAD'].min(), df['RAD'].max())
    TAX = st.sidebar.slider("TAX", df['TAX'].min(), df['TAX'].max())
    PTRATIO = st.sidebar.slider("PTRATIO", df['PTRATIO'].min(), df['PTRATIO'].max())
    B = st.sidebar.slider("B", df['B'].min(), df['B'].max())
    LSTAT = st.sidebar.slider("LSTAT", df['LSTAT'].min(), df['LSTAT'].max())
    MEDV = st.sidebar.slider("MEDV", df['MEDV'].min(), df['MEDV'].max())

    inputs_dict = {'crim': crim, 'ZN': ZN, 'INDUS': INDUS, 'CHAS': CHAS, 'NOX': NOX, 'RM': RM, 'AGE': AGE, 
                   'DIS': DIS, 'RAD': RAD, 'TAX': TAX, 'PTRATIO': PTRATIO, 'B': B, 'LSTAT': LSTAT, 'MEDV': MEDV}
    inputs_df = pd.DataFrame(inputs_dict, index=[0])
    return inputs_df 

data=inputs()

from sklearn.ensemble import RandomForestRegressor 

regressor=RandomForestRegressor()
regressor.fit(x,y)
result=regressor.predict(data)
result=round(result[0],1)
st.markdown(f"MEDV is {result}")


