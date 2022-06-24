# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 13:52:07 2022
@author: yej
"""

# %% Import Packages
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

st.set_page_config(layout="wide",page_title='Portfolio Analyzer')
# %% Load Data
@st.cache(suppress_st_warning=True)
def grabDF(Path):
    df = pd.read_excel(Path)
    df.rename(columns={"Date_": "Date"}, inplace=True)
    df.set_index('Date', inplace=True)
    df= df*100
    st.balloons()
    return df

data = grabDF('./TFM Factors.xlsx')
port_name = list(data.columns)

# %% Set up the dahsboard framework
st.header('Portfolio Analyzer')
      
# %% Side Bar Selection

st.sidebar.header("Model Assumptions")
start = st.sidebar.date_input(
    "Pick start date", max(data.index) - timedelta(days=365))
end = st.sidebar.date_input('Pick end date', max(data.index))
interval = str(end - start).split(',')[0] 

# Default Value for betas
eq_beta = round(0.585633067489484, 4)
fi_beta = round(0.068545723609491, 4)
inf_beta = round(0.215263542655233, 4)

EQbeta = st.sidebar.slider("Pick the beta for Equity (EQ)",
                           min_value=-1.00, max_value=1.00,
                           step=0.001, value=eq_beta)

FIbeta = st.sidebar.slider("Pick the beta for Fixed Income (FI)",
                           min_value=-1.00, max_value=1.00, step=0.001,
                           value=fi_beta)

INFbeta = st.sidebar.slider("Pick the beta for Inflation (INF)",
                            min_value=-1.00, max_value=1.00, step=0.001,
                            value=inf_beta)

df = data.loc[start:end]
               

#%% Portfolio Calculation 
port_returns = df.copy()

for col in port_name:
    if col == 'EQ':
        port_returns['EQ_adj_returns'] = df[col]* EQbeta
    elif col == 'INF':
        port_returns['INF_adj_returns'] = df[col]*INFbeta
    else:
        port_returns['FI_adj_returns'] = df[col]*FIbeta

# Compute Total portfolio return
port_returns['Portfolio Weighted Return'] = port_returns['EQ_adj_returns'] + \
    port_returns['INF_adj_returns']+port_returns['FI_adj_returns']

df['Portfolio']  = port_returns['Portfolio Weighted Return']

# %% Volatility Calculation
port_std = np.std(df['Portfolio'])*(len(df) ** 0.5) 
EQ_std = np.std(df['EQ'])*(len(df) ** 0.5) 
INF_std = np.std(df['INF'])*(len(df) ** 0.5) 
FI_std = np.std(df['FI'])*(len(df) ** 0.5)
std_data = {'Asset': ['EQ', 'FI','INF', 'Portfolio'], 
            'Volatility':[EQ_std,INF_std,FI_std,port_std] }
vol_df = pd.DataFrame(std_data)
vol_df['Volatility'] = round(vol_df['Volatility'],2)
vol_df.set_index('Asset', inplace = True)
vol_df = vol_df.T


#%% KPI
st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
 
col11, col12, col13, col14 = st.columns(4)
col11.metric('Selected Time Interval', interval)
col12.metric("Equity Volatility","{:.2%}".format(EQ_std/100))
col13.metric("Inflation Volatility","{:.2%}".format(INF_std/100))
col14.metric("Fixed Income Volatility","{:.2%}".format(FI_std/100))

st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
   
# %% Asset Class Returns Graph + Export Function
st.subheader("Asset Class & Portfolio Return (%)")
returns_df= df.copy()
returns_df['Date'] = returns_df.index

returns_df = pd.melt(returns_df, id_vars=['Date'],value_vars=['EQ', 'FI','INF', 'Portfolio'] )
returns_df.columns = ['Date','Asset Class/Portfolio','Returns']
returns_df['Year-Mo'] = returns_df['Date'].dt.strftime("%Y-%b")
fig1 = px.bar(returns_df,
             x='Asset Class/Portfolio',
             y='Returns',
             labels={'Returns':'Returns (%)'},
             color='Asset Class/Portfolio',
             animation_frame='Year-Mo',
             animation_group='Asset Class/Portfolio',
             range_y=[min(returns_df['Returns']), max(returns_df['Returns'])],
           )


fig2 = px.line(df,
              y=['EQ', 'FI','INF', 'Portfolio'] ,
              x=df.index,
              range_y=(min(df.columns), max(df.columns)),
              range_x=(start, end))



col1, col2 = st.columns(2)
with col1:
    fig1.update_layout(autosize=True)
    st.plotly_chart(fig1)
    
with col2:
    fig2.update_layout(autosize=True)
    st.plotly_chart(fig2)


#%% Histogram


df.index=df.index.strftime('%Y-%m-%d')
fig_vol = px.histogram(df, 
                       x="Portfolio", 
                       labels={"Portfolio":'Portfolio Weighted Return (%)'},
                       nbins=25,text_auto=True)

fig_ac = px.histogram(df, 
                       x=['EQ','FI','INF'], 
                       labels={"Portfolio": 'Asset Class Return (%)'},
                       nbins=25,text_auto=True)
                       
col3, col4 =  st.columns([1,1])

with col3:
    st.subheader('Histogram of Portfolio Returns (%)')
    fig_vol.update_layout(autosize=True)
    st.plotly_chart(fig_vol)
with col4:
    st.subheader("Histogram of Asset Class Returns (%)")
    fig_ac.update_layout(autosize=True)
    st.plotly_chart(fig_ac)

stats_df = df.describe()
stats_df = stats_df.applymap("{0:.2f}".format)
df = df.applymap("{0:.2f}".format)

col5, col6 =  st.columns([1,1])
with col5:
    st.subheader('Asset Class & Portfolio Returns Data Summary')
    st.dataframe(stats_df)
    
with col6:
    st.subheader('Asset Class & Portfolio Returns Data (%)')
    st.dataframe(df)


st.balloons()



