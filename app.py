# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 13:52:07 2022

@author: yej
"""

import streamlit as st
import pandas as pd
st.header('Title of site')
st.write('it is working!')
st.button("OK")

sp500 = pd.read_csv('StockData/SP500.csv')

start= st.sidebar.date_input("Pick start date")
end = st.sidebar.date_input('Pick end date')
