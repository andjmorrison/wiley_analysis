import json
# import matplotlib
# import matplotlib.pyplot as plt
import os
import pandas as pd
# import plotly.express as px
import streamlit as st

# matplotlib.style.use('fivethirtyeight')

path_dir = '.'
files = [x for x in os.listdir(path_dir) if x.endswith('.csv')]

df_article = pd.read_csv(files[0])
df_oa = pd.read_csv(files[1])

doi = df_article['DI'].dropna().values
di = df_oa['doi'].dropna().values

# perform merge on DOI
df = pd.merge(left=df_article, right=df_oa, 
         left_on='DI', right_on='doi',
         how='inner')

df['published_date'] = pd.to_datetime(df['published_date'])
df['published_date'] = df['published_date'].dt.date
quarter = pd.PeriodIndex(df['published_date'], freq='Q')
df['quarter'] = [f'{x.year}Q{x.quarter}' for x in quarter]


# st.bar_chart(df.groupby(df['oa_status']).agg({'doi':'count'}))

df2 = pd.DataFrame(df.groupby('quarter').agg({'doi':'count'}))
st.line_chart(df2)