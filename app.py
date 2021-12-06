import os
import pandas as pd
import plotly.express as px
import streamlit as st

# title
st.title('Wiley OA Publishing Analysis')

# summary
st.markdown(''' 

This project aims to provide actionable intelligence for contract negotiation with Wiley. Analysis performed on article data to determine extent of open access availability, scope, and frequencies.

#### Data Source:
1. Article data from Web of Science
2. Open Access availability from Unpaywall

#### Limitations
* Nulls present in dataset dataset 1's id column
* No pricing information on journals
* Formatting -- although data pulled contain 

#### Analysis
* [x]  Trend versus seasonality
* [x]  Trend *and* seasonality
* [x]  Categorical
-- ML
* [ ]  Regression
* [ ]  Classification
* [ ]  Feature Analysis

''')

# sidebar
st.sidebar.title('Plots')
st.sidebar.text('Seasonality')
plot_line_m = st.sidebar.checkbox('Articles Published by Month', False)
plot_line_q = st.sidebar.checkbox('Articles Published by Quarter', False)

st.sidebar.text('Categorical')
plot_bar_oa = st.sidebar.checkbox('Open Access Categories', False)
plot_bar_oaev = st.sidebar.checkbox('Open Access Evidence', True)
table = st.sidebar.checkbox('Show Data', False)

# @st.cache
def init():

    # files
    path_dir = '.'
    files = [x for x in os.listdir(path_dir) if x.endswith('.csv')]

    # df setup
    df_article = pd.read_csv(files[0])
    df_oa = pd.read_csv(files[1])

    # perform merge on DOI
    df = pd.merge(left=df_article, right=df_oa, 
            left_on='DI', right_on='doi',
            how='inner')

    # date coersion
    df['published_date'] = pd.to_datetime(df['published_date'])
    df['published_date'] = df['published_date'].dt.date
    
    # quarter
    quarter = pd.PeriodIndex(df['published_date'], freq='Q')
    df['Quarter'] = [f'{x.year}Q{x.quarter}' for x in quarter]

    # month
    month_key = {
        '1':'Jan',
        '2':'Feb',
        '3':'Mar',
        '4':'Apr',
        '5':'May',
        '6':'Jun',
        '7':'Jul',
        '8':'Aug',
        '9':'Sept',
        '10':'Oct',
        '11':'Nov',
        '12':'Dec',
    }
    month = pd.PeriodIndex(df['published_date'], freq='M')
    df['Month'] = [f'{x.year}, {month_key[str(x.month)]}' for x in month]

    st.balloons()

    return(df)

#init
df=init()


# groupby OA status
if plot_bar_oa:
    df_groupby_oastatus = df.groupby(df['oa_status']).agg({'doi':'count'})
    df_groupby_oastatus.reset_index(inplace=True)
    df_groupby_oastatus.rename(
        columns={'doi':'Article Count', 'oa_status': 'Open Access Status'}, inplace=True)
    df_groupby_oastatus['Open Access Status'] = df_groupby_oastatus['Open Access Status'].str.capitalize()
    fig_gb_oastat = px.bar(df_groupby_oastatus, x='Open Access Status', y="Article Count", title='Articles Published By OA Status')
    
    # render
    st.markdown('# Open Access Status')
    st.plotly_chart(fig_gb_oastat)
    st.markdown('---')
    st.markdown('#### Data')
    st.dataframe(df_groupby_oastatus, width=800)
    st.markdown('---')

# groupby q and plot
if plot_line_q:
    df_groupby_quarter = pd.DataFrame(df.groupby('Quarter')
        .agg({'doi':'count'})).rename(columns={'doi':'Article Count'})
    fig_gb_q = px.line(df_groupby_quarter, y="Article Count", title='Articles Published By Quarter')
    
    # render
    st.markdown('# Quarterly')
    st.plotly_chart(fig_gb_q)
    st.markdown('---')
    st.markdown('#### Data')
    st.dataframe(df_groupby_quarter, width=800)
    st.markdown('---')

# groupby month and plot
if plot_line_m:
    df_groupby_month = pd.DataFrame(df.groupby('Month')
        .agg({'doi':'count'})).rename(columns={'doi':'Article Count'})
    fig_gb_m = px.line(df_groupby_month, y="Article Count", title='Articles Published By Month')
    fig_gb_m.update_xaxes(tickangle=-90)

    # render
    st.markdown('# Monthly')
    st.plotly_chart(fig_gb_m)
    st.markdown('---')
    st.markdown('#### Data')
    st.dataframe(df_groupby_month, width=800)
    st.markdown('---')

if plot_bar_oaev:
    df['OA Evidence'] = df['best_oa_evidence'].str.rsplit('(', expand=True)[0]
    df['Via'] = df['best_oa_evidence'].str.rsplit('(', expand=True)[1].str.replace(')','')
    df['Via'] = df['Via'].str.replace('via','')

    df_groupby_oa_ev = pd.DataFrame(df.groupby([df['OA Evidence'],df['Via']]).agg({'doi':'count'}))
    df_groupby_oa_ev.reset_index(inplace=True)
    df_groupby_oa_ev.rename(columns={0:'OA Evidence', 1:'Via', 'doi':'Article Count'}, inplace=True)
    fig_gb_oaev = px.bar(df_groupby_oa_ev, x='OA Evidence', y="Article Count", color='Via', title='Articles with OA Evidence')
    fig_gb_oaev.update_xaxes(tickangle=-90)
    
    # render
    st.markdown('# OA Evidence')
    st.plotly_chart(fig_gb_oaev)
    st.markdown('---')
    st.markdown('#### Data')
    st.dataframe(df_groupby_oa_ev, width=800)
    st.markdown('---')


# dataframe
if table:
    # render
    st.markdown('#### Combined Dataset')
    st.dataframe(df, width=2000)
    st.markdown('---')