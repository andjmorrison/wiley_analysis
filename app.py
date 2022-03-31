from fbprophet import Prophet
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st

def main():
    # title
    st.title('Wiley OA Publishing Analysis')

    # summary
    with st.expander('About This Project'):
        st.markdown(''' 

        This project aims to provide actionable intelligence for contract negotiation with Wiley. Analysis performed on article data to determine extent of open access availability, scope, and frequencies.

        #### Data Source:
        1. Article data from Web of Science
        2. Open Access availability from Unpaywall

        #### Limitations
        * No data on articles being potentially *removed* from open access catalogs. May need to employ web-scraping/API query or similar methodology to keep track of those data.
        * Nulls present in ID column for Dataset 1, reducing dataset upon merging.
        * No pricing information on journals.
        * Formatting -- although data pulled contain subject information, articles often are multidisciplinary, which creates difficulty in analysing trends for discrete research fields.

        #### Technologies
        * Pandas: Cleaning, Aggregation, Exploration
        * SciKit Learn: Regression, Basic Classification
        * Facebook 'Prophet': Time Series Analysis
        * Statsmodels API: Time Series Analysis and Projection
        * Plotly: Visualizations
        * Streamlit: Interface

        #### Analysis
        * [x]  Categorical
        * [x]  Time Series
        * [x]  Trend *and* seasonality
        * [x]  Time Series Analysis
        * [x]  Regression

        ''')

    # sidebar
    st.sidebar.title('Plots')
    st.sidebar.text('Categorical')
    plot_bar_oa = st.sidebar.checkbox('Open Access Categories', False)
    plot_bar_oaev = st.sidebar.checkbox('Open Access Evidence', True)

    st.sidebar.text('Time Series')
    plot_line_m = st.sidebar.checkbox('Articles Published by Month', False)
    plot_line_q = st.sidebar.checkbox('Articles Published by Quarter', False)
    option_yhat_sarimax = st.sidebar.checkbox('Articles Published Projection (SARIMAX)', False)
    option_prophet = st.sidebar.checkbox('Articles Published Projection (Prophet)', False)

    st.sidebar.text('Regression')
    reg = st.sidebar.checkbox('Regression Analyses', False)

    st.sidebar.text('Data')
    table = st.sidebar.checkbox('Show Data', False)

    # @st.cache
    def init():

        # files
        path_dir = '.'
        print(os.getcwd())
        files = [x for x in os.listdir(path_dir) if x.endswith('.csv')]
        print('files: ', files)
        file_oa = files[[idx for idx, s in enumerate(files) if 'OA' in s][0]]
        file_art = files[[idx for idx, s in enumerate(files) if 'Article' in s][0]]

        # df setup
        df_article = pd.read_csv(file_art)
        df_oa = pd.read_csv(file_oa)

        print(df_oa.keys())

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

        # st.balloons()

        return(df)

    # regressions
    def evaluate_model(model, col1, col2):
        X = col1
        y = col2

        X = [np.array([x]) for x in X]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model.fit(X_train_scaled, y_train)
        score = model.score(X_test_scaled, y_test)

        predictions = model.predict(X_test_scaled)

        return ({"model":f'{model}'.split('(')[0], 'score':score, 'predictions': predictions})

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
        st.markdown('##### Open Access Status')
        st.plotly_chart(fig_gb_oastat)
        st.markdown('---')
        st.markdown('#### Data')
        st.dataframe(df_groupby_oastatus, width=800)
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
            st.markdown('#### OA Evidence')
            st.plotly_chart(fig_gb_oaev)
            st.markdown('---')
            st.markdown('#### Data')
            st.dataframe(df_groupby_oa_ev, width=800)
            st.markdown('---')

    # groupby month and plot
    if plot_line_m:
        df_groupby_month = pd.DataFrame(df.groupby('Month')
            .agg({'doi':'count'})).rename(columns={'doi':'Article Count'})
        fig_gb_m = px.line(df_groupby_month, y="Article Count", title='Articles Published By Month')
        fig_gb_m.update_xaxes(tickangle=-90)

        # render
        st.markdown('#### Monthly')
        st.plotly_chart(fig_gb_m)
        st.markdown('---')
        st.markdown('#### Data')
        st.dataframe(df_groupby_month, width=800)
        st.markdown('---')

    # groupby q and plot
    if plot_line_q:
        df_groupby_quarter = pd.DataFrame(df.groupby('Quarter')
            .agg({'doi':'count'})).rename(columns={'doi':'Article Count'})
        fig_gb_q = px.line(df_groupby_quarter, y="Article Count", title='Articles Published By Quarter')
        
        # render
        st.markdown('##### Quarterly')
        st.plotly_chart(fig_gb_q)
        st.markdown('---')
        st.markdown('#### Data')
        st.dataframe(df_groupby_quarter, width=800)
        st.markdown('---')

    if option_yhat_sarimax:
        # groupby date
        df_groupby_daily = pd.DataFrame(df.groupby('published_date').agg({'doi':'count'}))

        # reset index
        df_groupby_daily.reset_index(inplace=True)

        # coerce to datetime
        df_groupby_daily['published_date'] = pd.to_datetime(df_groupby_daily['published_date'])

        # set index
        df_groupby_daily.set_index('published_date', inplace=True)

        # combine into monthly (end of month)
        y_sarimax = df_groupby_daily['doi'].resample('M').sum()
        # y = y.cumsum() (for testing with cumulative sum)
        data_sarimax = y_sarimax

        # fit 
        model = SARIMAX(data_sarimax, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
        model_fit = model.fit(disp=False)

        # make prediction -- new y
        yhat_sarimax = model_fit.predict(len(data_sarimax)-4, len(data_sarimax)+4)
        fig_sarimax = px.line(pd.DataFrame({'predicted':yhat_sarimax,'actual':data_sarimax}))
        fig_sarimax.update_layout(
            title='Predicted Open Access Journal Counts (New, Monthly)<br><i>SARIMAX</i>',
            xaxis_title='Monthly',
            yaxis_title='New OA Count',
            legend_title='Value'
        )
        st.markdown('#### Time Series Prediction')
        st.plotly_chart(fig_sarimax)

    if option_prophet:
        # groupby date
        df_groupby_daily = pd.DataFrame(df.groupby('published_date').agg({'doi':'count'}))

        # reset index
        df_groupby_daily.reset_index(inplace=True)

        # coerce to datetime
        df_groupby_daily['published_date'] = pd.to_datetime(df_groupby_daily['published_date'])

        # set index
        df_groupby_daily.set_index('published_date', inplace=True)

        # combine into monthly (end of month)
        y_prophet = df_groupby_daily['doi'].resample('M').sum()
        # y_prophet = y_prophet.cumsum() (for testing with cumulative sum)

        data_prophet = y_prophet.reset_index().rename(columns={'published_date':'ds','doi':'y'})

        data_prophet['y'] = data_prophet['y'] #.cumsum()

        future = pd.DataFrame(pd.date_range(data_prophet['ds'][50].date(),'2021-12-31',freq='M')).rename(columns={0:'ds'})
        model = Prophet()
        model.fit(data_prophet)
        predictions_all = model.predict(future)
        predictions = predictions_all[['ds','yhat_lower','yhat_upper','yhat']]

        #init fig
        fig_prophet = go.Figure()

        # actual
        fig_prophet.add_trace(go.Scatter(x=data_prophet['ds'], y=data_prophet['y'],
            fill=None,
            mode='lines',
            line_color='blue',
            name='Actual'))

        # yhat lower bounds
        fig_prophet.add_trace(go.Scatter(x=predictions['ds'], y=predictions['yhat_lower'],
            fill=None,
            mode='lines',
            line_color='indigo',
            name='Lower Bounds'))

        # yhat upper bounds
        fig_prophet.add_trace(go.Scatter(
            x=predictions['ds'],
            y=predictions['yhat_upper'],
            fill='tonexty', # fill area
            mode='lines', 
            line_color='indigo',
            opacity=.5,
            name='Upper Bounds'))

        # yhat
        fig_prophet.add_trace(go.Scatter(
            x=predictions['ds'],
            y=predictions['yhat'],
            fill=None,
            mode='lines', 
            line_color='red',
            name='Predicted'))

        fig_prophet.update_layout(
            title='Predicted Open Access Journal Counts (New, Monthly)<br><i>Prophet</i>',
            xaxis_title='Monthly',
            yaxis_title='New OA Count',
            legend_title='Value'
        )

        # show
        st.markdown('#### Time Series Prediciton')
        st.plotly_chart(fig_prophet)
        st.markdown('---')

    # regression
    if reg:
        # define models
        linregress = linear_model.LinearRegression()
        ridge = linear_model.Ridge()
        lasso = linear_model.Lasso(max_iter=2000, alpha=0.1)
        elastic = linear_model.ElasticNet(alpha=.001)
        logistic = linear_model.LogisticRegression(max_iter=2000)
        randomforest = RandomForestRegressor()
        extratrees = ExtraTreesRegressor()
        adaboost = AdaBoostRegressor()
        knn = KNeighborsRegressor()
        svr = SVR(C=1.0, epsilon=0.2)

        models = [
        linregress,
        ridge,
        lasso,
        elastic,
        logistic,
        randomforest,
        extratrees,
        adaboost,
        knn,
        svr
        ]

        results = []
        for model in models:
            results.append(evaluate_model(model, df['U2'], df['Z9']))
        pred_max = np.array([x['predictions'].max() for x in results]).max()

        # build figure
        fig_reg = go.Figure()

        for result in results:
        
            if result['score'] > 0.01:

                print(result['model'], result['score'])
                fig_reg.add_trace(go.Scatter(
                    x=df['U2'], 
                    y=result['predictions'],
                    opacity=0.5,
                    fill=None,
                    mode='markers',
                    name=result['model']))
            
        fig_reg.add_trace(go.Scatter(
                x=df['U2'], 
                y=df['Z9'],
                opacity=0.5,
                fill=None,
                mode='markers',
                marker = {'color':'black'},
                name='Actual',))

        fig_reg.update_layout(yaxis_range=[-15, 400], xaxis_range=[-15,400], title='U2 x Z9 Regressions<br><i>Sklearn</i>',)

        scores = [{'Model': x["model"], 'Score': round(x["score"], 6)} for x in results]

        fig = px.scatter(df[['U2','Z9']],x='U2',y='Z9',trendline='ols')
        fig.update_layout(title='U2 x Z9 OLS Linear Regression<br><i>Plotly</i>')

        # render
        st.markdown('#### Regressions')
        st.plotly_chart(fig_reg)
        st.dataframe(pd.DataFrame(scores))
        st.plotly_chart(fig)
        st.markdown('---')
        st.markdown('#### Data')
        st.dataframe(df[['U2','Z9']], width=800)
        st.markdown('---')

    # dataframe
    if table:
        # render
        st.markdown('#### Combined Dataset')
        st.dataframe(df, width=2000)
        st.markdown('---')

if __name__ == '__main__':
    main()