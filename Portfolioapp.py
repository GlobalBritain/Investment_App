import streamlit as st 
# auto reload every module to reflect new changes
#%load_ext autoreload
# tell the system it is in auto reload. So it should keep autoreloading. Need to make change before it recognises change
#%autoreload 2
# This is important for creating modules. 
import edhec_risk_kit as ed
import numpy as np
import pandas as pd
import edhec_risk_kit as ed
import matplotlib.pyplot as plt
from pandas_datareader import data as wb
import Grab_assets as GA
import datetime
import plotly.express as px



# Data
Portfolio = GA.get_Etoro_rtns()
Portfolio_rtns = ed.annual_returns(Portfolio, 250) #Portfolio_stats(Portfolio, 0.75, 250).mean()
Portfolio_rtns = round(Portfolio_rtns.mean(),2)




st.sidebar.header("Options")

#page_options = 

#['Introduction', 'Investment growth', 'Portfolio Summary', 'Asset analytics']
# pages = st.sidebar.selectbox("Choose page", options=page_options)

#if pages == 'Introduction':
 #   st.header("Portfolio Analysis")
  #  st.text("All my portfolios and analysis will be managed from here")
    
#elif pages == 'Investment growth':
    
    
wealth_charts = st.beta_columns([3,2])

# container to hold one side of the chart
#wealth_growth = st.beta_container()

#with wealth_growth:

# Date choice
date_choice = st.beta_columns([2,2,6,6])

#Portfolio return
P_Rtn = st.sidebar.beta_columns(1)
# investment filters
filters = st.sidebar.beta_columns(1)   #([4,4,2.5,4])

# chart
charts = st.beta_columns([3,2])

# chart showing value of investment overtime
#investment_value = st.beta_columns([5,3])
# Invested_capital_column = st.beta_columns([1,5])
Invested_capital = wealth_charts[0].subheader("Net worth chart")

# Portfolio
portfolio = pd.DataFrame(columns=['Date', 'Regular Contribution', 'Return/Loss', 'Balance'])
# date
# start data
today = datetime.date.today()

# variables to choose data from
capital_insert = filters[0].number_input("Initial value", value=15000, min_value=0) # start investment value
Additional_contribution = filters[0].number_input("Regular contribution", value=1000) # regular contribution
#contribution_periods = filters[0].selectbox("Choose period", options = ['m', 'a']) # date type
period_of_investments = filters[0].number_input("Period to grow wealth", value=10, min_value=1, step=1) 
date_range = pd.date_range(start=today, periods=period_of_investments, freq='m') # date
Portfolio_choice = filters[0].selectbox("Portfolio Return Choice", options=['Global Britain Theme'])



if Portfolio_choice:
    P_Rtn[0].text("Portfolio Returns")
    Returns_p = P_Rtn[0].success(Portfolio_rtns*100)
    interest_rate = Portfolio_rtns
    
    investment = [Additional_contribution]*len(date_range)
    return_losses = []
    balances = []

    for date in date_range:
        current_return_loss = (interest_rate/12)*capital_insert
        return_losses.append(round(current_return_loss,2))
        balances.append(round(capital_insert + current_return_loss,2))
        capital_insert += (current_return_loss + Additional_contribution)

    portfolio['Date'] = pd.to_datetime(date_range).date
    portfolio['Regular Contribution'] = investment
    portfolio['Return/Loss'] = return_losses
    portfolio['Balance'] = balances

    portfolio.set_index('Date', inplace=True)

    balance_at_end = balances[-1]
    st.text("Final Portfolio Balance")
    st.success(balance_at_end)

    # Graph
    plotly_fig = px.line(data_frame=portfolio,x=portfolio.index,y=portfolio['Balance'], title="Portfolio Balance")
    #plotly_fig.layout.plot_bgcolor="white"
    st.plotly_chart(plotly_fig,use_container_width=True)

show_data_table = st.checkbox('Show Data Table')
if show_data_table:
    st.dataframe(portfolio.style.set_precision(2)) 

        
    


    # type of metric
    #capital_type = ["Stats", "Monetary amount"]
    #capital = wealth_charts[0].radio("Capital Type", capital_type)
    #if not capital == "Stats":
    
    #today = datetime.today()
    #datem = datetime(today.year, today.month, 1)
    #month = today.month
    #start_date = date_choice[0].date_input("Start date", min_value=today)
    
    
    
    
    # Need to create a new dataframe that measures total investment value
    #pd.DataFrame(data=capital_insert)


    # types of wealth
   # wealth = ('Total', 'Income', 'Profit')
    #wealth_charts[0].radio("Wealth segment", wealth)

    # net_worth_plot = date_choice[1].success(round(Portfolio_stats['Annual Returns'],3)*100)

    # risk container
    #risk_stats = st.beta_container()
    #with risk_stats:
    # Allocation category
        #allocation_col = st.beta_columns([5,3])
        #Allocation_capital = wealth_charts[1].subheader("Portfolio Risks")
    
    #Portfolio Summary
    #summary_stats = st.beta_columns(1)
    #summary_stats[0].header("Portfolio Statistics")
    
    # Chart showcasing investments by rating
    #investment_value = st.beta_columns(2)
    #investment_value[0].write("Investments by rating")
    #investment_value[1].write("Investments by risk rating")
    
#elif pages == 'Portfolio Summary':
 #   st.header("Portfolio Summary")
  #  st.write("Summary of all portfolios we are invested in")
    
    # Store all the below, before chart, in a component
   # Quick_stats_summary = st.beta_container()
    
   # with Quick_stats_summary:
    #    filter_options = st.beta_columns(2)
     #   test = ['UK as the enemy part']
      #  filter_options[0].selectbox("Choose Theme", options=test)
       # Portfolios = ["Tech and manufacturing"]
      #  filter_options[1].selectbox("Choose Portfolio", options=Portfolios)

   #     timeframe = ('All','1m', '3m','6m','1y','2y','3y','5y')
    #    time_period = st.beta_columns([3,6,4,7])
     #   time_period[3].radio("Time period", timeframe) #, index=1)
      #  st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

      #  important_stats = st.beta_columns([4,3,3,3])
        # Capital amount (what portfolio is worth since inception)
#        important_stats[0].markdown("#### Total Net Worth")            
 #       important_stats[1].markdown("#### Total Invested Capital")
  #      important_stats[2].markdown("#### Total Investments")
   #     important_stats[3].markdown("#### Total Rate of Return")

    