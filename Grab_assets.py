# Copy new columns - https://stackoverflow.com/questions/18674064/how-do-i-insert-a-column-at-a-specific-column-index-in-pandas

# Source https://algotrading101.com/learn/yahoo-finance-api-guide/

import pandas as pd
from yahoo_fin.stock_info import get_data
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import edhec_risk_kit as ed
import numpy as np
import pytz # timezone fo your own data
from collections import OrderedDict


# Chemicals portfolio
def chemicals_tickr():
    tckr_list = ['CRDA.L', 'JMAT.L', 'VCT.L', 'TET.L', 'ZTF.L', 'VRS.L', 'DCTA.L', 'SCPA.L', 'HAYD.L',
                 'AGM.L', 'IOF.L', 'ITX.L', 'HDD.L','SCT.L','AVV.L']
    return tckr_list

def get_Chem_prices():
    #Get the data
    ticker_list = ['CRDA.L', 'JMAT.L', 'VCT.L', 'TET.L', 'ZTF.L', 'VRS.L', 'DCTA.L', 'SCPA.L', 'HAYD.L',
                 'AGM.L', 'IOF.L', 'ITX.L', 'HDD.L', 'SCT.L', 'AVV.L']
    historical_datas = {}
    Portfolio = pd.DataFrame(historical_datas)
    for ticker in ticker_list:
        Portfolio[ticker] = get_data(ticker,  start_date="01/01/2009",  index_as_date = True, interval="1d")['adjclose']
        Portfolio.index.tz_localize('UTC')
    P_Rename = Portfolio.rename(columns = {'CRDA.L': 'Croda International Price',
                                          'JMAT.L': 'Johnson Matthey Plc Price', 
                                          'VCT.L': 'Victrex Price', 
                                          'TET.L': 'Treatt Price',
                                          'ZTF.L': 'Zotefoams Price' , 
                                          'VRS.L': 'Versarien Price', 
                                          'DCTA.L': 'Directa Plus Price', 
                                          'SCPA.L': 'Scapa Group Price' , 
                                          'HAYD.L': 'Haydale Graphene Price',
                                          'AGM.L': 'Applied Graphene Price', 
                                          'IOF.L': 'Iofina Price', 
                                          'ITX.L': 'Itaconix Price', 
                                          'HDD.L': 'Hardide Price',
                                          'SCT.L': 'Softcat Price',
                                          'AVV.L': 'Aveva Group Price'
                                             })
    return P_Rename


# Alternative
def Chem_rtns_alt():
    ticker_list = ['CRDA.L', 'JMAT.L', 'VCT.L', 'TET.L', 'ZTF.L', 'VRS.L', 'DCTA.L', 'SCPA.L', 'HAYD.L',
                 'AGM.L', 'IOF.L', 'ITX.L', 'HDD.L','SCT.L', 'AVV.L']
    
    # Get the data from yahoo finance
    Assets = wb.DataReader(ticker_list, data_source='yahoo', start='2009-01-01')['Adj Close']
    Returns = Assets.pct_change() 
    P_Clean = Returns.fillna(0)
    P_Rename = P_Clean.rename(columns = {'CRDA.L': 'Croda International Return',
                                          'JMAT.L': 'Johnson Matthey Plc Returns', 
                                          'VCT.L': 'Victrex Returns', 
                                          'TET.L': 'Treatt Returns',
                                          'ZTF.L': 'Zotefoams Returns' , 
                                          'VRS.L': 'Versarien Returns', 
                                          'DCTA.L': 'Directa Plus Returns', 
                                          'SCPA.L': 'Scapa Group Returns' , 
                                          'HAYD.L': 'Haydale Graphene Returns',
                                          'AGM.L': 'Applied Graphene Returns', 
                                          'IOF.L': 'Iofina Returns', 
                                          'ITX.L': 'Itaconix Returns', 
                                          'HDD.L': 'Hardide Returns',
                                          'SCT.L': 'Softcat Returns',
                                          'AVV.L': 'Aveva Group Returns'
                                             }, inplace=True)
    return P_Rename  


def get_Chem_rtns():
    #Get the data
    ticker_list = ['CRDA.L', 'JMAT.L', 'VCT.L', 'TET.L', 'ZTF.L', 'VRS.L', 'DCTA.L', 'SCPA.L', 'HAYD.L',
                 'AGM.L', 'IOF.L', 'ITX.L', 'HDD.L', 'SCT.L', 'AVV.L']
    historical_datas = {}
    Portfolio = pd.DataFrame(historical_datas)
    for ticker in ticker_list:
        Portfolio[ticker] = get_data(ticker,  start_date="01/01/2009",  index_as_date = True, interval="1d")['adjclose']
    
    #Get returns 
    Returns = Portfolio.pct_change()
    P_Clean = Returns.fillna(0)
    P_Rename = P_Clean.rename(columns = {'CRDA.L': 'Croda International Return',
                                          'JMAT.L': 'Johnson Matthey Plc Returns', 
                                          'VCT.L': 'Victrex Returns', 
                                          'TET.L': 'Treatt Returns',
                                          'ZTF.L': 'Zotefoams Returns' , 
                                          'VRS.L': 'Versarien Returns', 
                                          'DCTA.L': 'Directa Plus Returns', 
                                          'SCPA.L': 'Scapa Group Returns' , 
                                          'HAYD.L': 'Haydale Graphene Returns',
                                          'AGM.L': 'Applied Graphene Returns', 
                                          'IOF.L': 'Iofina Returns', 
                                          'ITX.L': 'Itaconix Returns', 
                                          'HDD.L': 'Hardide Returns',
                                          'SCT.L': 'Softcat Returns',
                                          'AVV.L': 'Aveva Group Returns'
                                             })
    return P_Rename  


def get_Chem_5y_rtns():
    #Get the data
    ticker_list = ['CRDA.L', 'JMAT.L', 'VCT.L', 'TET.L', 'ZTF.L', 'VRS.L', 'DCTA.L', 'SCPA.L', 'HAYD.L',
                 'AGM.L', 'IOF.L', 'ITX.L', 'HDD.L', 'SCT.L', 'AVV.L']
    historical_datas = {}
    Portfolio = pd.DataFrame(historical_datas)
    for ticker in ticker_list:
        Portfolio[ticker] = get_data(ticker,  start_date="01/01/2016",  index_as_date = True, interval="1d")['adjclose']
    
    #Get returns 
    Returns = Portfolio.pct_change()
    P_Clean = Returns.fillna(0)
    P_Rename = P_Clean.rename(columns = {'CRDA.L': 'Croda International Return',
                                          'JMAT.L': 'Johnson Matthey Plc Returns', 
                                          'VCT.L': 'Victrex Returns', 
                                          'TET.L': 'Treatt Returns',
                                          'ZTF.L': 'Zotefoams Returns' , 
                                          'VRS.L': 'Versarien Returns', 
                                          'DCTA.L': 'Directa Plus Returns', 
                                          'SCPA.L': 'Scapa Group Returns' , 
                                          'HAYD.L': 'Haydale Graphene Returns',
                                          'AGM.L': 'Applied Graphene Returns', 
                                          'IOF.L': 'Iofina Returns', 
                                          'ITX.L': 'Itaconix Returns', 
                                          'HDD.L': 'Hardide Returns',
                                          'SCT.L': 'Softcat Returns',
                                          'AVV.L': 'Aveva Group Returns'
                                             })
    return P_Rename  



# 3Infrastructure - don't touch this function
def ThreeIn_Price():
    # Get the data from yahoo finance
    ThreeIn = wb.DataReader('3IN.L', data_source='yahoo', start='2009-01-01')
    # Choose just the adjusted price
    Columns = ['Adj Close']
    ThreeI = ThreeIn[Columns] 
    Three = ThreeI.rename(columns = {'Adj Close': '3IN.L'})
    return Three

def get_Etoro_prices():
    #Get the data
    ticker_list = ['AVV.L', 'SCT.L', 'ROR.L', 'OCDO.L', 'CCC.L', '3IN.L','AVST.L', 'ASC.L', 'SPX.L',
           'ECM.L', 'TRN.L', 'PLTR','GOG.L','FGP.L','SHB.L']
    historical_datas = {}
    Portfolio = pd.DataFrame(historical_datas)
    for ticker in ticker_list:
        Portfolio[ticker] = get_data(ticker,  start_date="01/01/2009",  index_as_date = True, interval="1d")['adjclose']
    P_Rename = Portfolio.rename(columns = {'AVV.L': 'Aveva Prices',
                                          'SCT.L': 'Softcat Prices', 
                                          'ROR.L': 'Rotork Prices', 
                                          'OCDO.L': 'Ocado Prices',
                                          'CCC.L': 'Computacenter Prices' , 
                                          '3IN.L': '3Infrastructure Prices', 
                                          'AVST.L': 'Avast Prices', 
                                          'ASC.L': 'ASOS Prices' , 
                                          'SPX.L': 'Spirax Prices',
                                          'ECM.L': 'Electrocomponents Prices', 
                                          'TRN.L': 'Trainline Prices', 
                                          'PLTR': 'Palantir Returns', 
                                          'GOG.L': 'Go-Ahead Prices',
                                          'FGP.L': 'First Group Prices',
                                          'SHB.L': 'Shaftesbury PLC Prices'
                                             })
    return P_Rename
    
def get_Etoro_rtns():
    #Get the data
    ticker_list = ['AVV.L', 'SCT.L', 'ROR.L', 'OCDO.L', 'CCC.L','AVST.L', 'ASC.L', 'SPX.L',
           'ECM.L', 'TRN.L', 'PLTR', 'GOG.L','FGP.L','SHB.L'] 
    historical_datas = {}
    Portfolio = pd.DataFrame(historical_datas)
    for ticker in ticker_list:
        Portfolio[ticker] = get_data(ticker,  start_date="01/01/2009",  index_as_date = True, interval="1d")['adjclose']
    # Swap 3Infrastructure - bad data
    ThreeIn = ThreeIn_Price()  # Get 3In price new data
    Portfolio.insert(loc=5, column='3IN.L', value=ThreeIn)
 
    # remove old ThreeIn
    #Portfolio.drop(columns=['3IN.L'], inplace = True) # This just gives a preview of what our data set will look like. Its # not actually done. Must add 'in place = True' to have effect
   
    #Get returns 
    Returns = Portfolio.pct_change()
    P_Clean = Returns.fillna(0)
    P_Rename = P_Clean.rename(columns = {'AVV.L': 'Aveva Returns',
                                              'SCT.L': 'Softcat Returns', 
                                              'ROR.L': 'Rotork Returns', 
                                              'OCDO.L': 'Ocado Returns', 
                                              'CCC.L': 'Computacenter Returns' , 
                                              '3IN.L': '3Infrastructure Returns', 
                                              'AVST.L': 'Avast Returns', 
                                              'ASC.L': 'ASOS Returns' , 
                                              'SPX.L': 'Spirax Returns',
                                              'ECM.L': 'Electrocomponents Returns', 
                                              'TRN.L': 'Trainline Returns', 
                                              'PLTR': 'Palantir Returns', 
                                              'GOG.L': 'Go-Ahead Group Returns',
                                              'FGP.L': 'First Group Returns',
                                              'SHB.L': 'Shaftesbury PLC Returns'
                                             })
    return P_Rename  


def get_Brexit_COVID_recovery_prices():
    #Get the data
    ticker_list = ['LGEN.L', 'BOO.L', 'AO.L' ] 
    historical_datas = {}
    Portfolio = pd.DataFrame(historical_datas)
    for ticker in ticker_list:
        Portfolio[ticker] = get_data(ticker,  start_date="01/01/2009",  index_as_date = True, interval="1d")['adjclose']
    P_Rename = Portfolio.rename(columns = {'LGEN.L': 'Legal and General Returns',
                                          'BOO.L': 'Boohoo Returns', 
                                          'AO.L': 'AO World Returns'
                                             })
    return P_Rename
    
def get_Brexit_COVID_recovery_rtns():
    #Get the data
    ticker_list = ['LGEN.L', 'BOO.L', 'AO.L' ] 
    historical_datas = {}
    Portfolio = pd.DataFrame(historical_datas)
    for ticker in ticker_list:
        Portfolio[ticker] = get_data(ticker,  start_date="01/01/2009",  index_as_date = True, interval="1d")['adjclose']
 
    #Get returns 
    Returns = Portfolio.pct_change()
    P_Clean = Returns.fillna(0)
    P_Rename = P_Clean.rename(columns = {'LGEN.L': 'Legal and General Returns',
                                          'BOO.L': 'Boohoo Returns', 
                                          'AO.L': 'AO World Returns'
                                             })
    return P_Rename  


def get_CANZUK_prices():
    #Get the data
    ticker_list = ['BUR', 'OBL.AX', 'LIT.L'] 
    historical_datas = {}
    Portfolio = pd.DataFrame(historical_datas)
    for ticker in ticker_list:
        Portfolio[ticker] = get_data(ticker,  start_date="01/01/2009",  index_as_date = True, interval="1d")['adjclose']
    P_Rename = Portfolio.rename(columns={'BUR': 'Burford Capital Prices',
                                          'OBL.AX': 'Omni Bridgeway Limited Prices', 
                                          'LIT.L': 'Litigation Capital Management Limited Prices'
                                             })
    return P_Rename
    
def get_CANZUK_rtns():
    #Get the data
    Prices = get_CANZUK_prices()
    
    #Get returns 
    Returns = Prices.pct_change()
    R_Clean = Returns.fillna(0)
    R_Rename = R_Clean.rename(columns = {'BUR': 'Burford Capital Returns',
                                          'OBL.AX': 'Omni Bridgeway Limited Returns', 
                                          'LIT.L': 'Litigation Capital Management Limited Returns'
                                             })
    return R_Rename  


# Portfolio returns and volatility
# Create portfolio

def portfolio_returns(weights, returns):
    """weights -> returns"""
    
    # take the weights, transpose it and take the matrix multiplication
    return weights.T@ returns

# Volatility
def portfolio_volatility(weights, covmat):
    """Weights -> Covariance"""
    
    # Weights transposes, matrix multiply with covmatrix and matrix multiply this with weights and square root the answer
    return (weights.T @ covmat @ weights)**0.5



# Graph of prices    
def show_me():
    P_Rename = get_Etoro_prices()
    Normalisation = P_Rename.iloc[0]
    Draw_G = (P_Rename/Normalisation * 100).plot(figsize = (15,8));
    Graph = plt.show()
    return Graph


# Gross return
def Gross_return():
    P_Rename = get_rtns()
    Portfolio_rtns = P_Rename
    Gross_rtn = 1 + Portfolio_rtns
    return Gross_rtn


def Optimize_MinR_Vc():
    P_Rename = get_rtns()
    Portfolio_rtns = P_Rename
    #Number of assets in the portfolio
    tckr_list = ticker_list()
    Assets = tckr_list
    num_assets = len(Assets)
    # Lists for Portfolio creation
    Portfolio_returns = []
    Portfolio_Volatilities = []
    Portfolio_GrossR = []
    Aveva_Returns_weight = []
    Softcat_Returns_weight = []
    Rotork_Returns_weight = []
    Ocado_Returns_weight = []
    Computacenter_Returns_weight = [] 
    TInfrastructure_Returns_weight = []
    Avast_Returns_weight = []
    ASOS_Returns_weight = []
    Spirax_Returns_weight = []
    Electrocomponents_Returns_weight = [] 
    Trainline_Returns_weight = []
    Palnuatir_Returns_weight = []
    #Optimising for expected returns and standard deviation
    Gross_rtn = Gross_return()
    
    for x in range (100000):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        Portfolio_returns.append(np.sum(weights * Portfolio_rtns.mean() * 250)) # expected returns
        Portfolio_Volatilities.append(np.sqrt(np.dot(weights.T,np.dot(Portfolio_rtns.cov() * 250, weights)))) # standard deviation 
        Portfolio_GrossR.append(np.sum(weights * Gross_rtn.mean() * 250)) # Gross returns
        Aveva_Returns_weight.append(weights[0])
        Softcat_Returns_weight.append(weights[1])  
        Rotork_Returns_weight.append(weights[2]) 
        Ocado_Returns_weight .append(weights[3]) 
        Computacenter_Returns_weight.append(weights[4]) 
        TInfrastructure_Returns_weight.append(weights[5])
        Avast_Returns_weight.append(weights[6])  
        ASOS_Returns_weight.append(weights[7])
        Spirax_Returns_weight.append(weights[8])
        Electrocomponents_Returns_weight.append(weights[9])
        Trainline_Returns_weight.append(weights[10])
        Palantir_Returns_weight.append(weights[11])
    # Create an array of data for portfolio
    Portfolio_returns = np.array(Portfolio_returns)
    Portfolio_Volatilities = np.array(Portfolio_Volatilities)
    Portfolio_GrossR = np.array(Portfolio_GrossR)
    Aveva_Returns_Weight = np.array(Aveva_Returns_weight)
    Softcat_Returns_Weight = np.array(Softcat_Returns_weight)
    Rotork_Returns_Weight = np.array(Rotork_Returns_weight)
    Ocado_Returns_Weight = np.array(Ocado_Returns_weight)
    Computacenter_Returns_Weight = np.array(Computacenter_Returns_weight)
    TInfrastructure_Returns_Weight = np.array(TInfrastructure_Returns_weight)
    Avast_Returns_Weight = np.array(Avast_Returns_weight)
    ASOS_Returns_Weight = np.array(ASOS_Returns_weight)
    Spirax_Returns_Weight = np.array(Spirax_Returns_weight)
    Electrocomponents_Returns_Weight = np.array(Electrocomponents_Returns_weight)
    Trainline_Returns_Weight = np.array(Trainline_Returns_weight)
    Palantir_Returns_Weight = np.array(Palantir_Returns_weight)
    #Creating a table
    Portfolios = pd.DataFrame({'Return': Portfolio_returns, 
                           'Volatility': Portfolio_Volatilities,
                           'Gross Return': Portfolio_GrossR,
                           'Aveva Weight': Aveva_Returns_weight,
                           'Softcat Weight': Softcat_Returns_weight, 
                           'Rotork Weight': Rotork_Returns_weight,
                            'Ocado Weight': Ocado_Returns_weight,  
                            'Computacenter Weight': Computacenter_Returns_weight,
                            '3Infrastructure Weight': TInfrastructure_Returns_weight,
                            'Avast Weight': Avast_Returns_weight,
                            'ASOS Weight': ASOS_Returns_weight,
                            'Spirax Weight': Spirax_Returns_weight,
                            'Electrocomponents': Electrocomponents_Returns_weight,
                            'Trainline': Trainline_Returns_weight,
                            'Palantir': Palantir_Returns_weight})
    
    # Custom Portfolios
    # if volatitlity is within this range, where is volatility when you search for max return?
    Min_return = Portfolios[(Portfolios['Volatility']>=.135) & (Portfolios['Volatility']<=14.358)].min()['Return']
    Return = Portfolios.iloc[np.where(Portfolios['Return']==Min_return)]
    Min_return_1 = Portfolios[(Portfolios['Volatility']>=.200) & (Portfolios['Volatility']<=9.00)].min()['Return']
    Return_2 = Portfolios.iloc[np.where(Portfolios['Return']==Min_return_1)]
    Min_return_2 = Portfolios[(Portfolios['Volatility']>=.300) & (Portfolios['Volatility']<=8.00)].min()['Return']
    Return_3 = Portfolios.iloc[np.where(Portfolios['Return']==Min_return_2)]
    Min_return_3 = Portfolios[(Portfolios['Volatility']>=.400) & (Portfolios['Volatility']<=7.00)].min()['Return']
    Return_4 = Portfolios.iloc[np.where(Portfolios['Return']==Min_return_3)]
    Min_return_4 = Portfolios[(Portfolios['Volatility']>=.500) & (Portfolios['Volatility']<=6.00)].min()['Return']
    Return_5 = Portfolios.iloc[np.where(Portfolios['Return']==Min_return_4)]
    Min_return_5 = Portfolios[(Portfolios['Volatility']>=.600) & (Portfolios['Volatility']<=5.00)].min()['Return']
    Return_6 = Portfolios.iloc[np.where(Portfolios['Return']==Min_return_5)]
    Min_return_6 = Portfolios[(Portfolios['Volatility']>=.700) & (Portfolios['Volatility']<=4.00)].min()['Return']
    Return_7 = Portfolios.iloc[np.where(Portfolios['Return']==Min_return_6)]
    Min_return_7 = Portfolios[(Portfolios['Volatility']>=.800) & (Portfolios['Volatility']<=3.00)].min()['Return']
    Return_8= Portfolios.iloc[np.where(Portfolios['Return']==Min_return_7)]
    Min_return_8 = Portfolios[(Portfolios['Volatility']>=.900) & (Portfolios['Volatility']<=2.00)].min()['Return']
    Return_8= Portfolios.iloc[np.where(Portfolios['Return']==Min_return_8)]
    Min_return_9 = Portfolios[(Portfolios['Volatility']>=.100) & (Portfolios['Volatility']<=1.00)].min()['Return']
    Return_9= Portfolios.iloc[np.where(Portfolios['Return']==Min_return_9)]
    
    Final_MinOp = pd.concat([Return,Return_2, Return_3, Return_4, Return_5, Return_6,
                        Return_7, Return_8, Return_9])
    return Final_MinOp


def Optimize_MaxR_Vc():
    P_Rename = get_rtns()
    Portfolio_rtns = P_Rename
    #Number of assets in the portfolio
    tckr_list = ticker_list()
    Assets = tckr_list
    num_assets = len(Assets)
    # Lists for Portfolio creation
    Portfolio_returns = []
    Portfolio_Volatilities = []
    Portfolio_GrossR = []
    Aveva_Returns_weight = []
    Softcat_Returns_weight = []
    Rotork_Returns_weight = []
    Ocado_Returns_weight = []
    Computacenter_Returns_weight = [] 
    TInfrastructure_Returns_weight = []
    Avast_Returns_weight = []
    ASOS_Returns_weight = []
    Spirax_Returns_weight = []
    Electrocomponents_Returns_weight = [] 
    Trainline_Returns_weight = []
    Palantir_Returns_weight = []
    #Optimising for expected returns and standard deviation
    Gross_rtn = Gross_return()
    
    for x in range (100000):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        Portfolio_returns.append(np.sum(weights * Portfolio_rtns.mean() * 250)) # expected returns
        Portfolio_Volatilities.append(np.sqrt(np.dot(weights.T,np.dot(Portfolio_rtns.cov() * 250, weights)))) # standard deviation 
        Portfolio_GrossR.append(np.sum(weights * Gross_rtn.mean() * 250)) # Gross returns
        Aveva_Returns_weight.append(weights[0])
        Softcat_Returns_weight.append(weights[1])  
        Rotork_Returns_weight.append(weights[2]) 
        Ocado_Returns_weight .append(weights[3]) 
        Computacenter_Returns_weight.append(weights[4]) 
        TInfrastructure_Returns_weight.append(weights[5])
        Avast_Returns_weight.append(weights[6])  
        ASOS_Returns_weight.append(weights[7])
        Spirax_Returns_weight.append(weights[8])
        Electrocomponents_Returns_weight.append(weights[9])
        Trainline_Returns_weight.append(weights[10])
        Palantir_Returns_weight.append(weights[11])
    # Create an array of data for portfolio
    Portfolio_returns = np.array(Portfolio_returns)
    Portfolio_Volatilities = np.array(Portfolio_Volatilities)
    Portfolio_GrossR = np.array(Portfolio_GrossR)
    Aveva_Returns_Weight = np.array(Aveva_Returns_weight)
    Softcat_Returns_Weight = np.array(Softcat_Returns_weight)
    Rotork_Returns_Weight = np.array(Rotork_Returns_weight)
    Ocado_Returns_Weight = np.array(Ocado_Returns_weight)
    Computacenter_Returns_Weight = np.array(Computacenter_Returns_weight)
    TInfrastructure_Returns_Weight = np.array(TInfrastructure_Returns_weight)
    Avast_Returns_Weight = np.array(Avast_Returns_weight)
    ASOS_Returns_Weight = np.array(ASOS_Returns_weight)
    Spirax_Returns_Weight = np.array(Spirax_Returns_weight)
    Electrocomponents_Returns_Weight = np.array(Electrocomponents_Returns_weight)
    Trainline_Returns_Weight = np.array(Trainline_Returns_weight)
    Palantir_Returns_Weight = np.array(Palantir_Returns_weight)
    #Creating a table
    Portfolios = pd.DataFrame({'Return': Portfolio_returns, 
                           'Volatility': Portfolio_Volatilities,
                           'Gross Return': Portfolio_GrossR,
                           'Aveva Weight': Aveva_Returns_weight,
                           'Softcat Weight': Softcat_Returns_weight, 
                           'Rotork Weight': Rotork_Returns_weight,
                            'Ocado Weight': Ocado_Returns_weight,  
                            'Computacenter Weight': Computacenter_Returns_weight,
                            '3Infrastructure Weight': TInfrastructure_Returns_weight,
                            'Avast Weight': Avast_Returns_weight,
                            'ASOS Weight': ASOS_Returns_weight,
                            'Spirax Weight': Spirax_Returns_weight,
                            'Electrocomponents': Electrocomponents_Returns_weight,
                            'Trainline': Trainline_Returns_weight,
                            'Palantir': Palantir_Returns_weight})
    
    # Custom Portfolios
    # if volatitlity is within this range, where is volatility when you search for max return?
    Max_return = Portfolios[(Portfolios['Volatility']>=.135) & (Portfolios['Volatility']<=14.358)].max()['Return']
    Return = Portfolios.iloc[np.where(Portfolios['Return']==Max_return)]
    Max_return_1 = Portfolios[(Portfolios['Volatility']>=.200) & (Portfolios['Volatility']<=9.00)].max()['Return']
    Return_2 = Portfolios.iloc[np.where(Portfolios['Return']==Max_return_1)]
    Max_return_2 = Portfolios[(Portfolios['Volatility']>=.300) & (Portfolios['Volatility']<=8.00)].max()['Return']
    Return_3 = Portfolios.iloc[np.where(Portfolios['Return']==Max_return_2)]
    Max_return_3 = Portfolios[(Portfolios['Volatility']>=.400) & (Portfolios['Volatility']<=7.00)].max()['Return']
    Return_4 = Portfolios.iloc[np.where(Portfolios['Return']==Max_return_3)]
    Max_return_4 = Portfolios[(Portfolios['Volatility']>=.500) & (Portfolios['Volatility']<=6.00)].max()['Return']
    Return_5 = Portfolios.iloc[np.where(Portfolios['Return']==Max_return_4)]
    Max_return_5 = Portfolios[(Portfolios['Volatility']>=.600) & (Portfolios['Volatility']<=5.00)].max()['Return']
    Return_6 = Portfolios.iloc[np.where(Portfolios['Return']==Max_return_5)]
    Max_return_6 = Portfolios[(Portfolios['Volatility']>=.700) & (Portfolios['Volatility']<=4.00)].max()['Return']
    Return_7 = Portfolios.iloc[np.where(Portfolios['Return']==Max_return_6)]
    Max_return_7 = Portfolios[(Portfolios['Volatility']>=.800) & (Portfolios['Volatility']<=3.00)].max()['Return']
    Return_8= Portfolios.iloc[np.where(Portfolios['Return']==Max_return_7)]
    Max_return_8 = Portfolios[(Portfolios['Volatility']>=.900) & (Portfolios['Volatility']<=2.00)].max()['Return']
    Return_8= Portfolios.iloc[np.where(Portfolios['Return']==Max_return_8)]
    Max_return_9 = Portfolios[(Portfolios['Volatility']>=.100) & (Portfolios['Volatility']<=1.00)].max()['Return']
    Return_9= Portfolios.iloc[np.where(Portfolios['Return']==Max_return_9)]
    
    Final_MinOp = pd.concat([Return,Return_2, Return_3, Return_4, Return_5, Return_6,
                        Return_7, Return_8, Return_9])
    return Final_MinOp

#Gross return vol constraint
def Optimize_MaxGR_Vc():
    P_Rename = get_rtns()
    Portfolio_rtns = P_Rename
    Gross_rtn = Gross_return()
    #Number of assets in the portfolio
    tckr_list = ticker_list()
    Assets = tckr_list
    num_assets = len(Assets)
    # Lists for Portfolio creation
    Portfolio_returns = []
    Portfolio_Volatilities = []
    Portfolio_GrossR = []
    Aveva_Returns_weight = []
    Softcat_Returns_weight = []
    Rotork_Returns_weight = []
    Ocado_Returns_weight = []
    Computacenter_Returns_weight = [] 
    TInfrastructure_Returns_weight = []
    Avast_Returns_weight = []
    ASOS_Returns_weight = []
    Spirax_Returns_weight = []
    Electrocomponents_Returns_weight = [] 
    Trainline_Returns_weight = []
    Palantir_Returns_weight = []
    
    #Optimising for expected returns and standard deviation
    for x in range (100000):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        Portfolio_returns.append(np.sum(weights * Portfolio_rtns.mean() * 250)) # expected returns
        Portfolio_Volatilities.append(np.sqrt(np.dot(weights.T,np.dot(Portfolio_rtns.cov() * 250, weights)))) # standard deviation 
        Portfolio_GrossR.append(np.sum(weights * Gross_rtn.mean() * 250)) # Gross returns
        Aveva_Returns_weight.append(weights[0])
        Softcat_Returns_weight.append(weights[1])  
        Rotork_Returns_weight.append(weights[2]) 
        Ocado_Returns_weight .append(weights[3]) 
        Computacenter_Returns_weight.append(weights[4]) 
        TInfrastructure_Returns_weight.append(weights[5])
        Avast_Returns_weight.append(weights[6])  
        ASOS_Returns_weight.append(weights[7])
        Spirax_Returns_weight.append(weights[8])
        Electrocomponents_Returns_weight.append(weights[9])
        Trainline_Returns_weight.append(weights[10])
        Palantir_Returns_weight.append(weights[11])
    # Create an array of data for portfolio
    Portfolio_returns = np.array(Portfolio_returns)
    Portfolio_Volatilities = np.array(Portfolio_Volatilities)
    Portfolio_GrossR = np.array(Portfolio_GrossR)
    Aveva_Returns_Weight = np.array(Aveva_Returns_weight)
    Softcat_Returns_Weight = np.array(Softcat_Returns_weight)
    Rotork_Returns_Weight = np.array(Rotork_Returns_weight)
    Ocado_Returns_Weight = np.array(Ocado_Returns_weight)
    Computacenter_Returns_Weight = np.array(Computacenter_Returns_weight)
    TInfrastructure_Returns_Weight = np.array(TInfrastructure_Returns_weight)
    Avast_Returns_Weight = np.array(Avast_Returns_weight)
    ASOS_Returns_Weight = np.array(ASOS_Returns_weight)
    Spirax_Returns_Weight = np.array(Spirax_Returns_weight)
    Electrocomponents_Returns_Weight = np.array(Electrocomponents_Returns_weight)
    Trainline_Returns_Weight = np.array(Trainline_Returns_weight)
    Palantir_Returns_Weight = np.array(Palantir_Returns_weight)
    #Creating a table
    Portfolios = pd.DataFrame({'Return': Portfolio_returns, 
                           'Volatility': Portfolio_Volatilities,
                           'Gross Return': Portfolio_GrossR,
                           'Aveva Weight': Aveva_Returns_weight,
                           'Softcat Weight': Softcat_Returns_weight, 
                           'Rotork Weight': Rotork_Returns_weight,
                            'Ocado Weight': Ocado_Returns_weight,  
                            'Computacenter Weight': Computacenter_Returns_weight,
                            '3Infrastructure Weight': TInfrastructure_Returns_weight,
                            'Avast Weight': Avast_Returns_weight,
                            'ASOS Weight': ASOS_Returns_weight,
                            'Spirax Weight': Spirax_Returns_weight,
                            'Electrocomponents': Electrocomponents_Returns_weight,
                            'Trainline': Trainline_Returns_weight,
                            'Palantir': Palantir_Returns_weight})
    
    # Custom Portfolios
    # if volatitlity is within this range, where is volatility when you search for max Gross return?
    Max_Greturn = Portfolios[(Portfolios['Volatility']>=.135) & (Portfolios['Volatility']<=14.358)].max()['Gross Return']
    Return = Portfolios.iloc[np.where(Portfolios['Gross Return']==Max_Greturn)]
    Max_Greturn_1 = Portfolios[(Portfolios['Volatility']>=.200) & (Portfolios['Volatility']<=9.00)].max()['Gross Return']
    Return_2 = Portfolios.iloc[np.where(Portfolios['Gross Return']==Max_Greturn_1)]
    Max_Greturn_2 = Portfolios[(Portfolios['Volatility']>=.300) & (Portfolios['Volatility']<=8.00)].max()['Gross Return']
    Return_3 = Portfolios.iloc[np.where(Portfolios['Gross Return']==Max_Greturn_2)]
    Max_Greturn_3 = Portfolios[(Portfolios['Volatility']>=.400) & (Portfolios['Volatility']<=7.00)].max()['Gross Return']
    Return_4 = Portfolios.iloc[np.where(Portfolios['Gross Return']==Max_Greturn_3)]
    Max_Greturn_4 = Portfolios[(Portfolios['Volatility']>=.500) & (Portfolios['Volatility']<=6.00)].max()['Gross Return']
    Return_5 = Portfolios.iloc[np.where(Portfolios['Gross Return']==Max_Greturn_4)]
    Max_Greturn_5 = Portfolios[(Portfolios['Volatility']>=.600) & (Portfolios['Volatility']<=5.00)].max()['Gross Return']
    Return_6 = Portfolios.iloc[np.where(Portfolios['Gross Return']==Max_Greturn_5)]
    Max_Greturn_6 = Portfolios[(Portfolios['Volatility']>=.700) & (Portfolios['Volatility']<=4.00)].max()['Gross Return']
    Return_7 = Portfolios.iloc[np.where(Portfolios['Gross Return']==Max_Greturn_6)]
    Max_Greturn_7 = Portfolios[(Portfolios['Volatility']>=.800) & (Portfolios['Volatility']<=3.00)].max()['Gross Return']
    Return_8= Portfolios.iloc[np.where(Portfolios['Gross Return']==Max_Greturn_7)]
    Max_Greturn_8 = Portfolios[(Portfolios['Volatility']>=.900) & (Portfolios['Volatility']<=2.00)].max()['Gross Return']
    Return_8= Portfolios.iloc[np.where(Portfolios['Gross Return']==Max_Greturn_8)]
    Max_Greturn_9 = Portfolios[(Portfolios['Volatility']>=.100) & (Portfolios['Volatility']<=1.00)].max()['Gross Return']
    Return_9= Portfolios.iloc[np.where(Portfolios['Gross Return']==Max_Greturn_9)]
    
    Final_MaxGOp = pd.concat([Return,Return_2, Return_3, Return_4, Return_5, Return_6,
                        Return_7, Return_8, Return_9])
    return Final_MaxGOp


#Gross return vol constraint
def Optimize_MinGR_Vc():
    P_Rename = get_rtns()
    Portfolio_rtns = P_Rename
    Gross_rtn = Gross_return()
    #Number of assets in the portfolio
    tckr_list = ticker_list()
    Assets = tckr_list
    num_assets = len(Assets)
    # Lists for Portfolio creation
    Portfolio_returns = []
    Portfolio_Volatilities = []
    Portfolio_GrossR = []
    Aveva_Returns_weight = []
    Softcat_Returns_weight = []
    Rotork_Returns_weight = []
    Ocado_Returns_weight = []
    Computacenter_Returns_weight = [] 
    TInfrastructure_Returns_weight = []
    Avast_Returns_weight = []
    ASOS_Returns_weight = []
    Spirax_Returns_weight = []
    Electrocomponents_Returns_weight = [] 
    Trainline_Returns_weight = []
    Palantir_Returns_weight = []
    
    #Optimising for expected returns and standard deviation
    for x in range (100000):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        Portfolio_returns.append(np.sum(weights * Portfolio_rtns.mean() * 250)) # expected returns
        Portfolio_Volatilities.append(np.sqrt(np.dot(weights.T,np.dot(Portfolio_rtns.cov() * 250, weights)))) # standard deviation 
        Portfolio_GrossR.append(np.sum(weights * Gross_rtn.mean() * 250)) # Gross returns
        Aveva_Returns_weight.append(weights[0])
        Softcat_Returns_weight.append(weights[1])  
        Rotork_Returns_weight.append(weights[2]) 
        Ocado_Returns_weight .append(weights[3]) 
        Computacenter_Returns_weight.append(weights[4]) 
        TInfrastructure_Returns_weight.append(weights[5])
        Avast_Returns_weight.append(weights[6])  
        ASOS_Returns_weight.append(weights[7])
        Spirax_Returns_weight.append(weights[8])
        Electrocomponents_Returns_weight.append(weights[9])
        Trainline_Returns_weight.append(weights[10])
        Palantir_Returns_weight.append(weights[11])
    # Create an array of data for portfolio
    Portfolio_returns = np.array(Portfolio_returns)
    Portfolio_Volatilities = np.array(Portfolio_Volatilities)
    Portfolio_GrossR = np.array(Portfolio_GrossR)
    Aveva_Returns_Weight = np.array(Aveva_Returns_weight)
    Softcat_Returns_Weight = np.array(Softcat_Returns_weight)
    Rotork_Returns_Weight = np.array(Rotork_Returns_weight)
    Ocado_Returns_Weight = np.array(Ocado_Returns_weight)
    Computacenter_Returns_Weight = np.array(Computacenter_Returns_weight)
    TInfrastructure_Returns_Weight = np.array(TInfrastructure_Returns_weight)
    Avast_Returns_Weight = np.array(Avast_Returns_weight)
    ASOS_Returns_Weight = np.array(ASOS_Returns_weight)
    Spirax_Returns_Weight = np.array(Spirax_Returns_weight)
    Electrocomponents_Returns_Weight = np.array(Electrocomponents_Returns_weight)
    Trainline_Returns_Weight = np.array(Trainline_Returns_weight)
    Palantir_Returns_Weight = np.array(Palantir_Returns_weight)
    #Creating a table
    Portfolios = pd.DataFrame({'Return': Portfolio_returns, 
                           'Volatility': Portfolio_Volatilities,
                           'Gross Return': Portfolio_GrossR,
                           'Aveva Weight': Aveva_Returns_weight,
                           'Softcat Weight': Softcat_Returns_weight, 
                           'Rotork Weight': Rotork_Returns_weight,
                            'Ocado Weight': Ocado_Returns_weight,  
                            'Computacenter Weight': Computacenter_Returns_weight,
                            '3Infrastructure Weight': TInfrastructure_Returns_weight,
                            'Avast Weight': Avast_Returns_weight,
                            'ASOS Weight': ASOS_Returns_weight,
                            'Spirax Weight': Spirax_Returns_weight,
                            'Electrocomponents': Electrocomponents_Returns_weight,
                            'Trainline': Trainline_Returns_weight,
                            'Palantir': Palantir_Returns_weight})
    
    # Custom Portfolios
    # if volatitlity is within this range, where is volatility when you search for max Gross return?
    Min_Greturn = Portfolios[(Portfolios['Volatility']>=.135) & (Portfolios['Volatility']<=14.358)].min()['Gross Return']
    Return = Portfolios.iloc[np.where(Portfolios['Gross Return']==Min_Greturn)]
    Min_Greturn_1 = Portfolios[(Portfolios['Volatility']>=.200) & (Portfolios['Volatility']<=9.00)].min()['Gross Return']
    Return_2 = Portfolios.iloc[np.where(Portfolios['Gross Return']==Min_Greturn_1)]
    Min_Greturn_2 = Portfolios[(Portfolios['Volatility']>=.300) & (Portfolios['Volatility']<=8.00)].min()['Gross Return']
    Return_3 = Portfolios.iloc[np.where(Portfolios['Gross Return']==Min_Greturn_2)]
    Min_Greturn_3 = Portfolios[(Portfolios['Volatility']>=.400) & (Portfolios['Volatility']<=7.00)].min()['Gross Return']
    Return_4 = Portfolios.iloc[np.where(Portfolios['Gross Return']==Min_Greturn_3)]
    Min_Greturn_4 = Portfolios[(Portfolios['Volatility']>=.500) & (Portfolios['Volatility']<=6.00)].min()['Gross Return']
    Return_5 = Portfolios.iloc[np.where(Portfolios['Gross Return']==Min_Greturn_4)]
    Min_Greturn_5 = Portfolios[(Portfolios['Volatility']>=.600) & (Portfolios['Volatility']<=5.00)].min()['Gross Return']
    Return_6 = Portfolios.iloc[np.where(Portfolios['Gross Return']==Min_Greturn_5)]
    Min_Greturn_6 = Portfolios[(Portfolios['Volatility']>=.700) & (Portfolios['Volatility']<=4.00)].min()['Gross Return']
    Return_7 = Portfolios.iloc[np.where(Portfolios['Gross Return']==Min_Greturn_6)]
    Min_Greturn_7 = Portfolios[(Portfolios['Volatility']>=.800) & (Portfolios['Volatility']<=3.00)].min()['Gross Return']
    Return_8= Portfolios.iloc[np.where(Portfolios['Gross Return']==Min_Greturn_7)]
    Min_Greturn_8 = Portfolios[(Portfolios['Volatility']>=.900) & (Portfolios['Volatility']<=2.00)].min()['Gross Return']
    Return_8= Portfolios.iloc[np.where(Portfolios['Gross Return']==Min_Greturn_8)]
    Min_Greturn_9 = Portfolios[(Portfolios['Volatility']>=.100) & (Portfolios['Volatility']<=1.00)].min()['Gross Return']
    Return_9= Portfolios.iloc[np.where(Portfolios['Gross Return']==Min_Greturn_9)]
    
    Final_MaxGOp = pd.concat([Return,Return_2, Return_3, Return_4, Return_5, Return_6,
                        Return_7, Return_8, Return_9])
    return Final_MaxGOp



def Optimize_MaxVolR_rets():
    P_Rename = get_rtns()
    Portfolio_rtns = P_Rename
    Gross_rtn = Gross_return()
    #Number of assets in the portfolio
    tckr_list = ticker_list()
    Assets = tckr_list
    num_assets = len(Assets)
    # Lists for Portfolio creation
    Portfolio_returns = []
    Portfolio_Volatilities = []
    Portfolio_GrossR = []
    Aveva_Returns_weight = []
    Softcat_Returns_weight = []
    Rotork_Returns_weight = []
    Ocado_Returns_weight = []
    Computacenter_Returns_weight = [] 
    TInfrastructure_Returns_weight = []
    Avast_Returns_weight = []
    ASOS_Returns_weight = []
    Spirax_Returns_weight = []
    Electrocomponents_Returns_weight = [] 
    Trainline_Returns_weight = []
    Palantir_Returns_weight = []
    
    #Optimising for expected returns and standard deviation
    for x in range (100000):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        Portfolio_returns.append(np.sum(weights * Portfolio_rtns.mean() * 250)) # expected returns
        Portfolio_Volatilities.append(np.sqrt(np.dot(weights.T,np.dot(Portfolio_rtns.cov() * 250, weights)))) # standard deviation 
        Portfolio_GrossR.append(np.sum(weights * Gross_rtn.mean() * 250)) # Gross returns
        Aveva_Returns_weight.append(weights[0])
        Softcat_Returns_weight.append(weights[1])  
        Rotork_Returns_weight.append(weights[2]) 
        Ocado_Returns_weight .append(weights[3]) 
        Computacenter_Returns_weight.append(weights[4]) 
        TInfrastructure_Returns_weight.append(weights[5])
        Avast_Returns_weight.append(weights[6])  
        ASOS_Returns_weight.append(weights[7])
        Spirax_Returns_weight.append(weights[8])
        Electrocomponents_Returns_weight.append(weights[9])
        Trainline_Returns_weight.append(weights[10])
        Palantir_Returns_weight.append(weights[11])
    # Create an array of data for portfolio
    Portfolio_returns = np.array(Portfolio_returns)
    Portfolio_Volatilities = np.array(Portfolio_Volatilities)
    Portfolio_GrossR = np.array(Portfolio_GrossR)
    Aveva_Returns_Weight = np.array(Aveva_Returns_weight)
    Softcat_Returns_Weight = np.array(Softcat_Returns_weight)
    Rotork_Returns_Weight = np.array(Rotork_Returns_weight)
    Ocado_Returns_Weight = np.array(Ocado_Returns_weight)
    Computacenter_Returns_Weight = np.array(Computacenter_Returns_weight)
    TInfrastructure_Returns_Weight = np.array(TInfrastructure_Returns_weight)
    Avast_Returns_Weight = np.array(Avast_Returns_weight)
    ASOS_Returns_Weight = np.array(ASOS_Returns_weight)
    Spirax_Returns_Weight = np.array(Spirax_Returns_weight)
    Electrocomponents_Returns_Weight = np.array(Electrocomponents_Returns_weight)
    Trainline_Returns_Weight = np.array(Trainline_Returns_weight)
    Palantir_Returns_Weight = np.array(Palantir_Returns_weight)
    #Creating a table
    Portfolios = pd.DataFrame({'Return': Portfolio_returns, 
                           'Volatility': Portfolio_Volatilities,
                           'Gross Return': Portfolio_GrossR,
                           'Aveva Weight': Aveva_Returns_weight,
                           'Softcat Weight': Softcat_Returns_weight, 
                           'Rotork Weight': Rotork_Returns_weight,
                            'Ocado Weight': Ocado_Returns_weight,  
                            'Computacenter Weight': Computacenter_Returns_weight,
                            '3Infrastructure Weight': TInfrastructure_Returns_weight,
                            'Avast Weight': Avast_Returns_weight,
                            'ASOS Weight': ASOS_Returns_weight,
                            'Spirax Weight': Spirax_Returns_weight,
                            'Electrocomponents': Electrocomponents_Returns_weight,
                            'Trainline': Trainline_Returns_weight,
                            'Palantir': Palantir_Returns_weight})
    
    # Custom Portfolios
    # if volatitlity is within this range, where is volatility when you search for max Gross return?
    Max_Volatility = Portfolios[(Portfolios['Return']>=.179) & (Portfolios['Return']<=4.436)].max()['Volatility']
    Volatility = Portfolios.iloc[np.where(Portfolios['Volatility']==Max_Volatility)]
    Max_Volatility_2 = Portfolios[(Portfolios['Return']>=.200) & (Portfolios['Return']<=9.00)].max()['Volatility']
    Volatility_2 = Portfolios.iloc[np.where(Portfolios['Volatility']==Max_Volatility_2)]
    Max_Volatility_3 = Portfolios[(Portfolios['Return']>=.300) & (Portfolios['Return']<=8.00)].max()['Volatility']
    Volatility_3 = Portfolios.iloc[np.where(Portfolios['Volatility']==Max_Volatility_3)]
    Max_Volatility_4 = Portfolios[(Portfolios['Return']>=.400) & (Portfolios['Return']<=7.00)].max()['Volatility']
    Volatility_4 = Portfolios.iloc[np.where(Portfolios['Volatility']==Max_Volatility_4)]
    Max_Volatility_5 = Portfolios[(Portfolios['Return']>=.500) & (Portfolios['Return']<=6.00)].max()['Volatility']
    Volatility_5 = Portfolios.iloc[np.where(Portfolios['Volatility']==Max_Volatility_5)]
    Max_Volatility_6 = Portfolios[(Portfolios['Return']>=.600) & (Portfolios['Return']<=5.00)].max()['Volatility']
    Volatility_6 = Portfolios.iloc[np.where(Portfolios['Volatility']==Max_Volatility_6)]
    Max_Volatility_7 = Portfolios[(Portfolios['Return']>=.700) & (Portfolios['Return']<=4.00)].max()['Volatility']
    Volatility_7 = Portfolios.iloc[np.where(Portfolios['Volatility']==Max_Volatility_7)]
    Max_Volatility_8 = Portfolios[(Portfolios['Return']>=.800) & (Portfolios['Return']<=3.00)].max()['Volatility']
    Volatility_8 = Portfolios.iloc[np.where(Portfolios['Volatility']==Max_Volatility_8)]
    Max_Volatility_9 = Portfolios[(Portfolios['Return']>=.900) & (Portfolios['Return']<=2.00)].max()['Volatility']
    Volatility_9 = Portfolios.iloc[np.where(Portfolios['Volatility']==Max_Volatility_9)]
    Max_Volatility_10 = Portfolios[(Portfolios['Return']>=1.00) & (Portfolios['Return']<=1.00)].max()['Volatility']
    Volatility_10 = Portfolios.iloc[np.where(Portfolios['Volatility']==Max_Volatility_10)]
    
    Final_MaxVol = pd.concat([Volatility,Volatility_2, Volatility_3, Volatility_4, Volatility_5, Volatility_6,
                        Volatility_7, Volatility_8, Volatility_9, Volatility_10])
    return Final_MaxVol

def Optimize_MinVolR_rets():
    P_Rename = get_rtns()
    Portfolio_rtns = P_Rename
    Gross_rtn = Gross_return()
    #Number of assets in the portfolio
    tckr_list = ticker_list()
    Assets = tckr_list
    num_assets = len(Assets)
    # Lists for Portfolio creation
    Portfolio_returns = []
    Portfolio_Volatilities = []
    Portfolio_GrossR = []
    Aveva_Returns_weight = []
    Softcat_Returns_weight = []
    Rotork_Returns_weight = []
    Ocado_Returns_weight = []
    Computacenter_Returns_weight = [] 
    TInfrastructure_Returns_weight = []
    Avast_Returns_weight = []
    ASOS_Returns_weight = []
    Spirax_Returns_weight = []
    Electrocomponents_Returns_weight = [] 
    Trainline_Returns_weight = []
    Palantir_Returns_weight = []
    
    #Optimising for expected returns and standard deviation
    for x in range (100000):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        Portfolio_returns.append(np.sum(weights * Portfolio_rtns.mean() * 250)) # expected returns
        Portfolio_Volatilities.append(np.sqrt(np.dot(weights.T,np.dot(Portfolio_rtns.cov() * 250, weights)))) # standard deviation 
        Portfolio_GrossR.append(np.sum(weights * Gross_rtn.mean() * 250)) # Gross returns
        Aveva_Returns_weight.append(weights[0])
        Softcat_Returns_weight.append(weights[1])  
        Rotork_Returns_weight.append(weights[2]) 
        Ocado_Returns_weight .append(weights[3]) 
        Computacenter_Returns_weight.append(weights[4]) 
        TInfrastructure_Returns_weight.append(weights[5])
        Avast_Returns_weight.append(weights[6])  
        ASOS_Returns_weight.append(weights[7])
        Spirax_Returns_weight.append(weights[8])
        Electrocomponents_Returns_weight.append(weights[9])
        Trainline_Returns_weight.append(weights[10])
        Palantir_Returns_weight.append(weights[11])
    # Create an array of data for portfolio
    Portfolio_returns = np.array(Portfolio_returns)
    Portfolio_Volatilities = np.array(Portfolio_Volatilities)
    Portfolio_GrossR = np.array(Portfolio_GrossR)
    Aveva_Returns_Weight = np.array(Aveva_Returns_weight)
    Softcat_Returns_Weight = np.array(Softcat_Returns_weight)
    Rotork_Returns_Weight = np.array(Rotork_Returns_weight)
    Ocado_Returns_Weight = np.array(Ocado_Returns_weight)
    Computacenter_Returns_Weight = np.array(Computacenter_Returns_weight)
    TInfrastructure_Returns_Weight = np.array(TInfrastructure_Returns_weight)
    Avast_Returns_Weight = np.array(Avast_Returns_weight)
    ASOS_Returns_Weight = np.array(ASOS_Returns_weight)
    Spirax_Returns_Weight = np.array(Spirax_Returns_weight)
    Electrocomponents_Returns_Weight = np.array(Electrocomponents_Returns_weight)
    Trainline_Returns_Weight = np.array(Trainline_Returns_weight)
    Palantir_Returns_Weight = np.array(Palantir_Returns_weight)
    #Creating a table
    Portfolios = pd.DataFrame({'Return': Portfolio_returns, 
                           'Volatility': Portfolio_Volatilities,
                           'Gross Return': Portfolio_GrossR,
                           'Aveva Weight': Aveva_Returns_weight,
                           'Softcat Weight': Softcat_Returns_weight, 
                           'Rotork Weight': Rotork_Returns_weight,
                            'Ocado Weight': Ocado_Returns_weight,  
                            'Computacenter Weight': Computacenter_Returns_weight,
                            '3Infrastructure Weight': TInfrastructure_Returns_weight,
                            'Avast Weight': Avast_Returns_weight,
                            'ASOS Weight': ASOS_Returns_weight,
                            'Spirax Weight': Spirax_Returns_weight,
                            'Electrocomponents': Electrocomponents_Returns_weight,
                            'Trainline': Trainline_Returns_weight,
                            'Palantir': Palantir_Returns_weight})
    
    # Custom Portfolios
    # if volatitlity is within this range, where is volatility when you search for max Gross return?
    Min_Volatility = Portfolios[(Portfolios['Return']>=.179) & (Portfolios['Return']<=4.436)].min()['Volatility']
    Volatility = Portfolios.iloc[np.where(Portfolios['Volatility']==Min_Volatility)]
    Min_Volatility_2 = Portfolios[(Portfolios['Return']>=.200) & (Portfolios['Return']<=9.00)].min()['Volatility']
    Volatility_2 = Portfolios.iloc[np.where(Portfolios['Volatility']==Min_Volatility_2)]
    Min_Volatility_3 = Portfolios[(Portfolios['Return']>=.300) & (Portfolios['Return']<=8.00)].min()['Volatility']
    Volatility_3 = Portfolios.iloc[np.where(Portfolios['Volatility']==Min_Volatility_3)]
    Min_Volatility_4 = Portfolios[(Portfolios['Return']>=.400) & (Portfolios['Return']<=7.00)].min()['Volatility']
    Volatility_4 = Portfolios.iloc[np.where(Portfolios['Volatility']==Min_Volatility_4)]
    Min_Volatility_5 = Portfolios[(Portfolios['Return']>=.500) & (Portfolios['Return']<=6.00)].min()['Volatility']
    Volatility_5 = Portfolios.iloc[np.where(Portfolios['Volatility']==Min_Volatility_5)]
    Min_Volatility_6 = Portfolios[(Portfolios['Return']>=.600) & (Portfolios['Return']<=5.00)].min()['Volatility']
    Volatility_6 = Portfolios.iloc[np.where(Portfolios['Volatility']==Min_Volatility_6)]
    Min_Volatility_7 = Portfolios[(Portfolios['Return']>=.700) & (Portfolios['Return']<=4.00)].min()['Volatility']
    Volatility_7 = Portfolios.iloc[np.where(Portfolios['Volatility']==Min_Volatility_7)]
    Min_Volatility_8 = Portfolios[(Portfolios['Return']>=.800) & (Portfolios['Return']<=3.00)].min()['Volatility']
    Volatility_8 = Portfolios.iloc[np.where(Portfolios['Volatility']==Min_Volatility_8)]
    Min_Volatility_9 = Portfolios[(Portfolios['Return']>=.900) & (Portfolios['Return']<=2.00)].min()['Volatility']
    Volatility_9 = Portfolios.iloc[np.where(Portfolios['Volatility']==Min_Volatility_9)]
    Min_Volatility_10 = Portfolios[(Portfolios['Return']>=1.00) & (Portfolios['Return']<=1.00)].min()['Volatility']
    Volatility_10 = Portfolios.iloc[np.where(Portfolios['Volatility']==Min_Volatility_10)]
    
    Final_MinVol = pd.concat([Volatility,Volatility_2, Volatility_3, Volatility_4, Volatility_5, Volatility_6,
                        Volatility_7, Volatility_8, Volatility_9, Volatility_10])
    return Final_MinVol


def Optimize_MaxR_GR_rets():
    P_Rename = get_rtns()
    Portfolio_rtns = P_Rename
    Gross_rtn = Gross_return()
    #Number of assets in the portfolio
    tckr_list = ticker_list()
    Assets = tckr_list
    num_assets = len(Assets)
    # Lists for Portfolio creation
    Portfolio_returns = []
    Portfolio_Volatilities = []
    Portfolio_GrossR = []
    Aveva_Returns_weight = []
    Softcat_Returns_weight = []
    Rotork_Returns_weight = []
    Ocado_Returns_weight = []
    Computacenter_Returns_weight = [] 
    TInfrastructure_Returns_weight = []
    Avast_Returns_weight = []
    ASOS_Returns_weight = []
    Spirax_Returns_weight = []
    Electrocomponents_Returns_weight = [] 
    Trainline_Returns_weight = []
    Palantir_Returns_weight = []
    
    #Optimising for expected returns and standard deviation
    for x in range (100000):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        Portfolio_returns.append(np.sum(weights * Portfolio_rtns.mean() * 250)) # expected returns
        Portfolio_Volatilities.append(np.sqrt(np.dot(weights.T,np.dot(Portfolio_rtns.cov() * 250, weights)))) # standard deviation 
        Portfolio_GrossR.append(np.sum(weights * Gross_rtn.mean() * 250)) # Gross returns
        Aveva_Returns_weight.append(weights[0])
        Softcat_Returns_weight.append(weights[1])  
        Rotork_Returns_weight.append(weights[2]) 
        Ocado_Returns_weight .append(weights[3]) 
        Computacenter_Returns_weight.append(weights[4]) 
        TInfrastructure_Returns_weight.append(weights[5])
        Avast_Returns_weight.append(weights[6])  
        ASOS_Returns_weight.append(weights[7])
        Spirax_Returns_weight.append(weights[8])
        Electrocomponents_Returns_weight.append(weights[9])
        Trainline_Returns_weight.append(weights[10])
        Palantir_Returns_weight.append(weights[11])
    # Create an array of data for portfolio
    Portfolio_returns = np.array(Portfolio_returns)
    Portfolio_Volatilities = np.array(Portfolio_Volatilities)
    Portfolio_GrossR = np.array(Portfolio_GrossR)
    Aveva_Returns_Weight = np.array(Aveva_Returns_weight)
    Softcat_Returns_Weight = np.array(Softcat_Returns_weight)
    Rotork_Returns_Weight = np.array(Rotork_Returns_weight)
    Ocado_Returns_Weight = np.array(Ocado_Returns_weight)
    Computacenter_Returns_Weight = np.array(Computacenter_Returns_weight)
    TInfrastructure_Returns_Weight = np.array(TInfrastructure_Returns_weight)
    Avast_Returns_Weight = np.array(Avast_Returns_weight)
    ASOS_Returns_Weight = np.array(ASOS_Returns_weight)
    Spirax_Returns_Weight = np.array(Spirax_Returns_weight)
    Electrocomponents_Returns_Weight = np.array(Electrocomponents_Returns_weight)
    Trainline_Returns_Weight = np.array(Trainline_Returns_weight)
    Palantir_Returns_Weight = np.array(Palantir_Returns_weight)
    #Creating a table
    Portfolios = pd.DataFrame({'Return': Portfolio_returns, 
                           'Volatility': Portfolio_Volatilities,
                           'Gross Return': Portfolio_GrossR,
                           'Aveva Weight': Aveva_Returns_weight,
                           'Softcat Weight': Softcat_Returns_weight, 
                           'Rotork Weight': Rotork_Returns_weight,
                            'Ocado Weight': Ocado_Returns_weight,  
                            'Computacenter Weight': Computacenter_Returns_weight,
                            '3Infrastructure Weight': TInfrastructure_Returns_weight,
                            'Avast Weight': Avast_Returns_weight,
                            'ASOS Weight': ASOS_Returns_weight,
                            'Spirax Weight': Spirax_Returns_weight,
                            'Electrocomponents': Electrocomponents_Returns_weight,
                            'Trainline': Trainline_Returns_weight,
                            'Palantir': Palantir_Returns_weight})
    
    # Custom Portfolios
    # Within this return range, what is the maximum gross return? 
    Max_Greturn = Portfolios[(Portfolios['Gross Return']>=250.171) & (Portfolios['Gross Return']<=254.142)].max()['Return']
    Return = Portfolios.iloc[np.where(Portfolios['Return']== Max_Greturn)]
    Max_Greturn_2 = Portfolios[(Portfolios['Gross Return']>=250.100) & (Portfolios['Gross Return']<=254.100)].max()['Return']
    Return_2 = Portfolios.iloc[np.where(Portfolios['Return']== Max_Greturn_2)]
    Max_Greturn_3 = Portfolios[(Portfolios['Gross Return']>=250.200) & (Portfolios['Gross Return']<=254.000)].max()['Return']
    Return_3 = Portfolios.iloc[np.where(Portfolios['Return']== Max_Greturn_3)]
    Max_Greturn_4 = Portfolios[(Portfolios['Gross Return']>=251.300) & (Portfolios['Gross Return']<=253.900)].max()['Return']
    Return_4 = Portfolios.iloc[np.where(Portfolios['Return']== Max_Greturn_4)]
    Max_Greturn_5 = Portfolios[(Portfolios['Gross Return']>=251.400) & (Portfolios['Gross Return']<=253.800)].max()['Return']
    Return_5 = Portfolios.iloc[np.where(Portfolios['Return']== Max_Greturn_5)]
    Max_Greturn_6 = Portfolios[(Portfolios['Gross Return']>=251.500) & (Portfolios['Gross Return']<=253.700)].max()['Return']
    Return_6 = Portfolios.iloc[np.where(Portfolios['Return']== Max_Greturn_6)]
    Max_Greturn_7 = Portfolios[(Portfolios['Gross Return']>=251.600) & (Portfolios['Gross Return']<=253.600)].max()['Return']
    Return_7 = Portfolios.iloc[np.where(Portfolios['Return']== Max_Greturn_7)]
    Max_Greturn_8 = Portfolios[(Portfolios['Gross Return']>=251.700) & (Portfolios['Gross Return']<=253.500)].max()['Return']
    Return_8 = Portfolios.iloc[np.where(Portfolios['Return']== Max_Greturn_8)]
    Max_Greturn_9 = Portfolios[(Portfolios['Gross Return']>=251.800) & (Portfolios['Gross Return']<=253.400)].max()['Return']
    Return_9 = Portfolios.iloc[np.where(Portfolios['Return']== Max_Greturn_9)]
    Max_Greturn_10 = Portfolios[(Portfolios['Gross Return']>=251.900) & (Portfolios['Gross Return']<=253.300)].max()['Return']
    Return_10 = Portfolios.iloc[np.where(Portfolios['Return']== Max_Greturn_10)]
    Max_Greturn_10 = Portfolios[(Portfolios['Gross Return']>=252.100) & (Portfolios['Gross Return']<=253.200)].max()['Return']
    Return_10= Portfolios.iloc[np.where(Portfolios['Return']==Max_Greturn_10)]
    Max_Greturn_11 = Portfolios[(Portfolios['Gross Return']>=252.200) & (Portfolios['Gross Return']<=253.100)].max()['Return']
    Return_11= Portfolios.iloc[np.where(Portfolios['Return']==Max_Greturn_11)]
    Max_Greturn_12 = Portfolios[(Portfolios['Gross Return']>=252.300) & (Portfolios['Gross Return']<=253.000)].max()['Return']
    Return_12= Portfolios.iloc[np.where(Portfolios['Return']==Max_Greturn_12)]
    Max_Greturn_13 = Portfolios[(Portfolios['Gross Return']>=252.400) & (Portfolios['Gross Return']<=252.900)].max()['Return']
    Return_13= Portfolios.iloc[np.where(Portfolios['Return']==Max_Greturn_13)]
    Max_Greturn_14 = Portfolios[(Portfolios['Gross Return']>=252.500) & (Portfolios['Gross Return']<=252.800)].max()['Return']
    Return_14= Portfolios.iloc[np.where(Portfolios['Return']==Max_Greturn_14)]
    Max_Greturn_15 = Portfolios[(Portfolios['Gross Return']>=252.600) & (Portfolios['Gross Return']<=252.700)].max()['Return']
    Return_15= Portfolios.iloc[np.where(Portfolios['Return']==Max_Greturn_15)]
    Max_Greturn_16 = Portfolios[(Portfolios['Gross Return']>=252.700) & (Portfolios['Gross Return']<=252.600)].max()['Return']
    Return_16= Portfolios.iloc[np.where(Portfolios['Return']==Max_Greturn_16)]
    Max_Greturn_17 = Portfolios[(Portfolios['Gross Return']>=252.800) & (Portfolios['Gross Return']<=252.500)].max()['Return']
    Return_17= Portfolios.iloc[np.where(Portfolios['Return']==Max_Greturn_17)]
    Max_Greturn_18 = Portfolios[(Portfolios['Gross Return']>=252.900) & (Portfolios['Gross Return']<=252.400)].max()['Return']
    Return_18= Portfolios.iloc[np.where(Portfolios['Return']==Max_Greturn_18)]
    Max_Greturn_19 = Portfolios[(Portfolios['Gross Return']>=253.000) & (Portfolios['Gross Return']<=252.300)].max()['Return']
    Return_19= Portfolios.iloc[np.where(Portfolios['Return']==Max_Greturn_19)]
    
    Final_MaxGR_ROp = pd.concat([Return,Return_2, Return_3, Return_4, Return_5, Return_6,
                              Return_7, Return_8, Return_9,Return_10,Return_11,Return_12,
                              Return_13, Return_14, Return_15, Return_16, Return_17,
                              Return_18, Return_19])
    return Final_MaxGR_ROp

def Optimize_MaxGR_R():
    P_Rename = get_rtns()
    Portfolio_rtns = P_Rename
    Gross_rtn = Gross_return()
    #Number of assets in the portfolio
    tckr_list = ticker_list()
    Assets = tckr_list
    num_assets = len(Assets)
    # Lists for Portfolio creation
    Portfolio_returns = []
    Portfolio_Volatilities = []
    Portfolio_GrossR = []
    Aveva_Returns_weight = []
    Softcat_Returns_weight = []
    Rotork_Returns_weight = []
    Ocado_Returns_weight = []
    Computacenter_Returns_weight = [] 
    TInfrastructure_Returns_weight = []
    Avast_Returns_weight = []
    ASOS_Returns_weight = []
    Spirax_Returns_weight = []
    Electrocomponents_Returns_weight = [] 
    Trainline_Returns_weight = []
    Palantir_Returns_weight = []
    
    #Optimising for expected returns and standard deviation
    for x in range (100000):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        Portfolio_returns.append(np.sum(weights * Portfolio_rtns.mean() * 250)) # expected returns
        Portfolio_Volatilities.append(np.sqrt(np.dot(weights.T,np.dot(Portfolio_rtns.cov() * 250, weights)))) # standard deviation 
        Portfolio_GrossR.append(np.sum(weights * Gross_rtn.mean() * 250)) # Gross returns
        Aveva_Returns_weight.append(weights[0])
        Softcat_Returns_weight.append(weights[1])  
        Rotork_Returns_weight.append(weights[2]) 
        Ocado_Returns_weight .append(weights[3]) 
        Computacenter_Returns_weight.append(weights[4]) 
        TInfrastructure_Returns_weight.append(weights[5])
        Avast_Returns_weight.append(weights[6])  
        ASOS_Returns_weight.append(weights[7])
        Spirax_Returns_weight.append(weights[8])
        Electrocomponents_Returns_weight.append(weights[9])
        Trainline_Returns_weight.append(weights[10])
        Palantir_Returns_weight.append(weights[11])
    # Create an array of data for portfolio
    Portfolio_returns = np.array(Portfolio_returns)
    Portfolio_Volatilities = np.array(Portfolio_Volatilities)
    Portfolio_GrossR = np.array(Portfolio_GrossR)
    Aveva_Returns_Weight = np.array(Aveva_Returns_weight)
    Softcat_Returns_Weight = np.array(Softcat_Returns_weight)
    Rotork_Returns_Weight = np.array(Rotork_Returns_weight)
    Ocado_Returns_Weight = np.array(Ocado_Returns_weight)
    Computacenter_Returns_Weight = np.array(Computacenter_Returns_weight)
    TInfrastructure_Returns_Weight = np.array(TInfrastructure_Returns_weight)
    Avast_Returns_Weight = np.array(Avast_Returns_weight)
    ASOS_Returns_Weight = np.array(ASOS_Returns_weight)
    Spirax_Returns_Weight = np.array(Spirax_Returns_weight)
    Electrocomponents_Returns_Weight = np.array(Electrocomponents_Returns_weight)
    Trainline_Returns_Weight = np.array(Trainline_Returns_weight)
    Palantir_Returns_Weight = np.array(Palantir_Returns_weight)
    #Creating a table
    Portfolios = pd.DataFrame({'Return': Portfolio_returns, 
                           'Volatility': Portfolio_Volatilities,
                           'Gross Return': Portfolio_GrossR,
                           'Aveva Weight': Aveva_Returns_weight,
                           'Softcat Weight': Softcat_Returns_weight, 
                           'Rotork Weight': Rotork_Returns_weight,
                            'Ocado Weight': Ocado_Returns_weight,  
                            'Computacenter Weight': Computacenter_Returns_weight,
                            '3Infrastructure Weight': TInfrastructure_Returns_weight,
                            'Avast Weight': Avast_Returns_weight,
                            'ASOS Weight': ASOS_Returns_weight,
                            'Spirax Weight': Spirax_Returns_weight,
                            'Electrocomponents': Electrocomponents_Returns_weight,
                            'Trainline': Trainline_Returns_weight,
                            'Palantir': Palantir_Returns_weight})
    
    # Custom Portfolios
    # What would be the max gross return if return is within this range?  
    Max_GrossReturn = Portfolios[(Portfolios['Return']>=.179) & (Portfolios['Return']<=4.436)].max()['Gross Return']
    GrossReturn = Portfolios.iloc[np.where(Portfolios['Gross Return']== Max_GrossReturn)]
    Max_GrossReturn_2 = Portfolios[(Portfolios['Return']>=.200) & (Portfolios['Return']<=9.00)].max()['Gross Return']
    GrossReturn_2 = Portfolios.iloc[np.where(Portfolios['Gross Return']== Max_GrossReturn_2)]
    Max_GrossReturn_3 = Portfolios[(Portfolios['Return']>=.300) & (Portfolios['Return']<=8.00)].max()['Gross Return']
    GrossReturn_3 = Portfolios.iloc[np.where(Portfolios['Gross Return']== Max_GrossReturn_3)]
    Max_GrossReturn_4 = Portfolios[(Portfolios['Return']>=.400) & (Portfolios['Return']<=7.00)].max()['Gross Return']
    GrossReturn_4 = Portfolios.iloc[np.where(Portfolios['Gross Return']== Max_GrossReturn_4)]
    Max_GrossReturn_5 = Portfolios[(Portfolios['Return']>=.500) & (Portfolios['Return']<=6.00)].max()['Gross Return']
    GrossReturn_5 = Portfolios.iloc[np.where(Portfolios['Gross Return']== Max_GrossReturn_5)]
    Max_GrossReturn_6 = Portfolios[(Portfolios['Return']>=.600) & (Portfolios['Return']<=5.00)].max()['Gross Return']
    GrossReturn_6 = Portfolios.iloc[np.where(Portfolios['Gross Return']== Max_GrossReturn_6)]
    Max_GrossReturn_7 = Portfolios[(Portfolios['Return']>=.700) & (Portfolios['Return']<=4.00)].max()['Gross Return']
    GrossReturn_7 = Portfolios.iloc[np.where(Portfolios['Gross Return']== Max_GrossReturn_7)]
    Max_GrossReturn_8 = Portfolios[(Portfolios['Return']>=.800) & (Portfolios['Return']<=3.00)].max()['Gross Return']
    GrossReturn_8 = Portfolios.iloc[np.where(Portfolios['Gross Return']== Max_GrossReturn_8)]
    Max_GrossReturn_9 = Portfolios[(Portfolios['Return']>=.900) & (Portfolios['Return']<=2.00)].max()['Gross Return']
    GrossReturn_9 = Portfolios.iloc[np.where(Portfolios['Gross Return']== Max_GrossReturn_9)]
    Max_GrossReturn_10 = Portfolios[(Portfolios['Return']>=1.00) & (Portfolios['Return']<=1.00)].max()['Gross Return']
    GrossReturn_10 = Portfolios.iloc[np.where(Portfolios['Gross Return']== Max_GrossReturn_10)]
    
    Final_MaxReturn = pd.concat([GrossReturn,GrossReturn_2, GrossReturn_3, GrossReturn_4, GrossReturn_5, GrossReturn_6,
                        GrossReturn_7, GrossReturn_8, GrossReturn_9, GrossReturn_10])
    return Final_MaxReturn

def Optimize_Min_V_GR():
    P_Rename = get_rtns()
    Portfolio_rtns = P_Rename
    Gross_rtn = Gross_return()
    #Number of assets in the portfolio
    tckr_list = ticker_list()
    Assets = tckr_list
    num_assets = len(Assets)
    # Lists for Portfolio creation
    Portfolio_returns = []
    Portfolio_Volatilities = []
    Portfolio_GrossR = []
    Aveva_Returns_weight = []
    Softcat_Returns_weight = []
    Rotork_Returns_weight = []
    Ocado_Returns_weight = []
    Computacenter_Returns_weight = [] 
    TInfrastructure_Returns_weight = []
    Avast_Returns_weight = []
    ASOS_Returns_weight = []
    Spirax_Returns_weight = []
    Electrocomponents_Returns_weight = [] 
    Trainline_Returns_weight = []
    Palantir_Returns_weight = []
    
    #Optimising for expected returns and standard deviation
    for x in range (100000):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        Portfolio_returns.append(np.sum(weights * Portfolio_rtns.mean() * 250)) # expected returns
        Portfolio_Volatilities.append(np.sqrt(np.dot(weights.T,np.dot(Portfolio_rtns.cov() * 250, weights)))) # standard deviation 
        Portfolio_GrossR.append(np.sum(weights * Gross_rtn.mean() * 250)) # Gross returns
        Aveva_Returns_weight.append(weights[0])
        Softcat_Returns_weight.append(weights[1])  
        Rotork_Returns_weight.append(weights[2]) 
        Ocado_Returns_weight .append(weights[3]) 
        Computacenter_Returns_weight.append(weights[4]) 
        TInfrastructure_Returns_weight.append(weights[5])
        Avast_Returns_weight.append(weights[6])  
        ASOS_Returns_weight.append(weights[7])
        Spirax_Returns_weight.append(weights[8])
        Electrocomponents_Returns_weight.append(weights[9])
        Trainline_Returns_weight.append(weights[10])
        Palantir_Returns_weight.append(weights[11])
    # Create an array of data for portfolio
    Portfolio_returns = np.array(Portfolio_returns)
    Portfolio_Volatilities = np.array(Portfolio_Volatilities)
    Portfolio_GrossR = np.array(Portfolio_GrossR)
    Aveva_Returns_Weight = np.array(Aveva_Returns_weight)
    Softcat_Returns_Weight = np.array(Softcat_Returns_weight)
    Rotork_Returns_Weight = np.array(Rotork_Returns_weight)
    Ocado_Returns_Weight = np.array(Ocado_Returns_weight)
    Computacenter_Returns_Weight = np.array(Computacenter_Returns_weight)
    TInfrastructure_Returns_Weight = np.array(TInfrastructure_Returns_weight)
    Avast_Returns_Weight = np.array(Avast_Returns_weight)
    ASOS_Returns_Weight = np.array(ASOS_Returns_weight)
    Spirax_Returns_Weight = np.array(Spirax_Returns_weight)
    Electrocomponents_Returns_Weight = np.array(Electrocomponents_Returns_weight)
    Trainline_Returns_Weight = np.array(Trainline_Returns_weight)
    Palantir_Returns_Weight = np.array(Palantir_Returns_weight)
    #Creating a table
    Portfolios = pd.DataFrame({'Return': Portfolio_returns, 
                           'Volatility': Portfolio_Volatilities,
                           'Gross Return': Portfolio_GrossR,
                           'Aveva Weight': Aveva_Returns_weight,
                           'Softcat Weight': Softcat_Returns_weight, 
                           'Rotork Weight': Rotork_Returns_weight,
                            'Ocado Weight': Ocado_Returns_weight,  
                            'Computacenter Weight': Computacenter_Returns_weight,
                            '3Infrastructure Weight': TInfrastructure_Returns_weight,
                            'Avast Weight': Avast_Returns_weight,
                            'ASOS Weight': ASOS_Returns_weight,
                            'Spirax Weight': Spirax_Returns_weight,
                            'Electrocomponents': Electrocomponents_Returns_weight,
                            'Trainline': Trainline_Returns_weight,
                            'Palantir': Palantir_Returns_weight})
    
    # Custom Portfolios
    # Within this return range, what is the maximum gross return? 
    Max_Greturn = Portfolios[(Portfolios['Gross Return']>=250.171) & (Portfolios['Gross Return']<=254.142)].min()['Volatility']
    Return = Portfolios.iloc[np.where(Portfolios['Volatility']== Max_Greturn)]
    Max_Greturn_2 = Portfolios[(Portfolios['Gross Return']>=250.100) & (Portfolios['Gross Return']<=254.100)].min()['Volatility']
    Return_2 = Portfolios.iloc[np.where(Portfolios['Volatility']== Max_Greturn_2)]
    Max_Greturn_3 = Portfolios[(Portfolios['Gross Return']>=250.200) & (Portfolios['Gross Return']<=254.000)].min()['Volatility']
    Return_3 = Portfolios.iloc[np.where(Portfolios['Volatility']== Max_Greturn_3)]
    Max_Greturn_4 = Portfolios[(Portfolios['Gross Return']>=251.300) & (Portfolios['Gross Return']<=253.900)].min()['Volatility']
    Return_4 = Portfolios.iloc[np.where(Portfolios['Volatility']== Max_Greturn_4)]
    Max_Greturn_5 = Portfolios[(Portfolios['Gross Return']>=251.400) & (Portfolios['Gross Return']<=253.800)].min()['Volatility']
    Return_5 = Portfolios.iloc[np.where(Portfolios['Volatility']== Max_Greturn_5)]
    Max_Greturn_6 = Portfolios[(Portfolios['Gross Return']>=251.500) & (Portfolios['Gross Return']<=253.700)].min()['Volatility']
    Return_6 = Portfolios.iloc[np.where(Portfolios['Volatility']== Max_Greturn_6)]
    Max_Greturn_7 = Portfolios[(Portfolios['Gross Return']>=251.600) & (Portfolios['Gross Return']<=253.600)].min()['Volatility']
    Return_7 = Portfolios.iloc[np.where(Portfolios['Volatility']== Max_Greturn_7)]
    Max_Greturn_8 = Portfolios[(Portfolios['Gross Return']>=251.700) & (Portfolios['Gross Return']<=253.500)].min()['Volatility']
    Return_8 = Portfolios.iloc[np.where(Portfolios['Volatility']== Max_Greturn_8)]
    Max_Greturn_9 = Portfolios[(Portfolios['Gross Return']>=251.800) & (Portfolios['Gross Return']<=253.400)].min()['Volatility']
    Return_9 = Portfolios.iloc[np.where(Portfolios['Volatility']== Max_Greturn_9)]
    Max_Greturn_10 = Portfolios[(Portfolios['Gross Return']>=251.900) & (Portfolios['Gross Return']<=253.300)].min()['Volatility']
    Return_10 = Portfolios.iloc[np.where(Portfolios['Volatility']== Max_Greturn_10)]
    Max_Greturn_10 = Portfolios[(Portfolios['Gross Return']>=252.100) & (Portfolios['Gross Return']<=253.200)].min()['Volatility']
    Return_10= Portfolios.iloc[np.where(Portfolios['Volatility']==Max_Greturn_10)]
    Max_Greturn_11 = Portfolios[(Portfolios['Gross Return']>=252.200) & (Portfolios['Gross Return']<=253.100)].min()['Volatility']
    Return_11= Portfolios.iloc[np.where(Portfolios['Volatility']==Max_Greturn_11)]
    Max_Greturn_12 = Portfolios[(Portfolios['Gross Return']>=252.300) & (Portfolios['Gross Return']<=253.000)].min()['Volatility']
    Return_12= Portfolios.iloc[np.where(Portfolios['Volatility']==Max_Greturn_12)]
    Max_Greturn_13 = Portfolios[(Portfolios['Gross Return']>=252.400) & (Portfolios['Gross Return']<=252.900)].min()['Volatility']
    Return_13= Portfolios.iloc[np.where(Portfolios['Volatility']==Max_Greturn_13)]
    Max_Greturn_14 = Portfolios[(Portfolios['Gross Return']>=252.500) & (Portfolios['Gross Return']<=252.800)].min()['Volatility']
    Return_14= Portfolios.iloc[np.where(Portfolios['Volatility']==Max_Greturn_14)]
    Max_Greturn_15 = Portfolios[(Portfolios['Gross Return']>=252.600) & (Portfolios['Gross Return']<=252.700)].min()['Volatility']
    Return_15= Portfolios.iloc[np.where(Portfolios['Volatility']==Max_Greturn_15)]
    Max_Greturn_16 = Portfolios[(Portfolios['Gross Return']>=252.700) & (Portfolios['Gross Return']<=252.600)].min()['Volatility']
    Return_16= Portfolios.iloc[np.where(Portfolios['Volatility']==Max_Greturn_16)]
    Max_Greturn_17 = Portfolios[(Portfolios['Gross Return']>=252.800) & (Portfolios['Gross Return']<=252.500)].min()['Volatility']
    Return_17= Portfolios.iloc[np.where(Portfolios['Volatility']==Max_Greturn_17)]
    Max_Greturn_18 = Portfolios[(Portfolios['Gross Return']>=252.900) & (Portfolios['Gross Return']<=252.400)].min()['Volatility']
    Return_18= Portfolios.iloc[np.where(Portfolios['Volatility']==Max_Greturn_18)]
    Max_Greturn_19 = Portfolios[(Portfolios['Gross Return']>=253.000) & (Portfolios['Gross Return']<=252.300)].min()['Volatility']
    Return_19= Portfolios.iloc[np.where(Portfolios['Volatility']==Max_Greturn_19)]
    
    Final_MaxGR_ROp = pd.concat([Return,Return_2, Return_3, Return_4, Return_5, Return_6,
                              Return_7, Return_8, Return_9,Return_10,Return_11,Return_12,
                              Return_13, Return_14, Return_15, Return_16, Return_17,
                              Return_18, Return_19])
    return Final_MaxGR_ROp

