import pandas as pd
from scipy.optimize import minimize
from pandas_datareader import data as wb
import numpy as np
import math

    
#AV_r = Aveva_Group.pct_change()
# replace 'nan' value with 0
#AV_rClean = AV_r[Columns].fillna(0)
# Replace with the rename function
#Aveva = AV_rClean


def drawdown(return_series: pd.Series):
    """This function defines the drawdown of a stock
        calculates the wealth index
        the previous peaks,
        the percentage drawdown"""
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index-previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdown": drawdowns,
    })

def drawdown_Portfolio(r):
    """This function defines the drawdown of a stock
        calculates the wealth index
        the previous peaks,
        the percentage drawdown"""
    for column in r:
        wealth_index = 1000*(1+r).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index-previous_peaks)/previous_peaks
        Drawdown_stats = pd.concat([drawdowns], axis=1)
        Drawdown_summary = Drawdown_stats.mean()
        Largest_drawdown = Drawdown_stats.min()
        When = Drawdown_stats.idxmin()
        final_stats = pd.concat([Drawdown_summary, Largest_drawdown, When], axis=1)
        final_drawdown_stats = final_stats.rename(columns={0:'Drawdown', 1:'Highest Drawdown', 2:'When Drawdown Happened'}) 
        return final_drawdown_stats



#def rtns():
    # Get the data from yahoo finance
 #   ThreeIn = wb.DataReader('3IN.L', data_source='yahoo', start='2009-01-01')
    # Choose just the adjusted price
   # Columns = ['Adj Close']
  #  Aveva_Group = ThreeIn[Columns]
    #AV_r = Aveva_Group.pct_change()
    # replace 'nan' value with 0
    #AV_rClean = AV_r[Columns].fillna(0)
    # Replace with the rename function
    #Aveva = AV_rClean.rename(columns = {'Adj Close': '3i Infrastructure Returns'})
    #return Aveva


def returns():
    import pandas as pd
    sheet_id = "1u4NmVoBZRif-ng8PVT5r5pfAQpbb9O86H-hnoLkqyOw"
    sheet_name = "Values"
    gsheet_url ="https://docs.google.com/spreadsheets/d/{}/gviz/tq?tqx=out:csv&sheet={}".format(sheet_id, sheet_name)
    df = pd.read_csv(gsheet_url)
    return print(df)

def compound(r,periods_per_year):
    """
    returns the result of compounding the set of returns in r
    """
    return np.expm1(np.log1p(r).sum())*(periods_per_year**0.5)

def annual_returns(r, periods_per_year):
    """
    Annualizes a set of returns
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1


def annual_volatility(r, periods_per_year):
    return r.std()*np.sqrt(periods_per_year) 

def sharpe_ratio_annual(r, riskfree_rate, periods_per_year):
    rf_per_period = (1 + riskfree_rate)**(1/periods_per_year)-1
    excess_return = r - rf_per_period
    ann_ex_ret = annual_returns(excess_return, periods_per_year)
    ann_vol = annual_volatility(r, periods_per_year)
    return ann_ex_ret/ann_vol

def sortino_ratio_annual(r, riskfree_rate, periods_per_year):
    rf_per_period = (1 + riskfree_rate)**(1/periods_per_year)-1
    excess_return = r - rf_per_period
    ann_ex_ret = annual_returns(excess_return, periods_per_year)
    Neg_rtns_vol = Negative_risk_annual(r, periods_per_year)
    return ann_ex_ret/Neg_rtns_vol
    
def Positive_rtns_annual(r, periods_per_year):
    Positive_rtns = r[r>0]
    return annual_returns(Positive_rtns, periods_per_year)

def Negative_rtns_annual(r, periods_per_year):
    negative_rtns = r[r<0]
    return annual_returns(negative_rtns, periods_per_year)

def Negative_Positive_rtn_ratio(p,n):
    P_N_Risk_Return_ratio = n/p
    return P_N_Risk_Return_ratio
    
def Positive_risk_annual(r, periods_per_year):
    Positive_rtns = r[r>0]
    return annual_volatility(Positive_rtns, periods_per_year)

def Negative_risk_annual(r, periods_per_year):
    negative_rtns = r[r<0]
    return annual_volatility(negative_rtns, periods_per_year)

def Positive_Negative_vol_ratio(pv,nv):
    return nv/pv

def Portfolio_stats(r,riskfree_rate, periods_per_year):
    ann_rtns = annual_returns(r, periods_per_year)
    ann_vol = annual_volatility(r, periods_per_year)
    sharpe = sharpe_ratio_annual(r, riskfree_rate, periods_per_year)
    sortino = sortino_ratio_annual(r, riskfree_rate, periods_per_year)
    drawdown = drawdown_Portfolio(r)
    p = Positive_rtns_annual(r, periods_per_year)
    n = Negative_rtns_annual(r, periods_per_year)
    P_n_rtns_ratio = Negative_Positive_rtn_ratio(p,n)
    P_vol = Positive_risk_annual(r, periods_per_year)
    N_vol = Negative_risk_annual(r, periods_per_year)
    P_N_vol_ratio = Positive_Negative_vol_ratio(P_vol, N_vol)
    stats = pd.concat([ann_rtns, ann_vol, sharpe, sortino, drawdown, p, n, P_n_rtns_ratio, P_vol, N_vol, P_N_vol_ratio], axis=1) 
    Portfolio_stats = stats.rename(columns={0:'Annual Returns', 1:'Annual Volatility', 2:'Sharpe Ratio', 3:'Sortino Ratio', 4:'Positive Returns', 5:'Negative Returns', 6:'Negative Positive return Ratio',
                                            7:'Positive Risk', 8:'Negative Risk', 9:'Negative Positive Risk Ratio'})
    return Portfolio_stats
 



    
    





#def returns():
   # Assets = ['AVV.L', 'ROR.L', 'CCC.L', 'OCDO.L', '3IN.L', 'SCT.L', 'TRN.L', 'AVST.L', 'ASC.L', 
    #          'SPX.L','ECM.L','PLTR']
    #Portfolio = pd.DataFrame()
    #for a in Assets:
    #    Portfolio[a] = wb.DataReader(a,  data_source='yahoo', start='2009-1-1')['Adj Close'] 
    #A_rtns = Portfolio.pct_change()
    #A_rtns_Clean = A_rtns.fillna(0)
    #A_final = A_rtns_Clean.rename(columns = {'Adj Close': 'Returns'})
    #return A_final

# Show the returns graph
def show(p):
    return p.plot.line(figsize=(10,8))

def portfolio_returns(weights, returns):
    """weights -> returns"""
    
    # take the weights, transpose it and take the matrix multiplication
    return weights.T @ returns

# Volatility
def portfolio_volatility(weights, covmat):
    """Weights -> Covariance"""
    
    # Weights transposes, matrix multiply with covmatrix and matrix multiply this with weights and square root the answer
    return (weights.T @ covmat @ weights)**0.5 # square root of this function

def portfolio_sortino(weights, r, riskfree_rate, periods_per_year):
    """ Weights -> Portfolio Sortino"""
    sortino = sortino_ratio(r, riskfree_rate, periods_per_year)
    return weights.T @ sortino
    

    
def plot_two_asset_Portfolio(n_points, er, cov, style=""):
        
    if er.shape[0] !=2 or er.shape[0] !=2:
        raise ValueError("This portfolio can only be plotted using two assets!")

    # so for whatever w is, subtract 1 from it for every weight in a linear space that starts from 0 to 1 but only generate 20 numbers
    weights = [np.array([w, 1-w]) for w in np.linspace(0,1,n_points)]
    # get returns from this weights
    Returns = [portfolio_returns(w,er) for w in weights]
    Covariance = [portfolio_volatility(w,cov) for w in weights]
    Portfolio_final = pd.DataFrame({"Returns":Returns, "Volatility": Covariance})
    return Portfolio_final.plot.line(x="Volatility", y="Returns");

# Minimum volatility for a certain return (finding the weights)
def minimize_vol (target_return, er, cov):
    """Plot a list of weights that will achieve minimal vol for the portfolio"""
    # number of assets
    n = er.shape[0]
    # guess weights to achieve goal
    initial_guess = np.repeat(1/n, n)
    # make copies of this boundary for every asset
    boundary = ((0.0, 1.0),)*n
    # Return should be whatever the target is
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_returns(weights, er) # telling the function to find weights for your target return. This will be achieved only when the calculation (Target return - portfolio returns = 0). We do this because then we have created a weighted average at which the portfolio returns = my target return. 
        
    }
    # weights should equal one
    weights_sum_1 = {
        'type':'eq',
        'fun': lambda weights: np.sum(weights) - 1 # weights are found when this function = 0
    }
    # Optimiser (this is the weights)
    results = minimize(portfolio_volatility, initial_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(return_is_target, weights_sum_1),
                       bounds=boundary)
    return results.x

# Maximise Sortino for a certain return (finding the weights)
def minimize_sortino(risk_free_rate,target_return,er):
    """Plot a list of weights that will achieve minimal vol for the portfolio"""
    # number of assets
    n = er.shape[0]
    # guess weights to achieve goal
    initial_guess = np.repeat(1/n, n)
    # make copies of this boundary for every asset
    boundary = ((0.0, 1.0),)*n
    # Return should be whatever the target is
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_returns(weights, er) # telling the function to find weights for your target return. This will be achieved only when the calculation (Target return - portfolio returns = 0). We do this because then we have created a weighted average at which the portfolio returns = my target return. 
        
    }
    # weights should equal one
    weights_sum_1 = {
        'type':'eq',
        'fun': lambda weights: np.sum(weights) - 1 # weights are found when this function = 0
    }
    
    def portfolio_sortino(weights, er, riskfree_rate, periods_per_year):
        """ Weights -> Portfolio Sortino"""
        sortino = sortino_ratio(er, riskfree_rate, periods_per_year)
        return sortino.multiply(weights, axis=0)
    
    # Optimiser (this is the weights)
    results = minimize(portfolio_sortino, initial_guess,
                       args=(er,riskfree_rate,periods_per_year,), method='SLSQP',
                       options={'disp': False},
                       constraints=(return_is_target, weights_sum_1),
                       bounds=boundary)
    return results.x

# minimum vol for a certain return
def Sharpe_ratio(risk_free_rate,er,cov):
    """Plot a list of weights that will achieve max sharpe ratio for the portfolio"""
    # number of assets
    n = er.shape[0]
    # guess weights to achieve goal
    initial_guess = np.repeat(1/n, n)
    # make copies of this boundary for every asset
    boundary = ((0.0, 1.0),)*n
    # weights should equal one
    weights_sum_1 = {
        'type':'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    # function to calculate sharpe ratio
    def neg_sharpe_ratio(weights,risk_free_rate,er,cov):
        # portfolio return
        r = portfolio_returns(weights, er)
        # volatility return
        vol = portfolio_volatility(weights, cov)
        return -(r-risk_free_rate)/vol
    
    # Optimiser (maximise sharpe ratio by minimising the negative sharpe ratio)
    results = minimize(neg_sharpe_ratio, initial_guess,
                       args=(risk_free_rate,er,cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_1),
                       bounds=boundary)
    return results.x

# Target weights
def optimal_weights(n_points, er, cov):
    """ Get a list of weights for min and max returns"""
    # generate the target return give the min and max returns
    target_rtns = np.linspace(er.min(), er.max(), n_points)
    # for target rtns, loop through the function for what this would be and give me a set of weights
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rtns]
    return weights

def gmv(cov):
    """Returns weights of the Global Minimum 
    Variance Portfolio given the covariance matrix"""
    
    n = cov.shape[0]
    return Sharpe_ratio(0, np.repeat(1,n), cov)

def ewp(er,cov):
    n=er.shape[0]
    return np.repeat(1/n, n)
    
    

# multi asset portfolio for mimimum volatility portfolio. Built the capital market line for this minimum portfolio
def plot_EF_Portfolio(n_points, er, cov, style="", show_cml=False, riskfree_rate=0, show_ew=False,show_GMV=False):
    """
    Plot Efficient portfolio for n assets
    """
    weights = optimal_weights(n_points, er, cov)
    Returns = [portfolio_returns(w,er) for w in weights]
    Covariance = [portfolio_volatility(w,cov) for w in weights]
    Portfolio_final = pd.DataFrame({"Returns":Returns, 
                                    "Volatility":Covariance
                                   })
    ax = Portfolio_final.plot.scatter(x="Volatility", y="Returns", alpha=0.3,grid=True, style=style, figsize=(15,12), legend=True);    
    # if we choose an equally weighted portfolio...
    if show_ew:
        n=er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_returns(w_ew, er)
        vol_ew = portfolio_volatility(w_ew, cov)
        # plot EW 
        ax.plot([r_ew], [r_ew], color = "goldenrod", marker="o", markersize=12) 
        
     # if we choose an GMV weighted portfolio...
    if show_GMV:
        w_gmv = gmv(cov)
        r_gmv= portfolio_returns(w_gmv, er)
        vol_gmv = portfolio_volatility(w_gmv, cov)
        # plot EW 
        ax.plot([r_gmv], [vol_gmv], color = "midnightblue", marker="o", markersize=12) 
    
    # if we choose to show the Capital market line...
    if show_cml:
        ax.set_xlim(left = 0) # let the axist start from 0
        w_msr = Sharpe_ratio(riskfree_rate, er, cov)
        r_msr = portfolio_returns(w_msr,er)
        v_msr = portfolio_volatility(w_msr,cov)
        # Add cml
        cml_x = [0, v_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x,cml_y, color="pink", marker="o", linestyle="dashed", markersize=12, linewidth=2) 
        
        
def Optimal_Portfolio_Min_Vol(n_points, er, cov):
    """
    The final wewight values of the optimal portfolio 
    """
    weights = optimal_weights(n_points, er, cov)
    Returns = [portfolio_returns(w,er) for w in weights]
    Volatility = [portfolio_volatility(w,cov) for w in weights]
    Portfolio_final = pd.DataFrame({"Returns":Returns, 
                                    "Volatility": Volatility
                                   })
    return weights, Returns, Volatility 
    
# Getting the skewness of the return  
def skew_table(Assets_return):
    skewness_table = pd.concat([Assets_return.mean(), Assets_return.median(), Assets_return.mean()>Assets_return.median()], axis="columns")
    cols = ['Mean', 'Median', 'Skewed']
    skewness_table.columns = cols        
    return skewness_table

# Calculating the skewness of a stock's return
def skewness(r):
    """Alternative to scipy.stats.skew()"""
    expected_rtn = r - r.mean()
    # Use the population standard deviation
    sigma_r = r.std(ddof=0)
    exp = (expected_rtn**3).mean()
    return exp/sigma_r**3

#Kurtosis
def kurtosis(r):
    """Alternative to scipy.stats.Kurtosis()"""
    expected_rtn = r - r.mean()
    # Use the population standard deviation
    sigma_r = r.std(ddof=0)
    exp = (expected_rtn**4).mean()
    return exp/sigma_r**4

# jarque_bera test - have a 1% confidence level that this thing is normal
def is_normal(r, level = 0.01):
    import scipy.stats
    """Applies the jarque_bera test to see if a data is normal or not.
    Returns true or false"""
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level

# Semi-deviation calculation
def semi_dev(r):
    """Returns the semi-deviation"""
    is_negative = r < 0
    return r[is_negative].std(ddof=0)

def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    """
    if isinstance(r, pd.Series):
        is_negative = r < 0
        return r[is_negative].std(ddof=0)
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(semideviation)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

#VaR historic calculation
def var_historic(r, level=5):
    """
    VaR Historic
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return - np.percentile(r, level) #Don't report VaR as negative number
    else:
        raise TypeError("Expected r to be DataFrame or Series")

#VaR Gaussian
from scipy.stats import norm
def var_gaussian(r, level=5):
    # Compute the z-score assuming the returns were gaussian
    z = norm.ppf(level/100)
    return -(r.mean() + z *r.std(ddof=0))


#Modified Cornish Fisher 
def var_gaussian(r, level=5, modified=False):
    # Compute the z-score assuming the returns were gaussian
    z = norm.ppf(level/100)
    return -(r.mean() + z *r.std(ddof=0))
    if modified:
        #modify z score based on skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                 (z**2-1)*s/6 +
                 (z**3*z)*(k-3)/24 - 
                 (2*z**3 - 5*2)*(s**2)/36

            )
        
        return -(r.mean() + z *r.std(ddof=0))
            
#CVar       
def cvar_historic(r, level=5):
    """Computes cVar of pd series or table"""
    if isinstance(r, pd.Series):
        is_beyond = r<= -var_historic(r, level=level)
        return - r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else: 
        raise TypeError("Expected r to be a Series or DataFrame")
        
        
# Backtest

# Volatility
def portfolio_volatility_b(weights, covmat):
    """Weights -> Covariance"""
    
    # Weights transposes, matrix multiply with covmatrix and matrix multiply this with weights and square root the answer
    b = (weights.T).dot(covmat)
    c = b.dot(weights)
    return c**0.5

def portfolio_returns_b(weights, returns):
    """weights -> returns"""
    
    # take the weights, transpose it and take the matrix multiplication
    return (weights.T).dot(returns)

def Optimal_Portfolio_Min_VolB(n_points, er, cov):
    """
    The final wewight values of the optimal portfolio 
    """
    weights = optimal_weights_b(n_points, er, cov)
    Returns = [portfolio_returns_b(w,er) for w in weights]
    Volatility = [portfolio_volatility_b(w,cov) for w in weights]
    Portfolio_final = pd.DataFrame({"Returns":Returns, 
                                    "Volatility": Volatility
                                   })
    return weights

# Minimum volatility for a certain return (finding the weights)
def minimize_vol_b(target_return, er, cov):
    """Plot a list of weights that will achieve minimal vol for the portfolio"""
    from scipy.optimize import minimize
    import numpy as np
    # number of assets
    n = er.shape[0]
    # guess weights to achieve goal
    initial_guess = np.repeat(1/n, n)
    # make copies of this boundary for every asset
    boundary = ((0.0, 1.0),)*n
    # Return should be whatever the target is
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_returns_b(weights, er) # telling the function to find weights for your target return. This will be achieved only when the calculation (Target return - portfolio returns = 0). We do this because then we have created a weighted average at which the portfolio returns = my target return. 
        
    }
    # weights should equal one
    weights_sum_1 = {
        'type':'eq',
        'fun': lambda weights: np.sum(weights) - 1 # weights are found when this function = 0
    }
    # Optimiser (this is the weights)
    results = minimize(portfolio_volatility_b, initial_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(return_is_target, weights_sum_1),
                       bounds=boundary)
    return results.x

# Target weights
def optimal_weights_b(n_points, er, cov):
    """ Get a list of weights for min and max returns"""
    # generate the target return give the min and max returns
    target_rtns = np.linspace(er.min(), er.max(), n_points)
    # for target rtns, loop through the function for what this would be and give me a set of weights
    weights = [minimize_vol_b(target_return, er, cov) for target_return in target_rtns]
    return weights

def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    # set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = account_value
    if isinstance(risky_r, pd.Series): 
        risky_r = pd.DataFrame(risky_r, columns=["R"])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12 # fast way to set all values to a number
    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    safe_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floorval_history = pd.DataFrame().reindex_like(risky_r)
    peak_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        # recompute the new account value at the end of this step
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        # save the histories for analysis and plotting
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        safe_w_history.iloc[step] = safe_w
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak
    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth, 
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "Safe Allocation": safe_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r":risky_r,
        "safe_r": safe_r,
        "drawdown": drawdown,
        "peak": peak_history,
        "floor": floorval_history
    }
    
    return backtest_result

def summary_stats(r, riskfree_rate, periods_per_year):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annual_returns, periods_per_year)
    ann_vol = r.aggregate(annual_volatility, periods_per_year)
    ann_sr = r.aggregate(sharpe_ratio_annual, riskfree_rate, periods_per_year)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })

def gbm(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return (estimation of where what we are trying to predict has been heading in the past. It is the best estimation of the future we have)
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val

def show_gbm(n_scenerios, mu, sigma):
    """
    Plot Brownian motion results
    """
    s_0 = 100
    prices = ed.gbm(n_scenarios=n_scenerios, mu=mu, sigma=sigma, s_0=s_0)
    ax = prices.plot(legend=False,colour='indianred',alpha=0.5,linewidth=2, figsize=(15,9))
    ax.axhline(y=100, ls=":", color="black")
    # draw a dot at the origin
    ax.plot(0, s_0, marker='o', color='darkred', alpha=0.2)
    
def discount(t, r):
    """
    Compute the price of a pure discount bond that pays a dollar at time period t
    and r is the per-period interest rate
    """
    return (1+r)**(-t) 

def discount2(t, r):
    """
    Compute the price of a pure discount bond that pays a dollar at time period t
    and r is the per-period interest rate
    returns a |t| x |r| Series or DataFrame
    r can be a float, Series or DataFrame
    returns a DataFrame indexed by t
    """
    discounts = pd.DataFrame([(r+1)**-i for i in t])
    discounts.index = t
    return discounts
    
def pv(liabilities, r):
    """
    Compute the present value of a sequence of cash flows given by the time (as an index) and amounts
    r. What the present value of cash flows? 
    """
    dates = liabilities.index
    discounts = discount(dates, r)
    return (discounts*liabilities).sum()

def pv2(flows, r):
    """
    Compute the present value of a sequence of cash flows given by the time (as an index) and amounts
    r can be a scalar, or a Series or DataFrame with the number of rows matching the num of rows in flows
    """
    dates = flows.index
    discounts = discount2(dates, r)
    return discounts.multiply(flows, axis='rows').sum()

def funding_ratio(assets, liabilities, r):
    """
    Computes the funding ratio of a series of liabilities, based on an interest rate and current value of assets. 
    How does your funding ration change given your assets, liabilities and interest rate change? 
    """
    return assets/pv(liabilities, r)

def funding_ratio_series(assets, liabilities, r): # when we have multitudes of cashflows
    """
    Computes the funding ratio of a series of liabilities, based on an interest rate and current value of assets. 
    How does your funding ration change given your assets, liabilities and interest rate change? 
    """
    return pv(assets, r)/pv(liabilities, r)
    
def show_funding_ratio(assets, r):
    fr = funding_ratio(assets, liabilities, r)
    return print(f'{fr*100:.2f}')

def inst_to_ann(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.log1p(r)

def cir(n_years = 10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    """
    if r_0 is None: r_0 = b 
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year) + 1 # because n_years might be a float
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    ## For Price Generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    ####

    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    ####
    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        prices[step] = price(n_years-step*dt, rates[step])

    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    ### for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    ###
    return rates, prices

def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """
    Returns the series of cash flows generated by a bond,
    indexed by the payment/coupon number
    """
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt = principal*coupon_rate/coupons_per_year
    coupons = np.repeat(coupon_amt, n_coupons)
    coupon_times = np.arange(1, n_coupons+1)
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal
    return cash_flows

def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """
    Computes the price of a bond that pays regular coupons until maturity
    at which time the principal and the final coupon is returned
    This is not designed to be efficient, rather,
    it is to illustrate the underlying principle behind bond pricing!
    If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon date
    and the bond value is computed over time.
    i.e. The index of the discount_rate DataFrame is assumed to be the coupon number
    """
    if isinstance(discount_rate, pd.DataFrame): 
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year,
                                      discount_rate.loc[t])
        return prices
    else: # base case ... single time period
        if maturity <= 0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)
    
def bond_price_dataframe(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """
    Computes the price of a bond that pays regular coupons until maturity
    at which time the principal and the final coupon is returned
    This is not designed to be efficient, rather,
    it is to illustrate the underlying principle behind bond pricing!
    If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon date
    and the bond value is computed over time.
    i.e. The index of the discount_rate DataFrame is assumed to be the coupon number
    """
    if isinstance(discount_rate, pd.DataFrame): 
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price_dataframe(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year,
                                      discount_rate.loc[t])
        return prices
    else: # base case ... single time period
        if maturity <= 0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv2(cash_flows, discount_rate/coupons_per_year)    


def macaulay_duration(flows, discount_rate):
    """
    Computes the Macaulay Duration of a sequence of cash flows, given a per-period discount rate.
    It measures the relationship between the duration of the bond and price based on interest rates. If the duration is large
    then price will move a lot, the opposite for short duration. 
    """
    discounted_flows = discount(flows.index, discount_rate)*flows
    weights = discounted_flows/discounted_flows.sum()
    return np.average(flows.index, weights=weights)

def macaulay_duration2(flows, discount_rate):
    """
    Computes the Macaulay Duration of a sequence of cash flows, given a per-period discount rate.
    It measures the relationship between the duration of the bond and price based on interest rates. If the duration is large
    then price will move a lot, the opposite for short duration. 
    """
    discounted_flows = discount(flows.index, discount_rate)*pd.DataFrame(flows)
    weights = discounted_flows/discounted_flows.sum()
    return np.average(flows.index, weights=weights.iloc[:,0])

def match_durations(cf_t, cf_s, cf_l, discount_rate):
    """
    Returns the weight W in cf_s that, along with (1-W) in cf_l will have an effective
    duration that matches cf_t
    """
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    return (d_l - d_t)/(d_l - d_s)

def match_durations2(cf_t, cf_s, cf_l, discount_rate):
    """
    Returns the weight W in cf_s that, along with (1-W) in cf_l will have an effective
    duration that matches cf_t
    """
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    return (d_l - d_t)/(d_l - d_s)

def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    Computes the total return of a Bond based on monthly bond prices and coupon payments
    Assumes that dividends (coupons) are paid out at the end of the period (e.g. end of 3 months for quarterly div)
    and that dividends are reinvested in the bond
    """
    coupons = pd.DataFrame(data = 0, index=monthly_prices.index, columns=monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype=int)
    coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
    total_returns = (monthly_prices + coupons)/monthly_prices.shift()-1
    return total_returns.dropna()