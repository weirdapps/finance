# first import the libraries

import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import yfinance as yf
import pandas as pd

# read the csv file
df = pd.read_csv('/Users/plessas/Downloads/marketcap.csv')

# convert the csv file to a list
symbol_list = df['Symbol'].tolist()
symbol_list = symbol_list[0:50]

# connect to yahoo finance and download daily adjusted close prices for symbol_list
data = yf.download(symbol_list, start="2000-01-01", end="2023-05-20")

# calculate the daily returns
daily_returns = data['Adj Close'].pct_change()

# optimize the portfolio by maximizing the sharpe ratio


def get_ret_vol_sr(weights):
    """
    Takes in weights, returns array or return, volatility, sharpe ratio
    """
    weights = np.array(weights)
    ret = np.sum(daily_returns.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(
        daily_returns.cov() * 252, weights)))
    sr = ret/vol
    return np.array([ret, vol, sr])


def neg_sharpe(weights):
    return get_ret_vol_sr(weights)[2] * -1


def check_sum(weights):
    return np.sum(weights) - 1


cons = ({'type': 'eq', 'fun': check_sum})

bounds = ((0, 1),) * len(symbol_list)

init_guess = [1/len(symbol_list)] * len(symbol_list)

opt_results = minimize(neg_sharpe, init_guess,
                       method='SLSQP', bounds=bounds, constraints=cons)

opt_results

get_ret_vol_sr(opt_results.x)

# calculate the efficient frontier
frontier_y = np.linspace(0, 0.3, 100)


def minimize_volatility(weights):
    return get_ret_vol_sr(weights)[1]


frontier_volatility = []

for possible_return in frontier_y:
    cons = ({'type': 'eq', 'fun': check_sum},
            {'type': 'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})

    result = minimize(minimize_volatility, init_guess,
                      method='SLSQP', bounds=bounds, constraints=cons)

    frontier_volatility.append(result['fun'])

plt.figure(figsize=(12, 8))
plt.scatter(frontier_volatility, frontier_y, c=frontier_y /
            frontier_volatility, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.show()

# print a table of the stock symbols, the company names and the optimal weights in percent with two decimal places
optimal_weights = opt_results.x
optimal_weights = optimal_weights * 100
optimal_weights = optimal_weights.round(2)
optimal_weights = pd.DataFrame(optimal_weights, columns=['Optimal Weights'])
optimal_weights['Symbol'] = symbol_list
optimal_weights['Company Name'] = df['Name']
optimal_weights = optimal_weights[[
    'Symbol', 'Company Name', 'Optimal Weights']]
optimal_weights = optimal_weights.sort_values(
    by=['Optimal Weights'], ascending=False)
print(optimal_weights)
