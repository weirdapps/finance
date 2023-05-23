import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Set the file path for the CSV
csv_path = 'marketcap.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_path)

# Select the symbol column and convert it to a list
symbol_list = df['Symbol'].tolist()[:20]

# Download daily adjusted close prices for symbol_list from Yahoo Finance
data = yf.download(symbol_list, start="2000-01-01",
                   end="2023-05-20")["Adj Close"]

# Calculate the daily returns
daily_returns = data.pct_change()

# Optimize the portfolio by maximizing the Sharpe ratio


def get_ret_vol_sr(weights, returns):
    weights = np.array(weights)
    ret = np.sum(returns.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sr = ret / vol
    return ret, vol, sr


def neg_sharpe(weights, returns):
    return -get_ret_vol_sr(weights, returns)[2]


def check_sum(weights):
    return np.sum(weights) - 1


def minimize_volatility(weights, returns):
    return get_ret_vol_sr(weights, returns)[1]


# Set the optimization constraints and bounds
cons = ({'type': 'eq', 'fun': check_sum})
bounds = ((0, 1),) * len(symbol_list)
init_guess = [1 / len(symbol_list)] * len(symbol_list)

# Optimize the portfolio for the maximum Sharpe ratio
opt_results = minimize(neg_sharpe, init_guess, args=(
    daily_returns,), method='SLSQP', bounds=bounds, constraints=cons)

# Calculate the efficient frontier
frontier_y = np.linspace(0, 0.3, 100)
frontier_volatility = []

for possible_return in frontier_y:
    cons = ({'type': 'eq', 'fun': check_sum},
            {'type': 'eq', 'fun': lambda w: get_ret_vol_sr(w, daily_returns)[0] - possible_return})
    result = minimize(minimize_volatility, init_guess, args=(
        daily_returns,), method='SLSQP', bounds=bounds, constraints=cons)
    frontier_volatility.append(result['fun'])

# Plot the efficient frontier
plt.figure(figsize=(12, 8))
plt.scatter(frontier_volatility, frontier_y, c=frontier_y /
            frontier_volatility, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Efficient Frontier')
plt.show()

# Get the optimal weights and sort them
optimal_weights = opt_results.x * 100
optimal_weights = pd.DataFrame(optimal_weights, columns=[
                               'Optimal Weights'], index=symbol_list[:len(optimal_weights)])
optimal_weights['Optimal Weights'] = optimal_weights['Optimal Weights'].map(
    '{:.2f}%'.format)
optimal_weights.sort_values(
    by='Optimal Weights', ascending=False, inplace=True)

# Print the table of optimal weights
print(optimal_weights)

# Print the expected annual return, annual volatility, and Sharpe ratio
ret, vol, sr = get_ret_vol_sr(opt_results.x, daily_returns)
print('Expected Annual Return:', round(ret * 100, 2), '%')
print('Annual Volatility:', round(vol * 100, 2), '%')
print('Sharpe Ratio:', round(sr, 2))
