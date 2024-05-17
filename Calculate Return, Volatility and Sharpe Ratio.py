import pandas as pd
import numpy as np

returns = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                      header=0, index_col=0, parse_dates=[0], na_values=-99.99)
columns = ["Lo 10", "Hi 10"]  # Pull out the required columns
returns = returns[columns]
returns = returns / 100  # get the raw numbers (without percentage)
returns.columns = ["SmallCap", "LargeCap"]  # rename the columns
returns.plot.line()  # plot the returns

n_months = returns.shape[0]()  # calculate return per month
return_per_month = ((returns + 1).prod() ** (1 / n_months)) - 1
return_per_month

annualized_return = ((returns + 1).prod() ** (12 / n_months)) - 1  # calculate annualized return

annualized_vol = returns.std() * np.sqrt(12)  # Calculate Annualized Volatility (Risk)
annualized_vol

annualized_return / annualized_vol

risk_free_rate = 0.03  # Calculate Sharpe Ratio
excess_return = annualized_return - risk_free_rate
sharpe_ratio = excess_return / annualized_vol
sharpe_ratio

