# Max DrawDowns:
# In trading, maximum Drawdown (MDD) is the maximum loss observed from a portfolio's peak to trough before a new peak is
# reached. # It's calculated as the difference between the value of the lowest trough and the highest peak before the
# trough. # MDD is expressed as a percentage value and is calculated over a long time period when the value of an asset
# or # investment has gone through several boom-bust cycles.

import pandas as pd

me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                   header=0, index_col=0, parse_dates=True, na_values=-99.99)
rets = me_m[['Lo 10', 'Hi 10']]
rets.columns = ['SmallCap', 'LargeCap']
rets = rets / 100
rets.plot.line()

# Change the format of the index to actual date and tagging each of the returns as a monthly return.
rets.index = pd.to_datetime(rets.index, format="%Y%m")
rets.index = rets.index.to_period('M')
rets.head()

# Steps: 1.	Compute the wealth index:
wealth_index = 1000 * (1 + rets["LargeCap"]).cumprod()
wealth_index.plot.line()

# Steps: 2.	Compute previous peaks:
previous_peaks = wealth_index.cummax()
previous_peaks.plot()

# Steps: 3.	Compute the Drawdown - which is the wealth value as a percentage of the previous peaks:

drawdown = (wealth_index - previous_peaks) / previous_peaks
drawdown.plot()

drawdown.min()  # Find the min loss
drawdown["1975":].min()  # Find the min loss from 1975 to today
drawdown["1975":].idxmin()  # Find the index of min loss from 1975 to today
drawdown.idxmin()  # Find the index of min loss for all the time
