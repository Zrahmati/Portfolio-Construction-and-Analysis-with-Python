import pandas as pd
import scipy.stats

hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                  header=0, index_col=0, parse_dates=True)
hfi = hfi / 100
hfi.index = hfi.index.to_period('M')


#########  Calculate Skewness
def skewness(r):
    """
   Alternative to scipy.stats.skew()
    Computes the skewness of the supplies Series of DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # Use the population standard deviation, so set dof=0 (Degree of Freedom)
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 3).mean()  # exp is the expected value
    return exp / sigma_r ** 3


skewness(hfi).sort_values()

# Another formula to calculate skewness:
import scipy.stats

scipy.stats.skew(hfi)


######### Calculate Kutosis

def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplies Series of DataFrame
    Returns a flaot or a Series
    """
    demeaned_r = r - r.mean()  # Use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 4).mean()
    return exp / sigma_r ** 4


# Another formula to calculate Kurtosis:
scipy.stats.kurtosis(hfi)  # this one calculates the EXCESS kurtosis which is subtracted by 3.

######### Jarque_Bera test:

scipy.stats.jarque_bera(hfi)


# This approach tests all the data at once, which becomes problematic when applied to multiple hedge funds or stocks.
# To separate the data for each fund and perform the Jarque-Bera test individually, use the following code:


def is_normal(r, level=0.01):
    """
    Applies the Jarque_bera test to determine if a Series is normal or not
    Test is applied at 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise.
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level


is_normal(hfi)
hfi.aggregate(is_normal)  # This code will separate the funds and gives result for each of them.
