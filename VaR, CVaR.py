import pandas as pd

hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                  header=0, index_col=0, parse_dates=True)
hfi = hfi / 100
hfi.index = hfi.index.to_period('M')


#########  Calculate Historic VaR
def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


#########  Calculate Historic Conditional VaR
def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


#########  Calculate Gaussian VaR (Parametric VaR – Gaussian and Modified Cornish- Fisher VaR)

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


from scipy.stats import norm


def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level / 100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
             (z ** 2 - 1) * s / 6 +
             (z ** 3 - 3 * z) * (k - 3) / 24 -
             (2 * z ** 3 - 5 * z) * (s ** 2) / 36
             )
    return -(r.mean() + z * r.std(ddof=0))
