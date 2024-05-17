import pandas as pd

hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                  header=0, index_col=0, parse_dates=True)
hfi = hfi / 100
hfi.index = hfi.index.to_period('M')


#########  Calculate Semideviation
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


semideviation(hfi)
