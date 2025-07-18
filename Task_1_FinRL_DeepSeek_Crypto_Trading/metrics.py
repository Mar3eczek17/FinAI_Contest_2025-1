import numpy as np
import pandas as pd

def cumulative_returns(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return a pd.Series
    """
    return (1 + returns_pct).cumprod() - 1

def sharpe_ratio(returns_pct, risk_free=0):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return float
    """
    returns = np.array(returns_pct)
    if returns.std() == 0:
        sharpe_ratio = np.inf
    else:
        sharpe_ratio = (returns.mean()-risk_free) / returns.std()
    return sharpe_ratio

def max_drawdown(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return: float
    """
    cum_returns = (1 + returns_pct).cumprod()
    peak = cum_returns.expanding(min_periods=1).max()
    drawdown = (cum_returns/peak) - 1
    return drawdown.min()

def return_over_max_drawdown(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return: float
    """
    mdd = abs(max_drawdown(returns_pct))
    returns = cumulative_returns(returns_pct).iloc[-1]
    if mdd == 0:
        return np.inf
    return returns/mdd
