"""
Financial market methods
=========================
"""

import numpy as np
import yfinance as yf




def candle(fig,data,legend=None):
    """
    Convenient function to plot candle sticks.

    :param fig: figure object created with plotly (cf. example below)
    :param data: stock time series imported with yfinance (Pandas)
    :param legend: plot title

    Example: Candle stick plot for Microsoft stocks

    >>> import plotly.graph_objs as go
    >>> import yfinance
    >>> fig=go.Figure()
    >>> msft=yfinance.download(tickers='MSFT', period='1y', interval='1d')
    >>> candle(fig,msft)
    >>> fig.show()
    """
    import plotly.graph_objs as go

    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                             low=data['Low'], close=data['Close'], name=legend))

def normalize(x1,x2):
    """
    Given two tickers, this method normalizes them such that you can 
    plot them together. One possible usage of this method is to compare
    the stock price of the same company in different exchanges—e.g. NASDAQ
    and B3—and see how they compare.

    :param x1: yfinance stock time series #1
    :param x2: yfinance stock time series #2
    :returns: x2 stock data normalized to the same scale as x1

    Example:

    >>> adbe=yf.download(tickers='ADBE', period='3mo', interval='1d')
    >>> adbeBR=yf.download(tickers='ADBE34.SA', period='3mo', interval='1d')
    >>> adbe['Close'].plot(label='US')
    >>> x=normalize(adbe,adbeBR)
    >>> x.plot(label='BR')
    >>> legend()
    >>> title('Adobe')
    >>> grid()
    """
    return x2['Close']/x2['Close'][0]*x1['Close'][0]

def returns(ticker,dt='ytd',t0=None):
    """
    Convenient method for retrieving the returns of a stock over a given
    time period.

    :param ticker: the stock ticker
    :param dt: the period covered ending at "now". Possible options: 11d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max. OPTIONAL, default is year to date.
    :param t0: the initial date in the format '2021-03-17'. If t0 is specified, then dt should not be.
    :returns: the percentage return of the stock in the specified period

    # Examples: 

    Returns from Small Cap BR ETF since July 11th 2014:

    >>> returns('SMAL11.SA',t0='2014-07-11')

    Returns from VALE3 in the last two years:

    >>> returns('VALE3.SA','2y')
    """
    from datetime import date

    if t0 is None:
        data=yf.download(tickers=ticker, period=dt, interval='1d', progress=False)
    else:
        today = date.today()
        data=yf.download(tickers=ticker, start=t0, end=today.strftime("%Y-%m-%d"), progress=False)

    r=(data['Close'][-1]/data['Close'][0]-1)*100
    
    return round(r,1)



def returnsTS(x):
	"""
    Given a stock, this method returns the stock percentage returns as a time series.

    :param x: yfinance stock time series
    :returns: stock time series of percentage returns

	"""
	return (x['Close']/x['Close'][0]-1)*100 

