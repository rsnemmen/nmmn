"""
Financial market methods
=============================
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
    Given two tickers of the same company in different exchanges, this method normalizes them such
    that you can plot them together.
    """
    return x2['Close']/x2['Close'][0]*x1['Close'][0]

def returns(ticker,dt='ytd',t0=None):
    """
    Convenient method for retrieving the returns of a stock.
    
    t0 in the format '2021-03-17'
    """
    from datetime import date

    if t0 is None:
        data=yf.download(tickers=ticker, period=dt, interval='1d')
    else:
        today = date.today()
        data=yf.download(tickers=ticker, start=t0, end=today.strftime("%Y-%m-%d"))        
​
    r=(data['Close'][-1]/data['Close'][0]-1)*100
    
    return round(r,1)