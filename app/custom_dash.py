import itertools
import numpy as np
import pandas as pd
from pandas.plotting import lag_plot
#import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit

import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from dateutil.relativedelta import relativedelta # working with dates with style
from scipy.optimize import minimize              # for function minimization
import scipy.stats as scs

from itertools import product                    # some useful functions
#from tqdm import tqdm_notebook
import plotly
import plotly.offline as pyo
import plotly.graph_objs as go
import json
from sklearn.metrics import mean_absolute_error
#import matplotlib
#import matplotlib.pyplot as plt

#import warnings

# warnings.filterwarnings("ignore")
#warnings.simplefilter('ignore')
def cost_plot(brand):
    
    pd.set_option('display.max_columns', None)
    missing_values = ["n/a", "na", "--", " ",""] 
    dfc= pd.read_csv("data.csv", na_values=missing_values , index_col=None)
    dfc['Date'] = pd.to_datetime(dfc['Date'])
    print(dfc.head())
    dates = pd.date_range(start='2020-01-01', end='2021-05-31', freq='1d')
    df1 = dfc[['idProject','Category','Premier Delay','projectDuration','CLient','Date','paymentDone','nb_phases']]
    for i in range(len(df1)-1):
        if(df1['idProject'][i]==df1['idProject'][i+1] and df1['nb_phases'][i] < df1['nb_phases'][i+1]  ):
            df1.drop(labels=[i],axis=0,inplace=True)
    df1.groupby('idProject')
    df1.head()
    #df1['idProject'].value_counts()


  

    t1=go.Scatter(x=tsfinal[brand].index, y=tsfinal[brand].bottles)
    t2=go.Scatter(x=ts[brand].index, y=ts[brand].bottles)
    tr=[t1,t2]
    graph_json=json.dumps(tr,cls=plotly.utils.PlotlyJSONEncoder)
    return (graph_json)
#fig = go.Figure(data=tr)
    
    
#    figs.append(fig)
#i = 0
#for k in tsfinal.keys():
#    pyo.plot(figs[i], filename=r'C:\Users\esprit\Documents\Plots1'+k, auto_open=False)
#    i=i+1