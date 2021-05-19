# # # -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

import django
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.template import loader
from django.http import HttpResponse , request
from django import template
from json import dumps
from .custom_dash import cost_plot
from django.forms import *
from .forms import *
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from  scipy.stats import chi2 
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import sqlite3
from sqlite3 import OperationalError
import psycopg2
from pandas import DataFrame
import pandas_profiling as pp
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import pandas_profiling as pp
from sklearn.tree import DecisionTreeClassifier
import plotly
import plotly.graph_objs

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler # i will use Min Max Scaler 
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import OrdinalEncoder
import plotly.graph_objects as go

import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import plotly.offline as opy
import plotly.graph_objs as go



def df():
    pd.set_option('display.max_columns', None)
    missing_values = ["n/a", "na", "--", " ",""] 
    dfc= pd.read_csv("app/data.csv", na_values=missing_values , index_col=None)
    df1 = dfc[['idProject','Category','Premier Delay','projectDuration','CLient','Date','paymentDone','nb_phases']]
    for i in range(len(df1)-1):
        if(df1['idProject'][i]==df1['idProject'][i+1] and df1['nb_phases'][i] < df1['nb_phases'][i+1]  ):
            df1.drop(labels=[i],axis=0,inplace=True)
    df1.groupby('idProject')
    df1.head()
    df=df1.drop(df1[['idProject'  ,'Date','paymentDone' ]],axis=1)
    dfA=df1.drop(df1[['idProject','Date','paymentDone']],axis=1)
    df['CLient'] = df['CLient'].astype('category')
    df['Category'] = df['Category'].astype('category')
    number = LabelEncoder()
    df['Category'] = number.fit_transform(df1['Category'].astype('str'))
    #df['Premier Delay'] = number.fit_transform(df1['Premier Delay'].astype('str'))
    df['CLient'] = number.fit_transform(df1['CLient'].astype('str'))
    df['Date'] = number.fit_transform(df1['Date'].astype('str'))


    print(df.head())
    return df

def dfA():
    pd.set_option('display.max_columns', None)
    missing_values = ["n/a", "na", "--", " ",""] 
    dfc= pd.read_csv("app/data.csv", na_values=missing_values , index_col=None)
    df1 = dfc[['idProject','Category','Premier Delay','projectDuration','CLient','Date','paymentDone','nb_phases']]
    for i in range(len(df1)-1):
        if(df1['idProject'][i]==df1['idProject'][i+1] and df1['nb_phases'][i] < df1['nb_phases'][i+1]  ):
            df1.drop(labels=[i],axis=0,inplace=True)
    df1.groupby('idProject')
    df1.head()
    df=df1.drop(df1[['idProject' ,'Date','paymentDone'  ]],axis=1)
    dfA=df1.drop(df1[['idProject'],'Date','paymentDone'],axis=1)
    
    dfA['CLient'] = dfA['CLient'].astype('category')
    dfA['Category'] = dfA['Category'].astype('category')
    return dfA  

def df1():
    pd.set_option('display.max_columns', None)
    missing_values = ["n/a", "na", "--", " ",""] 
    dfc= pd.read_csv("app/data.csv", na_values=missing_values , index_col=None)
    df1 = dfc[['idProject','Category','Premier Delay','projectDuration','CLient','Date','paymentDone','nb_phases']]
    for i in range(len(df1)-1):
        if(df1['idProject'][i]==df1['idProject'][i+1] and df1['nb_phases'][i] < df1['nb_phases'][i+1]  ):
            df1.drop(labels=[i],axis=0,inplace=True)
    df1.groupby('idProject')
    df1.head()
    
    return df1

def knn():
        
    y=df()['Premier Delay']
    x=df().drop(['Premier Delay'], axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=23,shuffle=True)
    #robust = RobustScaler()
    #x_train=robust.fit_transform(x_train)
    #x_test=robust.transform(x_test)
    KNN_model = KNeighborsClassifier(n_neighbors=13, metric='minkowski',algorithm='auto',leaf_size=1,p=1,weights='uniform')
    KNN_model.fit(x_train, y_train)
    print(KNN_model.score(x_test, y_test))
    print(KNN_model.score(x_train, y_train))
    return KNN_model

def decision_tree():
    y=df()['Premier Delay']
    x=df().drop(['Premier Delay'], axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=23,shuffle=True)
    

    Tree_model = DecisionTreeClassifier(splitter= 'random',
                            max_depth=8,
                            criterion='gini',
                            random_state=5,shuffle=True)
    Tree_model.fit(x_train, y_train)
    print(Tree_model.score(x_test, y_test))
    print(Tree_model.score(x_train, y_train))
  
    clf = DecisionTreeClassifier()
    cv_score = cross_val_score(clf, x_train, y_train,scoring = 'accuracy',
                            cv = 12,
                            n_jobs = -1,
                            verbose = 5)
    cv_score
    clf.fit(x_train, y_train)
    return clf

def logreg():
    y=df()['Premier Delay']
    x=df().drop(['Premier Delay','Date'], axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=23,shuffle=True)
    

    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    return logreg

def naive_bai():
    y=df()['Premier Delay']
    x=df().drop(['Premier Delay'], axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=23,shuffle=True)
    
    #model
    modele = BernoulliNB()
    modele.fit(x_train, y_train)
   


def compareModels():

    # load dataset
    dataframe = df()
    array = dataframe.values
    Y=df1()['Premier Delay']
    X=df1().drop(['Premier Delay','Category','CLient','Date'], axis=1)
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1,shuffle=True)
    

    # prepare configuration for cross validation test harness
    seed = 7
    # prepare models
    models = [KNeighborsClassifier(),DecisionTreeClassifier(),LogisticRegression(),BernoulliNB()]

    # evaluate each model in turn
    MLA_columns = []
    MLA_compare = pd.DataFrame(columns = MLA_columns)


    row_index = 0
    for alg in models:
        
        predicted = alg.fit(x_train, y_train).predict(x_test)
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index,'MLA Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(alg.score(x_train, y_train), 4)
        MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round(alg.score(x_test, y_test), 4)
        row_index+=1
    
    MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)    
    MLA_compare
 
  
   
    trace = go.Figure(
                    data=[
                        go.Bar(
                            name="Original",
                            x=MLA_compare['MLA Name'],
                            y=MLA_compare['MLA Test Accuracy'],       
                            offsetgroup=0,
                        ),
                    ],
                    layout=go.Layout(
                        title="Data Prediction Models comparison",
                        yaxis_title="MLA Test Accuracy"
                    )
                )

    bar_div = opy.plot(trace, auto_open=False, output_type='div')
    context ={}
    context['graph'] = bar_div
    return context
    

@login_required(login_url="/login/")
def index(request):
    
    context = {}
    context['segment'] = 'index'

    html_template = loader.get_template( 'index.html' )
    return HttpResponse(html_template.render(context, request))

@login_required(login_url="/login/")
def pdash(request):
    
    context = {}
    context['segment'] = 'pdash'
    html_template = loader.get_template( 'pdash.html' )
    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def charts(request):
    
    context = {}
    context['segment'] = 'charts'

    html_template = loader.get_template( 'pcharts.html' )
    return HttpResponse(html_template.render(context, request))
@login_required(login_url="/login/")
def pcharts(request):
    
    context = {}
    context['segment'] = 'pycharts'

    html_template = loader.get_template( 'pycharts.html' )
    return HttpResponse(html_template.render(context, request))
@login_required(login_url="/login/")
def pythondoc(request):
    
    context = {}
    context['segment'] = 'pythondoc'

    html_template = loader.get_template( 'pythondoc.html' )
    return HttpResponse(html_template.render(context, request))
    


@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:
        
        load_template      = request.path.split('/')[-1]
        context['segment'] = load_template
        
        html_template = loader.get_template( load_template )
        return HttpResponse(html_template.render(context, request))
        
    except template.TemplateDoesNotExist:

        html_template = loader.get_template( 'page-404.html' )
        return HttpResponse(html_template.render(context, request))

    except:
    
        html_template = loader.get_template( 'page-500.html' )
        return HttpResponse(html_template.render(context, request))

@login_required(login_url="/login/")
def dash(request):
    'Example view that inserts content into the dash context passed to the dash application'
    context = {}
    return render(request, template_name='dashplot.html', context=context)




@login_required(login_url="/login/")
def formP(request):
    context = compareModels()
    context['segment'] = 'formP'
    html_template = loader.get_template( 'formP.html' )
    return HttpResponse(html_template.render(context, request))

@login_required(login_url="/login/")
def predict(request):

    if request.method == "POST": #and form1.validate_on_submit():
        temp={}
        from random import randrange
        temp['Category']=randrange(3)
        temp['projectDuration']=request.POST.get('duration')
        temp['CLient']=randrange(10)
        temp['nb_phases']=request.POST.get('nb_phases')

        testData = pd.DataFrame({'x':temp}).transpose()
      
        print(testData['Category'])
        #testData['duration'] = testData['duration'].astype('int64')
        #testData['nb_phases'] = testData['nb_phases'].astype('int64')
        print(testData)
        pred = logreg().predict(testData)
        context ={'graph':compareModels(),'pred':pred,'message':pred}
        context['segment'] = 'formP'
        html_template = loader.get_template( 'formP.html' )
        return HttpResponse(html_template.render(context, request))
    context ={'pred':None}
    context['segment'] = 'formP'
    html_template = loader.get_template( 'formP.html' )
    return HttpResponse(html_template.render(context, request))

#def DashPlots():

@login_required(login_url="/login/")
def DashPython(request):
    context = {}
    context['segment'] = 'charts'
    html_template = loader.get_template( 'charts.html' )
    return HttpResponse(html_template.render(context, request))
