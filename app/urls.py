# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path,include
from app import views
from app import dashboard1


urlpatterns = [

    # The home page
    path('', views.index, name='home'),
    path('charts/', views.charts, name='charts'),
    path('pcharts/', views.pcharts, name='pcharts'),
    path('pythondoc/', views.pythondoc, name='pythondoc'),
    path('django_plotly_dash/', include('django_plotly_dash.urls')),
    path('dash/', views.dash , name='dashs'),
    # Matches any html file
    #re_path(r'^.*\.*', views.pages, name='pages'),

]
