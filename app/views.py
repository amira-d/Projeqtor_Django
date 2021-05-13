# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.template import loader
from django.http import HttpResponse
from django import template

@login_required(login_url="/login/")
def index(request):
    
    context = {}
    context['segment'] = 'index'

    html_template = loader.get_template( 'index.html' )
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
def dash(request, **kwargs):
    'Example view that inserts content into the dash context passed to the dash application'

    context = {}

    # create some context to send over to Dash:
    dash_context = request.session.get("django_plotly_dash", dict())
    dash_context['django_to_dash_context'] = "I am Dash receiving context from Django"
    request.session['django_plotly_dash'] = dash_context

    return render(request, template_name='dashplot.html', context=context)