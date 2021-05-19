# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.contrib.auth.models import User
from django.db import models

# Create your models here.


class Project(models.Model):
    Title = models.CharField(max_length=25)
    Client = models.CharField(max_length=25)
    Sector = models.CharField(max_length=25)
    ProjectDuration = models.FloatField()
    NbPhases =models.IntegerField()


    def _str_(self): #show the actual city name on the dashboard
        return self.Sector

    class Meta: #show the plural of city as cities instead of citys
        verbose_name_plural = 'Projects'
 
 