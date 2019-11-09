#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import itertools as it
import random as rd
import numpy as np
import copy as co
import matplotlib.pyplot as plt
from matplotlib import pyplot
import scipy as sci
import scipy.spatial.distance as dist
from shapely.geometry import Point,MultiPoint,Polygon
from shapely.ops import nearest_points
from math import *
import pandas as pd
from geopandas import *
import timeit
import networkx as nx
import statsmodels.api as sm
import pickle
from scipy.integrate import quad
import statsmodels.api as sm
import scipy.stats




# In[20]:


def pdf(dd_profile,X,w):
    '''
    Compute X-pdf : frequency for w value

    '''
    density=sm.nonparametric.KDEUnivariate(dd_profile[X])
    density.fit()
    return(density.evaluate(w))



# In[21]:


def pdf_ratio(dd_profile_1,dd_profile_2,X1,X2,w):
    '''
    ratio of pdf for same value w

    '''
    return(pdf(dd_profile_1,X1,w)/pdf(dd_profile_2,X2,w))



# In[22]:


def cdf(dd_profile,X,w):
    '''
    Compute X-cdf for value w

    '''
    PDF=sm.nonparametric.KDEUnivariate(dd_profile[X])
    PDF.fit()
    c=quad(PDF.evaluate,0,w)
    return(c)





def relative_cdf(dd_profile_1,dd_profile_2,X1,X2,r):
    '''
    X2-cdf of X1-cdf inverse for quantile r

    '''
    Quantile1=scipy.stats.mstats.mquantiles(dd_profile_1[X1],r)
    FQuant=cdf(dd_profile_2,X2,Quantile1)
    fQuant=pdf(dd_profile_2,X2,Quantile1)
    f0Quant=pdf(dd_profile_1,X1,Quantile1)


    return(FQuant,fQuant,f0Quant)


def distribution_plots(data_compare,X,s) :
    '''
    Density Plots for n distributions with a s-step array'''
    minS=[0 for l in range (len(data_compare))]
    maxS=[0 for l in range (len(data_compare))]
    for i in range(len(data_compare)):
        minS[i]=min(data_compare[i][X[i]])
        maxS[i]=max(data_compare[i][X[i]])
    min_support=min(minS)
    max_support=max(maxS)
    vect=np.arange(min_support-10*s,max_support+10*s,s)
    for i in range(len(data_compare)):
        plt.plot(vect,pdf(data_compare[i],X[i],vect))



# In[1]:
