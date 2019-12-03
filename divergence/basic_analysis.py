#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 19:49:00 2019

@author: cdebezenac
"""

import numpy as np
import math
import pandas as pd
import geopandas as gdp
import matplotlib.pyplot as plt
from scipy.integrate import quad
import statsmodels.api as sm
import scipy.stats
import sys

class Demographics():
    '''Compute the general population statistics.
    Attributes
    ==========
        - population_matrix: numpy array of integers
                             ordered population per group per geographic unit.
        - sum_pop_local: numpy array of integers
                         total population in each unit.
        - sum_pop_group: numpy array of integers
                         total population in each group
        - total_pop: int
                     total population in total area.
        _ global_statistics: numpy array of float numbers
                             population share for each group.
        - relative_density: numpy array of float numbers
                            population share in each geographic unit.
                            
    Example
    =======
    
    
        '''
    def transform_population(geodata,groups):
        ''' Tranform population count for *groups* from dataframe *geodata* into numpy array.'''
        pop_matrix = np.zeros((len(geodata),len(groups)))
        for j in range (len(groups)):
            pop_matrix[:,j] = list(map(float,geodata[groups[j]]))
        return(pop_matrix)

    def __init__(self,geodata,groups):
        #transform population count in geodata
        self.population_matrix=Demographics.transform_population(geodata,groups)
        #total population in unit
        self.sum_pop_local = np.sum(self.population_matrix, axis=1)
        #total population of each group
        self.sum_pop_group = np.sum(self.population_matrix, axis=0)
        #total population
        self.total_pop=np.sum(self.population_matrix)
        #global proportions of each group
        self.global_statistics=self.sum_pop_group/self.total_pop
        #weight of each unit in the overall city
        self.relative_density=self.sum_pop_local/self.total_pop
        
        
        
class LorenzCurve():
    '''Plots the Lorenz Curve for one population group distribution and returns the related Gini index. 
    This index informs on the relative spatial concentration of social groups on a given territory. 
    It has been used as an aspatial segregation index in literature throughout the years. 
    
    Attributes
    ==========
        - group: string
        name of population group in data.
        - lorenz_array: numpy array of float
        cumulated proportions of ordered group count in all units.
        - gini: float
        gini index associated with the lorenz_curve. 
                
    Example
    =======
    
    
        '''
    
    def lorenz_list(dataset,variable):
        '''
        Create the list of cumulated population on the ordered units from smallest proportions to largest.
        Parameters
        ----------
        - dataset: Pandas or Geopandas dataframe
            dataframe with quantitative variable for concentration measures.
        - variable : string
            quantitative variable for analysis.
                
        Retruns
        -------
        numpy array of shape (1,len(dataset)) of discretised cumulated proportions.
        '''
        
        #access data and order it
        sorted_list=list(dataset[variable].copy())
        sorted_list.sort()
        
        #calculate the cumulated proportions
        lorenz=np.cumsum(np.array(sorted_list))
        total_population=lorenz[-1]
        lorenz=lorenz/total_population 
        return(lorenz)
    
    def __init__(self,data,group):
        #initialise class attributes
        self.group=group
        self.lorenz_array=LorenzCurve.lorenz_list(data,group)
        #calculate gini index as the area between the lorenz curve and the first bisector. 
        self.gini=(0.5-np.sum(self.lorenz_array)/len(data))/0.5
        
        
    def plot(self,title='Lorenz Curve',figure_size=(12,10),legend='Lorenz curve',line=1, line_color="black",show_axis=False,save=True,path='lorenz_curve.png'):
        '''
        Paramaters : 
                - title : Plot title.
                - figure size. 
                - legend : the legend of the plot.
                - line  : the curve width.
                - line_color : the curve color.
                - show_axis : show the x and y axis and ticks.
                - save : save the plot.
                - path : location of image.
            Returns :
                plot of the Lorenz_curve and Gini index. 
        
        '''
        plt.figure(figsize=figure_size)    
        #initialise the figure based on parameters
        ax = plt.subplot(111)    
        ax.spines["top"].set_visible(False)    
        ax.spines["bottom"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.spines["left"].set_visible(False)    
        
        if show_axis==False:
            ax.set_xticks([], [])
            ax.set_yticks([], [])
        else:
            ax.get_xaxis().tick_bottom()    
            ax.get_yaxis().tick_left()
       
        #plot the curve
        plt.plot(list(self.lorenz_array), lw=line, color=line_color, alpha=0.3)    

        plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                        labelbottom="on", left="off", right="off", labelleft="on")

        plt.text(0, -0.18,legend, fontsize=12, family='serif')
        
        #save the figure if wanted
        if save==True :
            plt.savefig(path)
    
 