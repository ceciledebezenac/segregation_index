#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !/usr/bin/env python
# coding: utf-8
#make an import from requirements doc
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





# In[11]:
class Demographics():
    '''Compute the general population statistics
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
    Methods
    =======

        '''
    def transform_population(geodata,groups):
        pop_matrix = np.zeros((len(geodata),len(groups)))
        for j in range (len(groups)):
            pop_matrix[:,j] = list(map(float,geodata[groups[j]]))
        return(pop_matrix)

    def __init__(self,geodata,groups):
        self.population_matrix=self.transform_population(geodata,groups)
        self.sum_pop_local = np.sum(self.population_matrix, axis=1)
        self.sum_pop_group = np.sum(self.population_matrix, axis=0)
        self.total_pop=np.sum(self.population_matrix)
        self.global_statistics=self.sum_pop_group/self.total_pop
        self.relative_density=self.sum_pop_local/self.total_pop

class LocalDivergenceProfile():
    '''Compute the divergence profile for a given geographic unit.
    Attributes
    ==========
        - geodata: CityDivergence object
                   the data from the unit environment (see CityDivergence)
        - origin: dictionary of attributes
                  id=index
                  coordinates=geographic coordinates of origin centroid.
        - path: str or graph
                distance type used to find neighbours.
        - groups: list of str
                  names of groups as keys in dataframe.


    Methods
    =======
        - find_neighbours()
            Returns: - neighbours: numpy array of index type?
                       ordered list of units from nearest to farthest
                     - distance_neighbours: numpy array of float numbers
                       ordered list of distances from neighbbours.
                     - sorted_population: numpy array of integers
                       new population matrix sorted by distance from origin.

        - draw_profile()
            Returns: - profile: numpy array of floats
                       divergence by aggregation levels
        - calculate_indexes()
            Returns: - max_profile: numpy array of floats
                       superiour envelope of dievergence profile (max divergence from the end).
                     - min_profile: numpy array of floats
                       inferior envelope of dievergence profile (min divergence from the beginning).
                     - max_index: float
                       weighted sum of max_profile
                     - min_index: float
                       weighted sum of min_profile
                     - delta_index: float
                       difference of superiour and inferiour envelope.
                     - expected_divergence: float
                       expected divergence on the choice of the aggregation level (mean value on aggregation level).


        '''

    def __init__(self,ddprofile,origin,key,path='crows',groups):
        self.geodata=ddprofile
        self.origin={'id':self.geodata.dataframe.iloc(origin)[key],'coordinates':self.geodata.dataframe.iloc(origin).geometry.centroid}
        self.path=path
        self.groups=groups

    def find_neighbours(self):
        if self.path='crows':
            distance_to_others=[self.origin['coordinates'].distance(i) for i in self.geodata.coordinates]#from the dd_profile dataframe
        else :
            distance_to_others=[self.path[self.origin['id']][i] for i in self.geodata.coordinates]
        self.neighbours = np.argsort(distance_to_others)
        self.distance_neighbours=np.sort(distance_to_others)

        if self.groups='all':
            self.sorted_population=self.geodata.demography.population_matrix[self.neighbours,:]
        else :
            self.sorted_population=self.geodata.demography.marginal_population_matrix[self.neighbours,:]


    def draw_profile(self):
        #cumulative sum for each group
        cumul_pop=np.cumsum(self.sorted_population,axis=0)
        #cumulative sum regardless of group
        all_cumul_pop=np.cumsum(cumul_pop,axis=1)
        #Division termes à termes des sommes cumulées partielles et de la population totale cumulée
        cumul_proportion_group = cumul_pop / all_cumul_pop[:,np.newaxis]
        #Division termes à termes de la proportion suivante (matrice) par la proportion globale de chaque groupe
        relative_cumul_proportion = cumul_proportion_group / self.geodata.demogrphy.global_statistics
        #Le log de ce rapport de rapport
        log_relative_cumul = np.log(relative_cumul_proportion)
        #Traiter les 0log(0)
        log_relative_cumul[log_relative_cumul == -inf] = 0
        #Remultiplier par les proportions cumulées de chaque groupe et en faire la somme sur le nombre de groupes
        self.profile = np.sum(np.multiply(cumul_proportion_group,log_relative_cumul),axis = 1)

    def calculate_indexes(self):
        sorted_pop_local=np.sum(self.sorted_population,axis=1)
        max_distortion=0#max from end
        min_distortion=divergence_from_origin.profile[0]#max from beginning
        self.max_profile=np.zeros(len(self.dataframe))
        self.min_profile=np.zeros(len(self.dataframe))

        for level in range(1,len(self.dataframe)+1):
            max_distortion = max(self.profile[-level] , max_distortion)
            self.max_profile[-level] = max_distortion
            min_distortion = min(min_distortion , self.profile[level-1])
            self.min_profile[level-1]=min_distortion
        delta_min_max=list(map(float.__sub__,self.max_profile , self.min_profile))
        self.max_index=np.sum(np.multiply(self.max_profile , sorted_pop_local)/self.geodata.demography.total_pop
        self.max_index_normal=self.max_index/self.geodatat.sup_distortion #normal version
        self.min_index=np.sum(np.multiply(self.min_profile , sorted_pop_local)/self.geodata.demography.total_pop
        self.delta_index=np.sum(np.multiply(delta_min_max , sorted_pop_local)/self.geodata.demography.total_pop
        self.expected_divergence=np.sum(np.multiply(self.profile , sorted_pop_local)/self.geodata.demography.total_pop



class DivergenceProfiles :

    def theoretical_max_distortion(geodata, sumpergroup,sharepergroup):
        new_pop=np.zeros((len(geodata),len(sumpergroup)))
        ordersum=sorted(sumpergroup)
        ordershare=sorted(sharepergroup)
        for i in range(len(geodata)):
            C=sum(sumpergroup)//len(geodata)#capacité moyenne d'une unité
            for j in range (len(sumpergroup)):
                new_pop[i,j]=min(C,ordersum[j])
                C-=new_pop[i,j]
                ordersum[j]-=new_pop[i,j]
        cumul_pop = np.cumsum(new_pop, axis=0)
        #vecteur des sommes cumulée sur tous les groupes
        sum_cumul_pop = np.sum(cumul_pop, axis=1)
        #Division termes à termes des sommes cumulées partielles et de la population totale cumulée
        cumul_proportions = cumul_pop / sum_cumul_pop[:,np.newaxis]
        #Division termes à termes de la proportion suivante (matrice) par la proportion globale de chaque groupe
        relative_cumul_proportions = cumul_proportions / ordershare
        #Le log de ce rapport de rapport
        log_relative_cumul = np.log(relative_cumul_proportions)
        #Traiter les 0log(0)
        log_relative_cumul[log_relative_cumul == -inf] = 0
        #Remultiplier par les proportions cumulées de chaque groupe et en faire la somme sur le nombre de groupes
        div = np.sum(np.multiply(cumul_proportions,log_relative_cumul),axis = 1)
        theo_max_distortion = np.sum(myDiv)/len(geodata)
        return(theo_max_distortion)

    def __init__(self,geodata,groups,key):
        self.dataframe=geodata.sample(frac=1).reset_index(drop=True)#verifier si ca cree une copir qui est permute ou non
        self.key=key
        self.groups=groups
        self.coordinates=self.dataframe.geometry.centroid
        self.demography=Demograhics(self.dataframe,self.groups)
        self.sup_distortion=theoretical_max_distortion(self.dataframe,self.demography.sum_pop_group,
                                                        self.demography.global_statistics)


    def set_profiles(self,path='crows',marginal_group='None'):

        if marginal_group!='None':
            others='not_'+marginal_group
            new_groups=[marginal_group,others]
            other_population=self.groups.copy()
            other_population.remove(marginal_group)
            self.dataframe[others]=self.dataframe[other_population].sum(axis=1)
            self.marginal_demography=Demographics(self.dataframe,new_groups)

        self.divergence_data=[]
        for origin in range(len(self.dataframe)):
            divergence_from_origin = LocalDivergenceProfile(self,origin,self.key,path,marginal_group)
            divergence_from_origin.find_neighbours()
            divergence_from_origin.draw_profile()
            divergence_from_origin.calculate_indexes()
            self.divergence_data.append(divergence_from_origin)


    def update_data(self):
        self.dataframe['max_index']=pd.Series([self.divergence_data[i].max_index for i in range(len(self.dataframe))],
        index=[i for i in range(len(self.dataframe))])
        self.dataframe['min_index']=pd.Series([self.divergence_data[i].min_index for i in range(len(self.dataframe))],
        index=[i for i in range(len(self.dataframe))])
        self.dataframe['delta_index']=pd.Series([self.divergence_data[i].delta_index for i in range(len(self.dataframe))],
        index=[i for i in range(len(self.dataframe))])
        self.dataframe['expected_divergence']=pd.Series([self.divergence_data[i].expected_divergence for i in range(len(self.dataframe))],
        index=[i for i in range(len(self.dataframe))])

    def save_to_file(self,file_name):
        pass
