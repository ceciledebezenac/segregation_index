#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:52:36 2019

@author: cdebezenac
"""

import numpy as np
from shapely.geometry import Point,MultiPoint,Polygon
from shapely.ops import cascaded_union
import math
import pandas as pd
import geopandas as gdp
from distortion.divergence.basic_analysis import Demographics




def binarize(matrix,threshold):
    '''Transform numeric matrix in binary form: value > 1
    Parameter
    ---------
    - matrix: numeric numpy array 
        matrix of numeric value to binarize based on the rule value < threshold
    - threshold: numeric (float, int)
         Threshold for True or False values in matrix. 
        
    Returns
    -------
    binarized matrix with value 0 if matrix[i,j]<threshold, else 1. 
    '''
    return((matrix>=threshold)*1)
    
    
def calculate_distances(coordinates):
    '''Computes the distance matrix representing the crow distance between centroids of polygons.
    Parameters
    ----------
    coordinates: numpy array, GeoSeries of Point type
        list of centroid Point coordinates in neighbouring spatial units. 
        
    Returns
    -------
    distance matrix : sqaure numpy array of floats
        matrix of euclidean distances between centroids of Polygons. 
    ''' 
    #initialise an 'empty' matrix
    distance_matrix=np.zeros((len(coordinates),len(coordinates)))
    for i in range(len(coordinates)):
        for j in range (i,len(coordinates)):
            #calculate the crow distance between centroids of i and j 
            distance_matrix[i,j] = coordinates[i].distance(coordinates[j])
            #crow distance being a metric, the matrix is symmetric
            distance_matrix[j,i] =  distance_matrix[i,j]
    return(distance_matrix)
            

def calculate_demographic_dissimilarity(population_matrix):
    '''Computes the dissimilarity matrix representing the difference in population count for each group in each unit.
    Parameters
    ----------
    population_matrix: numpy array of integers
        population count for categorial groups in each spatial unit.
    
    Returns
    -------
    demographic type dissimilarity matrix: sqaure numpy array
        matrix of euclidean distance between population proportions in any two units included in the input matrix. 
    '''
    #calculate the total population count in each unit
    sum_pop_local=np.sum(population_matrix,axis=1)
    #calculate each group's local proportion in each unit. 
    local_proportion=population_matrix/sum_pop_local[:,np.newaxis]
    #initialise an 'empty' square matrix
    demo_matrix=np.zeros((len(population_matrix),len(population_matrix)))
    
    for i in range(len(population_matrix)):
        for j in range (i,len(population_matrix)):
            #for each couple of units (i,j), compute the euclidean distance of population count
            demo_matrix[i,j] = np.sqrt(np.sum((local_proportion[i]-local_proportion[j])**2))
            #symmetric metric
            demo_matrix[j,i] =  demo_matrix[i,j]      
    return(demo_matrix)

def calculate_divergence_dissimilarity(population_matrix):
    '''Computes the KL divergence matrix representing the KL divergence between any two units in the city. 
    Parameters
    ----------
    population_matrix: numpy array of integers
        population count for categorial groups in each spatial unit.
    
    Returns
    -------
    divergence type dissimilarity matrix: sqaure numpy array
        matrix of KL divergence between any two units included in the input matrix, based on population distribution. 
    '''
    #check for null values of population and make them very small instead, to avoid KL divergence from going to infinity:
    pop_matrix=np.where(population_matrix==0, 0.0001, population_matrix)
    #calculate the total population count in each unit
    sum_pop_local=np.sum(pop_matrix,axis=1)
    #calculate each group's local proportion in each unit
    local_proportion=pop_matrix/sum_pop_local[:,np.newaxis]
    #initialise square matrix:
    divergence_matrix=np.zeros((len(pop_matrix),len(pop_matrix)))
    
    for unit1 in range(len(pop_matrix)):
        for unit2 in range (len(pop_matrix)):
            #go through all couples twice because KL divergence is NOT symmetric
            #calculate the ratio of local proportion on global proportions for each group in each unit. 
            relative_prop=local_proportion[unit1]/local_proportion[unit2]
            #calculate the log of the ratio
            log_relative = np.log(relative_prop)
            #check for exceptions and turn 0log0 values to 0
            log_relative[log_relative == -math.inf] = 0
            #fill in the divergence matrix with value
            divergence_matrix[unit1,unit2]=np.sum(np.multiply(local_proportion[unit1],log_relative))   
    return(divergence_matrix)


def is_adjacent(coordinates):
    '''Computes the adjacent matrix for the spatial units in the city. 
    Parameters
    ----------
    coordinates: GeoSeries of Polygons
        coordinates of spatial unit outile.
    
    Returns
    -------
    adjacent matrix: binary matrix of 0 or 1 if units touch in the "rook" sense, sharing a border. 
    '''
    adjacent=np.zeros((len(coordinates),len(coordinates)))
    for i in range(len(coordinates)):
        #use is_touching() to check if units are adjacent 
        is_touching=coordinates.touches(coordinates[i])
        #symmetrix relation:
        adjacent[i,:] = is_touching
        #add relation to itself (neighbours to itself)
        adjacent[i,i]=1
    return(adjacent)
        


class Neighbourhood():
    ''' 
    Builds the neighbourhood structure of spatial data unit based on the type of neighbourhood selected and the data attributes. 
    This neighbourhood represents more specifically all the spatial units in the city ordered according to a similarity rule, 
    from closest to farthest of a given unit. Without specifying a particular definition for neighborhood, 
    this structure will include the entire city at all possible scales. The algorithm is repeated for all possible origins. 
    
    Parameters
    ==========
        - geodata: Geopandas
                   Dataframe to analyse with geometry columns.
        - groups: list of string
                  list of names of categorical variables to analyse (ethnic groups for instance).
        - method: string
                  name of method to calculate dissimilarity between units. 
            
    
    
    Attributes
    ==========
        - coordinates: GeoSeries Polygon shape
                       coordinates of spatial units in data.
        - population_matrix: numpy array of integers ; shape (len(data),len(groups))
                             population count for all groups in all units. 
        - adjacency_matrix: square numpy array
                            binary matrix of touching neighbours (rook neighbourhood).
        - distance_matrix: square numpy array
                           symmetric matrix representing crow distances between all units. 
        - dissimilarity_matrix: square numpy array
                             non-symmetric or symmetric matrix representing a dimension of dissimilarity, 
                             bilateral KL divergence between units or demographic difference in population counts.       
        - neighbours: square numpy array of Index class
                      indexes of ordered neighbours for each unit.


    Methods
    =======
        - set_euclidean_neighbours()
            Returns: - distance_matrix: numpy array floats.
                       euclidean distances between units
                     - neighbours: numpy array of indices 
                       ordered list of units from smallest to largest crow distance for each unit

        - set_neighbourhood_type()
             
        
    Example
    =======
    
    
    
        
    '''
    
    def __init__(self,geodata,groups):
        #access coordinates of all Polygons in city
        self.coordinates=geodata.geometry
        #create a Demographic object and access the population_matrix (does not keep in memory the rest of the data)
        #(This step seems redundant with DivergenceProfile.demography and could be skipped if the city Demographic attribute was accessed directly, but this enables users to create neighbourhood structures independantly of the segregation analysis). 
        self.population_matrix=Demographics(geodata,groups).population_matrix
        #calculate the adjacecny matrix for coty coordinates. 
        self.adjacency_matrix=is_adjacent(self.coordinates)
        
    def set_euclidean_neighbours(self):
        '''
        Call this method to initialise a neighbourhood structure based on the crow distance between unit centroids.
       
        Returns
        -------
        - distance_matrix: numpy array floats
            euclidean distances between units
        - neighbours: numpy array of indices 
            ordered list of units from smallest to largest crow distance for each unit
            
        '''
        self.distance_matrix=calculate_distances(self.coordinates.centroid)
        self.neighbours=np.argsort(self.distance_matrix)
        
    def set_neighbourhood_type(self,method='demographic'):
        '''
        Call this method to create a hybrid neighbourhood structure based on adjacency and similarity relating to a particular variable specified in *method*.
        Parameters
        ----------
        method : string
            specific type of neighbourhood structure. Multiple choice in ['demographic','divergence']
                * 'demographic': the closest neighbour touches the origin and has the most similar population proportions (and so on).
                * 'divergence: the closest neighbour touches the origin and has the smallest bilateral KL divergence with the origin. 
            The *divergence* method is a non-linear transformation of the *demographic* method. 
        Returns
        -------
        - dissimilarity_matrix: numpy array of floats ; shape(len(coordinates)**2)
            dissimilarity between units given a distance or divergence method. 
        - neighbours : numpy array of indices ; shape(len(coordinates)**2)
            neighbourhood based on dissimilarity (divergence or demographic distance) and adjacency.'''
        
        if method=='demographic':
            #call the particular Neighbourhood method relating the the parameter method to compute the square dissimilarity matrix
            self.dissimilarity_matrix=calculate_demographic_dissimilarity(self.population_matrix)
        elif method=='divergence':
            self.dissimilarity_matrix=calculate_divergence_dissimilarity(self.population_matrix)
        else :
            raise ValueError('the '+method+' is unknown. Please choose between "euclidean", "divergence" or "demographic".')
        
        #access the maximum dissimilarity between any two units
        max_diff=self.dissimilarity_matrix.max()
        #initialise the list of matrices of incresaing adjacency degree : first, second, thrid, etc... degree neighbours. 
        adjacent_structure=[self.adjacency_matrix]
        #calculate the first degree dissimilarity matrix (dissimilarity between first degree neighbours)
        adjacent_difference=self.adjacency_matrix * self.dissimilarity_matrix
        #initialise degree of neighbourhood k
        k=1
        #initialise the adjacency matrix of degree k
        next_adjacent=1
        #include a stopping condition for the algorithm : that there are no more neighbours left to visit. 
        while np.sum(next_adjacent)>0:
            #multiply the k-1 degree matrix by the original adjacency matrix: this will give the k degree of neighbourhood: if b is a's neighbour and c is b's, then the next degree neighbour of a will include c. 
            k_neighbourhood=adjacent_structure[-1].dot(self.adjacency_matrix)
            #get the adjacent neighbors of degree k by excluding all existing ones already : use binarize() to set to 0 and 1 and subtract all the previous adjacency matrices from inferior neighbourhoo degree.
            next_adjacent=binarize(binarize(k_neighbourhood,1) - sum(adjacent_structure),1)
            adjacent_structure.append(next_adjacent)
            #calculate the dissimilarity matrix of degree k and add the maximum of dissimilarity multiplied by the neighbourhood degree, max_diff, to differentiate between degrees of neighbourhood once all dissimilarity matrices joint into one matrix.  
            adjacent_difference += next_adjacent * (self.dissimilarity_matrix + (max_diff+1) * k)
            k+=1  
        #sort by increasing order the rows of the matrix, this will order them first by degree of neighbourhood, then by smallest dissimilarity value.
        #this method avoids creating a dataframe for each unit with ordered degree and ordered dissimilarity but deals with all neighbours for all units in one single numpy array, which is quicker. 
        self.neighbours=np.argsort(adjacent_difference)
        
        
        
        
        
        
        
        