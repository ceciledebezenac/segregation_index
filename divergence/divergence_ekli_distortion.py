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




def _divergence(aGeodata, aOrigin, aCoordinates, aGEOID, aPopulationMatrix, aGlobalFigure, aSigma,  akey, aPath='crows'):
    """
    Calculation of  a Divergence trajectorie and Distorsion index, for a given origin

    Parameters
    ----------
    aGeodata          : a geopandas DataFrame with a geometry column.

    aOrigin       : integer index for point of origin

    aCoordinates          : a list of centroid coordinates of all points of dataframe.

    aGEOID          : an administrative index for path computation and vertex identification.

    aPopulationMatrix          : a numpy matrix with non-ordered population foat numbers.

    aGlobalFigure          : a float total population count .

    aSigma          : a float for kernel divergence computation.

    akey          : .

    aPath          : a Floyd Warshall graph with shortest path for each georeferenced point, identitfied by aGOEID.

    Returns
    ----------
    myNeighbors     : list of neighbors ordered by shortest distance

    myDiv           : array of divergence by agregation : trajectory from aOrigin

    myWeightedDiv   : array of divergence by kernel agregation : trajectory with gaussian kernel (aSigma variance)

    myWeightedSumPopPerLocation : an array of standardised population by gaussian weight for each location

    Notes
    -----
    Based Julien Randon-Furling, Madalina Olteau, Antoine Lucquiaud. "From Urban segregation to spatial structure detection ."


    """
    # Définir l'origine de la trajectoire
    myOrigin = aGeodata.iloc[aOrigin]
    myOriginPoint = myOrigin.geometry.centroid
    myOriginID=myOrigin[akey]

    if aPath=='crows':
        # Retourne la distance du point a tous les autres points:index naturel
        myDistancesToOtherPoints = [myOriginPoint.distance(i) for i in aCoordinates]

        # Retourne la liste des index des points tries selon la distance au point d origine
        myNeighbours = np.argsort(myDistancesToOtherPoints)
        myOrderedDistances=np.sort(myDistancesToOtherPoints)

    else :
        myDistancesToOtherPoints = [aPath[myOriginID][i] for i in aGEOID]
        myNeighbours = np.argsort(myDistancesToOtherPoints)
        myOrderedDistances=np.sort(myDistancesToOtherPoints)

    myLocalPopulation = aPopulationMatrix[myNeighbours,:]
    if aSigma=='NA':
        #Retourne la population de chaque unité dans l'ordre de voisinage:réindexage

        # Calcul la somme cumulative de chaque groupe
        myCumulPop = np.cumsum(myLocalPopulation, axis=0)
        #vecteur des sommes cumulée sur tous les groupes
        mySumCumulPop = np.sum(myCumulPop, axis=1)
        #Division termes à termes des sommes cumulées partielles et de la population totale cumulée
        myCumulPropPop = myCumulPop / mySumCumulPop[:,np.newaxis]
        #Division termes à termes de la proportion suivante (matrice) par la proportion globale de chaque groupe
        myCumulPropPopVSGlobalProp = myCumulPropPop / aGlobalFigure
        #Le log de ce rapport de rapport
        myLogCumulPropPopVSGlobalProp = np.log(myCumulPropPopVSGlobalProp)

        #Traiter les 0log(0)
        myLogCumulPropPopVSGlobalProp[myLogCumulPropPopVSGlobalProp == -inf] = 0

        #Remultiplier par les proportions cumulées de chaque groupe et en faire la somme sur le nombre de groupes
        myDiv = np.sum(np.multiply(myCumulPropPop,myLogCumulPropPopVSGlobalProp),axis = 1)
    
        myWeightedSumPopPerLocation='No weight assigned to population'

    else :
        myKernel = sci.exp(-myOrderedDistances ** 2 / aSigma ** 2)
        myLocalWeightedPopulation=(myLocalPopulation.transpose()*myKernel).transpose()
        myCumulWeightedPop = np.cumsum(myLocalWeightedPopulation, axis=0)
        mySumWeightedPop = np.sum(myLocalWeightedPopulation, axis=0)
        mySumCumulWeightedPop = np.sum(myCumulWeightedPop, axis=1)
        myWeightedSumPopPerLocation = np.sum(myLocalWeightedPopulation, axis=1)
        myCumulWeightedPropPop = myCumulWeightedPop / mySumCumulWeightedPop[:,np.newaxis]
        myCumulWeightedPropPopVSGlobalProp = myCumulWeightedPropPop / aGlobalFigure
        myLogCumulWeightedPropPopVSGlobalProp = np.log(myCumulWeightedPropPopVSGlobalProp)
        myLogCumulWeightedPropPopVSGlobalProp[myLogCumulWeightedPropPopVSGlobalProp == -inf] = 0
        myDiv = np.sum(np.multiply(myCumulWeightedPropPop,myLogCumulWeightedPropPopVSGlobalProp),axis = 1)


    return (myNeighbours, myDiv,myWeightedSumPopPerLocation)

def max_index(geodata, sumpergroup,sharepergroup):
    new_pop=np.zeros((len(geodata),len(sumpergroup)))
    ordersum=sorted(sumpergroup)
    ordershare=sorted(sharepergroup)
    for i in range(len(geodata)):
        C=sum(sumpergroup)//len(geodata)#capacité moyenne d'une unité
        for j in range (len(sumpergroup)):
            new_pop[i,j]=min(C,ordersum[j])
            C-=new_pop[i,j]
            ordersum[j]-=new_pop[i,j]
    myCumulPop = np.cumsum(new_pop, axis=0)
    #vecteur des sommes cumulée sur tous les groupes
    mySumCumulPop = np.sum(myCumulPop, axis=1)
    #Division termes à termes des sommes cumulées partielles et de la population totale cumulée
    myCumulPropPop = myCumulPop / mySumCumulPop[:,np.newaxis]
    #Division termes à termes de la proportion suivante (matrice) par la proportion globale de chaque groupe
    myCumulPropPopVSGlobalProp = myCumulPropPop / ordershare
    #Le log de ce rapport de rapport
    myLogCumulPropPopVSGlobalProp = np.log(myCumulPropPopVSGlobalProp)
    #Traiter les 0log(0)
    myLogCumulPropPopVSGlobalProp[myLogCumulPropPopVSGlobalProp == -inf] = 0
    #Remultiplier par les proportions cumulées de chaque groupe et en faire la somme sur le nombre de groupes
    myDiv = np.sum(np.multiply(myCumulPropPop,myLogCumulPropPopVSGlobalProp),axis = 1)
    mymaxDistortion = np.sum(myDiv)/len(geodata)
    return(mymaxDistortion)

def dd_profile_data(aGeodata, aPopulation,aKey,aPath='crows',aSig='NA'):
    """
    Calculation of Divergence trajectories and Distorsion index

    Parameters
    ----------
    aGeodata          : a geopandas DataFrame with a geometry column.

    aPopulation       : a list of strings refering to the name of variable in data that contains the population size of the group of interest

    aSig              : a float
                      Sigma float for kernel computation.

    aKey              : a string
                      Identification if aPath!='crows'

    aPath1            : a floyd Warshall graph
                      Distances for data, default 'crows'=crow distance.
    Returns
    ----------
    myAllDivergence   : list of floats
                        Divergence KL Trajectory from each geographic unit
    myAllDistortion   : list of floats
                        Distorsion (maxDivergence(-1)) Trajectory from each geographic unit

    myGeodata         : a geopandas DataFrame
                        A geopandas DataFrame that contains divergence, distorsion and distorsion index (no profile for shapefile conversion).
    Notes
    -----
    Based Julien Randon-Furling, Madalina Olteau, Antoine Lucquiaud. "From Urban segregation to spatial structure detection ."


    """

    if (str(type(aGeodata)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame, this function cannot run.'
        )

    if ('geometry' not in aGeodata.columns):
        aGeodata['geometry'] = aGeodata[aGeodata._geometry_column_name]
        aGeodata = aGeodata.drop([data._geometry_column_name], axis=1)
        aGeodata = aGeodata.set_geometry('geometry')

    if (type(aPopulation) is not list):
        raise TypeError('Variables are not referenced correctly : please enter a list of strings')

    # Calculer les statistiques générales sur les populations en question
    myNGroups = len(aPopulation)

    # Nombre d'unites
    myNUnits = len(aGeodata)

    # Initialiser les sorties
    myAllDivergence = []
    myDiv=[]
    myExDiv=[]
    myExKDiv=[]
    myAllMaxDistortion = []
    myAllMinDistortion = []
    myMaxDistortionIndex = []
    myMinDistortionIndex = []
    myAllDelta = []
    myDeltaDistortionIndex=[]

    # Permuter aléatoirement les index du dataframe et les réindexer : randomiser le choix des voisins à distance égale
    myGeodata = aGeodata.sample(frac=1).reset_index(drop=True)

    # Créer l'objet consituté de tous les points de la grille
    myCoordinates = myGeodata.geometry.centroid
    myGEOID= myGeodata[aKey]

    # Liste des populations par unité (en float)
    myPopulationMatrix = np.zeros((myNUnits,myNGroups))
    for j in range (myNGroups):
        myPopulationMatrix[:,j] = list(map(float,myGeodata[aPopulation[j]]))

    # Calcul la population totale de chaque unite
    mySumPopPerLocation = np.sum(myPopulationMatrix, axis=1)
    mySumPopPerGroup = np.sum(myPopulationMatrix, axis=0)
    myTotalPop=np.sum(myPopulationMatrix)
    myGlobalFigures=mySumPopPerGroup/myTotalPop
    myRelativeDensity=mySumPopPerLocation/myTotalPop
    myNormalIndex=max_index(myGeodata,mySumPopPerGroup,myGlobalFigures)
    # Une itération pour une origine i :
    for i in range (myNUnits):
        # Récupérer la divergence pour chaque niveau d'agrégation à partir du point i :
        myNeighbors, myDivergence, myWeightedPopPerLocation= _divergence(myGeodata, i, myCoordinates, myGEOID, myPopulationMatrix,myGlobalFigures,aSig,aKey,aPath)

        if aSig=='NA':
            # Initialiser les sorties de la boucle myDistortion :
            myMaxDistortion = np.zeros(myNUnits)
            myMinDistortion = np.zeros(myNUnits)
            myFunctionMax = 0
            myFunctionMin = myDivergence[0]

            for l in range (1, myNUnits+1):
                # définir la fonction myDistortion de manière la plus précise possible (comme max de toutes les suivantes)
                myFunctionMax = max(myDivergence[-l],myFunctionMax)
                myMaxDistortion[-l] = myFunctionMax
                myFunctionMin = min(myFunctionMin, myDivergence[l-1])
                myMinDistortion[l-1]=myFunctionMin

            DeltaMinMax=list(map(float.__sub__,myMaxDistortion,myMinDistortion))
            myMaxIndex = np.sum(np.multiply(myMaxDistortion,mySumPopPerLocation[myNeighbors]))/myTotalPop
            myMinIndex = np.sum(np.multiply(myMinDistortion,mySumPopPerLocation[myNeighbors]))/myTotalPop
            myDeltaIndex = np.sum(np.multiply(DeltaMinMax,mySumPopPerLocation[myNeighbors]))/myTotalPop
            myExpectedDivergence = np.sum(np.multiply(myDivergence,mySumPopPerLocation[myNeighbors]))/myTotalPop
            myMaxDistortionIndex.append(myMaxIndex/myNormalIndex)
            myMinDistortionIndex.append(myMinIndex/myNormalIndex)
            myAllDelta.append(DeltaMinMax/myNormalIndex)
            myDeltaDistortionIndex.append(myDeltaIndex/myNormalIndex)
            myExDiv.append(myExpectedDivergence)
            myAllMaxDistortion.append(list(myMaxDistortion))
            myAllMinDistortion.append(list(myMinDistortion))
        else :
            myTotalWeightedPop=np.sum(myWeightedPopPerLocation)
            myExpectedWeightedDivergence = np.sum(np.multiply(myDivergence,myWeightedPopPerLocation))/myTotalWeightedPop
            myExKDiv.append(myExpectedWeightedDivergence)

        myAllDivergence.append(list(myDivergence))
        myDiv.append(myDivergence[0])

    myGeodata['Divergence']=pd.Series(myDiv, index = [i for i in range (len(myGeodata))])

    if aSig=='NA':
        myGeodata['ExpectedDivergence'] = pd.Series(myExDiv, index = [i for i in range (len(myGeodata))])
        myGeodata['MaxDistortionIndex'] = pd.Series(myMaxDistortionIndex, index = [i for i in range (len(myGeodata))])
        myGeodata['MinDistortionIndex'] = pd.Series(myMinDistortionIndex, index = [i for i in range (len(myGeodata))])
        myGeodata['Delta']= pd.Series(myDeltaDistortionIndex, index = [i for i in range (len(myGeodata))])
        ExpectedDivergence=np.array(myExDiv)
        #WeightedMeanEKL=np.sum(np.multiply(ExpectedDivergence, myRelativeDensity))
        WeightedMeanDistortion=np.sum(np.multiply(myMaxDistortionIndex, myRelativeDensity))
        Divergence=np.array(myDiv)
        WeightedMeanDiv=np.sum(np.multiply(Divergence, myRelativeDensity))

    else :
        myGeodata['ExpectedKernelDivergence'] = pd.Series(myExKDiv, index = [i for i in range (len(myGeodata))])
        ExpectedDivergence=np.array(myExKDiv)
        WeightedMeanDistortion=np.sum(np.multiply(ExpectedDivergence, myRelativeDensity))
        Divergence=np.array(myDiv)
        WeightedMeanDiv=np.sum(np.multiply(Divergence, myRelativeDensity))

    return (myAllDivergence,myGeodata,WeightedMeanDiv,WeightedMeanDistortion)


# In[11]:


class DD_profile :
    """
    Calculation of divergence distorsion profile and index.
    Parameters
    ----------
    aGeodata          : a geopandas DataFrame with a geometry column.

    aPopulation       : list of strings
                        The names of variable in data that contains the population size of the groups of interest

    Attributes
    ----------
    divergence : list of floats
                Divergence profile for each geographic unit.
    distorsion : list of floats
                Distorsion profile for each geographic unit.
    index      : float between 0 and 1
                Distorsion index.

    all_data : a geopandas DataFrame
               A geopandas DataFrame that contains the calculated columns.
    """

    def __init__(self,aGeodata, aPopulation,aSig,aKey,aPath='crows'):

        aux = ListDivergence(aGeodata,aPopulation)[3]

        self.divergence = aux['Divergence']
        self.divergence_profile = aux['DivergenceProfile']
        if aSig=='NA':
            self.expected_divergence = aux['ExpectedDivergence']
            self.frame_breadth = aux['Delta']
        else :
            self.kernel_divergence = aux['ExpectedKernelDivergence']
        self._function = dd_profile_data





# In[12]:


def dd_marginal_profile_data(aGeodata, aPopulation,aKey, aPath='crows',singlePop='NA',aSig='NA'):

    """
    Calculation of Divergence trajectories and Distorsion index

    Parameters
    ----------
    aGeodata          : a geopandas DataFrame with a geometry column.

    aPopulation       : a list of strings refering to the name of variable in data that contains the population size of the group of interest

    aSig              : a float
                      Sigma float for kernel computation.

    aKey              : a string
                      Identification if aPath!='crows'

    aPath1            : a floyd Warshall graph
                      Distances for data, default 'crows'=crow distance.

    aSinglePop        : a string
                      Identity of population used for marginal divergence computation
    Returns
    ----------
    myAllDivergence   : list of floats
                        Divergence KL Trajectory from each geographic unit
    myAllDistortion   : list of floats
                        Distorsion (maxDivergence(-1)) Trajectory from each geographic unit

    myGeodata         : a geopandas DataFrame
                        A geopandas DataFrame that contains divergence, distorsion and distorsion index (no profile for shapefile conversion).
    Notes
    -----
    Based Julien Randon-Furling, Madalina Olteau, Antoine Lucquiaud. "From Urban segregation to spatial structure detection ."


    """

    if (str(type(aGeodata)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame, this function cannot run.'
        )

    if ('geometry' not in aGeodata.columns):
        aGeodata['geometry'] = aGeodata[aGeodata._geometry_column_name]
        aGeodata = aGeodata.drop([data._geometry_column_name], axis=1)
        aGeodata = aGeodata.set_geometry('geometry')

    #if ((type(aPopulation) is not list) or (len(aPopulation)==0)):
        #raise TypeError('Variables are not referenced correctly : please enter a list of strings')

    #if ((singlepop !='NA') and (singlepop is not in aPopulation)):
        #raise ValueError('The chosen reference group does not exist')


    myNewGeodata=aGeodata
    if singlePop=='NA' :

        # Calculer les statistiques générales sur les populations en question
        myNGroups = len(aPopulation)
        myPopulation=aPopulation

    else :
        myNGroups=2
        myOtherPopulation=aPopulation
        myOtherPopulation.remove(singlePop)
        PopData=aGeodata[myOtherPopulation]
        myOtherPopulationSum=PopData.sum(axis=1).copy()
        myPopulation=[singlePop,'Other']
        myNewGeodata.loc[:,'Other']=myOtherPopulationSum


    return (dd_profile_data(myNewGeodata,myPopulation,aKey,aPath,aSig))




# In[13]:


class DD_marginal_profile :
    """
    Calculation of marginal divergence distorsion profile and index given a population group.
    Parameters
    ----------
    aGeodata          : a geopandas DataFrame with a geometry column.

    aPopulation       : list of strings
                        The names of variable in data that contains the population size of the groups of interest
    singlepop         : string
                        The name of variable in data that contains the population size of the individual group of interest

    Attributes
    ----------
    divergence : list of floats
                Divergence profile for each geographic unit.
    distorsion : list of floats
                Distorsion profile for each geographic unit.
    index      : float between 0 and 1
                Distorsion index.

    all_data : a geopandas DataFrame
               A geopandas DataFrame that contains the calculated columns.
    """

    def __init__(self, aGeodata, aPopulation,aKey, aPath,singlePop, aSig):

        aux = dd_marginal_profile_data(aGeodata,aPopulation,aKey,aPath,singlePop,aSig)[3]
        if aSig=='NA':
            self.index=aux['ExpectedKernelDivergence']
        else :
            self.index=aux['ExpectedDivergence']
        self.divergence = aux['DivergenceProfile']
        self._function = _dd_profile_data





def dd_time_serie(aGeodata1, aGeodata2,aPopulation,aKey,aPath1,aPath2,aSig='NA',distortion='Add'):
    """
    Comparison of divergence index at 2 different times :

    Parameters
    ----------
    aGeodata1                  : a geopandas
                               DataFrame with a geometry column and data for time 1.

    aGeodata2                  : a geopandas
                               DataFrame with a geometry column and data for time 2.

    aPopulation                : a list of strings
                               Refering to the name of variable in data that contains the population size of the group of interest (if same groups in 1 and 2).

    aSig                       : a float
                               Sigma float for kernel computation.

    aKey                       : a string
                               Identification for divergence computation if not done.

    aPath1                     : a floyd Warshall graph
                               Distances for data1, default 'crows'=crow distance.

    aPath2                     : a floyd Warshall graph
                               Distances for data2, default 'crows'=crow distance.

    distortion_object          : a boolean
                               if divergence already computed and added to aGeodatax, then True, default False (recomputation of divergence index for both dataframes).

    Returns
    ----------
    full_data                  : a dataframe with differences between both databases.

    Notes
    -----
    Based Julien Randon-Furling, Madalina Olteau, Antoine Lucquiaud. "From Urban segregation to spatial structure detection ."


    """
    if distortion=='Add':
        dd_1=dd_profile_data(aGeodata1,aPopulation,aKey,aPath1,aSig)[3]
        dd_2=dd_profile_data(aGeodata2,aPopulation,aKey,aPath2,aSig)[3]
    else :
        dd_1=aGeodata1
        dd_2=aGeodata2
    if aPath1!=aPath2:
        #divergence data1 avec nouvelle matrice de distance
        dist_change=dd_profile_data(aGeodata1,aPopulation,aKey,aPath2,aSig)[3]

    if aSig=='NA':
        dd_1['ExpectedKernelDivergence']=dd_1['ExpectedDivergence']
        dd_2['ExpectedKernelDivergence']=dd_2['ExpectedDivergence']
    else :
        dd_1['ExpectedDivergence']=dd_1['ExpectedKernelDivergence']
        dd_2['ExpectedDivergence']=dd_2['ExpectedKernelDivergence']
    #Tri des données
    dd_1=dd_1[aPopulation+[aKey,'ExpectedDivergence','Divergence','ExpectedKernelDivergence']]
    dd_1.columns=aPopulation+[aKey,'ExpectedDivergence_1','Divergence_1','ExpectedKernelDivergence_1']
    dd_2=dd_2[aPopulation+[aKey,'ExpectedDivergence','Divergence','ExpectedKernelDivergence','geometry']]
    dd_2.columns=aPopulation+[aKey,'ExpectedDivergence_2','Divergence_2','ExpectedKernelDivergence_2','geometry']

    if aPath1!=aPath2:
        dist_change=dist_change[aPopulation+[aKey,'ExpectedDivergence','Divergence','ExpectedKernelDivergence']]
        dist_change.columns=aPopulation+[aKey,'ExpectedDivergence_12','Divergence_12','ExpectedKernelDivergence_12']

        #aligner les données suivant la clé
        full_data=pd.merge(pd.merge(dd_1,dd_2,on=aKey),dist_change,on=aKey)

        #La part de différence liée au changements démographiques
        full_data['composition_var_ratio']=(full_data.ExpectedDivergence_12-full_data.ExpectedDivergence_2)/(full_data.ExpectedDivergence_1-full_data.ExpectedDivergence_2)

        #La différence liée au changement géographique
        full_data['distance_var']=(full_data.ExpectedDivergence_1-full_data.ExpectedDivergence_12)

        #Différence dans le calcul de la divergence (changement geographique +demographique)
        full_data['total_var']=(full_data.ExpectedDivergence_1-full_data.ExpectedDivergence_2)

    else :

        full_data=pd.merge(dd_1,dd_2,on=aKey)
        full_data['total_var']=(full_data.ExpectedDivergence_1-full_data.ExpectedDivergence_2)

    #Traitements démographiques
    myPopulationMatrix_1 = np.zeros((len(full_data),len(aPopulation)))
    for j in range (len(aPopulation)):
        myPopulationMatrix_1[:,j] = list(map(float,full_data[aPopulation[j]+'_x']))

    myPopulationMatrix_2 = np.zeros((len(full_data),len(aPopulation)))
    for j in range (len(aPopulation)):
        myPopulationMatrix_2[:,j] = list(map(float,full_data[aPopulation[j]+'_y']))

    # Calcul la population totale de chaque unite
    mySumPopPerLocation_1 = np.sum(myPopulationMatrix_1, axis=1)
    mySumPopPerLocation_2 = np.sum(myPopulationMatrix_2, axis=1)

    #Population totale de chaque groupe
    mySumPopPerGroup_1 = np.sum(myPopulationMatrix_1, axis=0)
    mySumPopPerGroup_2 = np.sum(myPopulationMatrix_2, axis=0)

    #Proportion dans chaque unité pour chaque groupe :
    myPropPerLocation_1=np.transpose(np.divide(np.transpose(myPopulationMatrix_1),mySumPopPerLocation_1))
    myPropPerLocation_2=np.transpose(np.divide(np.transpose(myPopulationMatrix_2),mySumPopPerLocation_2))

    myLocalLog_1 = np.log(myPropPerLocation_1)
    myLocalLog_2 = np.log(myPropPerLocation_2)

    #Traiter les 0log(0)
    myLocalLog_1[myLocalLog_1 == -inf] = 0
    myLocalLog_2[myLocalLog_2 == -inf] = 0

    #Divergence Locale
    myLocalLogRatio=myLocalLog_2-myLocalLog_1
    myLocalDiv = np.sum(np.multiply(myPropPerLocation_2,myLocalLogRatio),axis = 1)

    myTotalPop_1=np.sum(myPopulationMatrix_1)
    myGlobalFigures_1=mySumPopPerGroup_1/myTotalPop_1

    myTotalPop_2=np.sum(myPopulationMatrix_2)
    myGlobalFigures_2=mySumPopPerGroup_1/myTotalPop_2

    for j in range (len(aPopulation)):
        full_data['share_'+aPopulation[j]+'_1']=myPropPerLocation_1[:,j]
        full_data['share_'+aPopulation[j]+'_2']=myPropPerLocation_2[:,j]
        full_data[aPopulation[j]+'_var']=myPropPerLocation_2[:,j]-myPropPerLocation_1[:,j]
    full_data['total_var_pop']=mySumPopPerLocation_2-mySumPopPerLocation_1
    full_data['local_divergence_ratio']=myLocalDiv
    return (full_data)




def DivergenceTraj(aDivergenceData,aKey,GeoUnits):
    '''
    Divergence Trajectories for i in GeoUnits

    '''
    for i in GeoUnits :
        aDivergenceData[aDivergenceData[aKey]==i].DivergenceProfile.plot()
