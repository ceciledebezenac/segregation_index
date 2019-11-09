import random as rd
import numpy as np
from shapely.geometry import Polygon
import pandas as pd
import geopandas as gdp




class RandomGrid():
    ''' 
    Random square lattice defined by size, population and dataframe.
    '''
    def __init__(self, height_of_grid, group_population ):
        self.size = height_of_grid**2
        self.height=height_of_grid
        self.nb_groups = len(group_population)
        self.population= group_population
        self.local_capacity=int(sum(np.array(self.population))/self.size)
        
    def populate(self):
        '''Uniformly random population attribution. '''
        Lat=[]
        Long=[]
        A=[]
        B=[]
        PolyLong=[]
        PolyLat=[]
        for i in range (self.height) :
            for j in range (self.height):
                Long.append(i)
                Lat.append(j)
                PolyLong.append([i,i,i+1,i+1])
                PolyLat.append([j,j+1,j+1,j])
        A=[i for i in range (self.size)]
        B=['unit_'+str(i) for i in range (self.size)]
        listpop=[]
        for l in range (self.nb_groups):
            listpop=listpop+[l for i in range (self.population[l])] 
        rd.shuffle(listpop)
        capacity=[self.local_capacity for i in range (self.size)]
        cap=np.array(capacity)
        pop=np.array(listpop)
        capcumul=np.cumsum(cap)
        unit=np.array([pop[:capcumul[0]]]+[pop[capcumul[i-1]:capcumul[i]] for i in range (1,self.size)])
        cap_group=[]
        for l in range (self.nb_groups):
            cap_group.append([(unit[i]==l).sum() for i in range (self.size)])
        self.data = pd.DataFrame({'A': A, 'B': B, 'Longitude': Long, 'Latitude': Lat,'PLongitude': PolyLong,'PLatitude': PolyLat},)
        for l in range (self.nb_groups):
            self.data['Population_'+str(l)]=pd.Series({'Popl':cap_group[l]})['Popl']
        


class GeoGrid():
    '''Geopandas format for dataframe based on polygon coordinates.'''
    
    def geocode(dataframe,polygon_longitude,polygon_latitude):
        #geometry = [Point(xy) for xy in zip(dataframe.longitude, dataframe.latitude)]
        geom = []
        for i in range (len(dataframe)):
            	geom.append(Polygon(zip(dataframe[polygon_longitude][i],dataframe[polygon_latitude][i])))
        crs = {'init': 'epsg:4326'}
        gdf = gdp.GeoDataFrame(dataframe, crs = crs, geometry = geom)
        gdf.drop([polygon_longitude,polygon_latitude],axis=1,inplace=True)
        return (gdf)
        
    
    def __init__(self,dataframe,longitude,latitude):
        self.geodata=GeoGrid.geocode(dataframe,longitude,latitude)
        
    
