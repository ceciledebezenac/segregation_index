
import numpy as np

from shapely.ops import cascaded_union
import math
import pandas as pd

import matplotlib.pyplot as plt
from divergence.basic_analysis import Demographics
from divergence.neighbourhood import Neighbourhood
import seaborn as sns


def filter_na(data,variable,show_null=True):
    '''Filter the null or Nan data from dataframe for plotting.'''
    if show_null==True:
        #show as an option what has been filtered out.
        data[data.isnull().any(axis=1)].plot()
        print('These units have been excluded from the analysis. Check data or continue without.\n\n' ,data[data.isnull().any(axis=1)]['TRACT'])
    return(data[data[variable].notnull()])
    
    
    
def kl_divergence_profile(population_matrix,ordered_proportions):
    ''' Compute the KL divergence profiles on aggregated units.
    Parameters
    ----------
    - population_matrix: numpy array of integers
        population matrix with ordered rows (all units ordered by similarity to a given origin)
    - ordered_proportions: numpy array of floats
        global proportions of each group in a given order 
        (this order is generally arbitrary but is important in case the data has been permuted in previous analysis, or for maximum segregation calculation.
        See theoretical_max_distortion()).
    
    Returns
    -------
    numpy array of KL divergence values for each aggregation level based on the ordered input data.
    
    '''
    #Calculate distortion on the sorted population matrix :
    #cumulated population count for each group going through the units in the increasing order.
    cumul_pop = np.cumsum(population_matrix, axis=0)
    #cumulated population count for all groups "".
    sum_cumul_pop = np.sum(cumul_pop, axis=1)
    #Proportions of each group on cumulated population : proportions in aggregated units p.
    cumul_proportions = cumul_pop / sum_cumul_pop[:,np.newaxis]
    #Division of proportions by the global statistics of the city : p/q
    relative_cumul_proportions = cumul_proportions / ordered_proportions
    #log(p/q)
    log_relative_cumul = np.log(relative_cumul_proportions)
    #deal with null execptions : 0log(0)=0
    log_relative_cumul[log_relative_cumul == -math.inf] = 0
    # multiply log of the ratio p and q by the local proportions : sum(plog(p/q)) on population groups (see KL divergence of probability distributions).
    div = np.sum(np.multiply(cumul_proportions,log_relative_cumul),axis = 1)
    return(div)
    
    
    
def theoretical_max_distortion(demographics):
        
    ''' 
    This build-in function calculates the maximum distortion possible given the global proportions of each group. 
    The maximum segregated configuration is met when each group is alone in the spatial units which form concentric areas around the smallest group.
    The population counts for all groups are ordered and progressively populated units around a arbitrary origin (unit 0) 
    until their maximum capacity has been reached. The next smallest group then starts populating the next units when the first group is placed 
    This process is repeated until all the groups have been placed in concentric areas. 
    This index is then used to normalise the local values of the effective distortion index.
    For more information, see documentation or Olteanu and al. (2019). 
    
    Parameters
    ----------
    gedata : Geopandas
    group_count : numpy array
        array of population count for each group
    
    Returns 
    -------
    theo_max_distortion : float
        the distortion_index (or expected_divergence) of the most segregated configuration as described above. 
    
    '''
    #initialise a population matrix with new order : 
    segregated_population=np.zeros_like(demographics.population_matrix)
    
    #sort the groups by population count to start placing the smallest groups.
    ordered_groups=sorted(demographics.sum_pop_group)
    ordered_proportions=sorted(demographics.global_statistics)
    
    #fill in the population matrix with the sorted population, group by group till the unit the maximum capacity (mean over all units) has been reached.
    for unit in range(len(segregated_population)):
        #mean capacity
        Capacity=sum(ordered_groups)//len(segregated_population)
        for group in range (len(ordered_groups)):
            #fill in unit with group until full or no more members left.
            segregated_population[unit,group]=min(Capacity,ordered_groups[group])
            Capacity-=segregated_population[unit,group]
            ordered_groups[group]-=segregated_population[unit,group]
    #calculate the divergence profile over the aggregation process with kl_divergence. 
    divergence_profile=kl_divergence_profile(segregated_population,ordered_proportions)
    #calculate the mean of the profile equal to the distortion_index of the most segregated unit in the most segregated configuration possible. 
    theo_max_distortion = np.sum(divergence_profile)/len(segregated_population)
    return(theo_max_distortion)
    
  

       
class LocalDivergenceProfile():
    '''Compute the divergence profile for a given geographic unit.
    Parameters
    ==========
        - city: DivergenceProfiles class 
                city data with dataframe, neighbourhood structure and demographic information
        - origin: integer
                  the location of the unit in the dataframe 
    Attributes
    ==========
        - geodata: CityDivergence object
                   the data from the unit environment (see CityDivergence)
        - origin: dictionary of attributes
                  id=index
                  coordinates=geographic coordinates of origin centroid.
        - neighbours: numpy array
                      index of ordered units by increasing dissimilarity.
        - sorted_population: numpy array
                             population count for groups in each unit, ordered by closest neighbours. 
        - sup_distortion: float
                          index of segregation in segregated configuration. 
        - profile: numpy array
                   KL divergence for increasingly aggregated neighbourhoods around origin, from local unit to entire city (where divergence is null). 
        - divergence: float
                      KL divergence of local unit.
        - max_profile: numpy array
                       focal distances (argmax(aggregated KL divergence < threshold)) for threshold values in [0,1]. See Olteanu and al. (2019). 
        - min_profile: numpy array
                       focal distances (argmin(aggregated KL divergence < threshold)) for threshold values in [0,1].
        - distortion_index: float
                     mean focal distances over all threshold values for last passage (max_profile)
        - min_index: float
                     mean focal distances over all threshold values for first passage (min_profile)
        - delta_index: float
                       difference distortion_index - min_index, or the enveloppe of the profile.
        - expected_divergence: float
                               expected value of KL divergence over the random choice of aggregated level (or empirical mean value of the KL divergence for all aggregated levels).
            

    Methods
    =======
        - calculate_indexes()
            Returns: indices as class attributes (max_profile, min_profile, distortion_index, min_index, delta_index, expected_divergence) 
            based on divergence profile.


        '''

    def __init__(self,city,origin):
        
        #acces city attributes to avoid keeping a copy of city in memory
        self.city_demography=city.demography
        #define data structure for origin as an identity, centroid (Point) and shape (Polygon)
        self.origin={'id':city.dataframe.index[origin],'coordinates':city.dataframe.geometry.centroid[origin],'polygon':city.dataframe.geometry[origin]}
        #acces the ordered neighbours of the origin point, from itself to the entire city. 
        self.neighbours=city.structure.neighbours[origin]
        #sort the population matrix to follow the order of neighbours.
        self.sorted_population=self.city_demography.population_matrix[self.neighbours,:]
        #access the normalisation coefficient attribute from the city
        self.sup_distortion=city.sup_distortion
        #compute the kl divergence profile for all levels of aggregation. 
        self.profile=kl_divergence_profile(self.sorted_population,self.city_demography.global_statistics)
        #add as an attribute the divergence of the local area for quick access and comparison if needed by users (comparing multiscale index and local index).
        self.divergence=self.profile[0]
        #initialise max and min profiles as zero numpy array of the size of the city (rows in dataframe)
        self.max_profile=np.zeros(len(self.sorted_population))
        self.min_profile=np.zeros(len(self.sorted_population))
        #initialise distortion indices before method call
        self.distortion_index=None
        self.min_index=None
        self.delta_index=None
        self.expected_divergence=None
    

    def calculate_indices(self):
        '''Calculate the indices from divergence profile as trjaectory summary.'''
        #total population in each unit in the right order (closest neighbours).
        sorted_pop_local=np.sum(self.sorted_population,axis=1)
        #initialise local variables
        max_distortion=0#max from end
        min_distortion=self.profile[0]
        #compute the max_profile as the maximum KL divergence value from the end : last passage for theoretically continuous thresholds. 
        #compute the min_profile as the minimum KL divergence value from the start : last passage for theoretically continuous thresholds. 
        for level in range(1,len(self.sorted_population) + 1):
            max_distortion = max(self.profile[-level] , max_distortion)
            self.max_profile[-level] = max_distortion
            min_distortion = min(min_distortion , self.profile[level-1])
            self.min_profile[level-1]=min_distortion
        #change data type of max_profile and min_profile for calculations 
        delta_min_max=list(map(float.__sub__,self.max_profile , self.min_profile))
        #compute distortion_index as weighted (by population count) mean of max_profile
        self.distortion_index=np.sum(np.multiply(self.max_profile , sorted_pop_local))/self.city_demography.total_pop
        #normalise distortion_index with normalisation coefficient sup_distortion.
        self.distortion_index_normal=self.distortion_index / self.sup_distortion 
        #compute min_index as weighted (by population count) mean of min_profile
        self.min_index=np.sum(np.multiply(self.min_profile , sorted_pop_local))/self.city_demography.total_pop
        #compute delta_index as weighted (by population count) mean of delta_min_max
        self.delta_index=np.sum(np.multiply(delta_min_max , sorted_pop_local))/self.city_demography.total_pop
        #compute expected_divergence as weighted mean of profile. 
        self.expected_divergence=np.sum(np.multiply(self.profile , sorted_pop_local))/self.city_demography.total_pop



class DivergenceProfiles :
    ''' 
    Calculate distortion indices for city units.
    Given a geopandas dataframe and a list of categorical variables (ethnic affiliation), use `DivergenceProfiles`
    to compute divergence profiles of all statistical units and calculate focal distances over all threshold values. 
    Update dataframe with local indices and plot the map for the different indices. 
    
    Parameters
    ==========
    geodata : Geopandas class
        the city data including population count and geometry column.
    groups : list
        the list of variables used to categorise the population : the ethnic groups used to measure the segregation. 
    path : string
        the path used to compute the neighbourhood structure. 
        modality=[*'euclidean'*,*'demographic'*,*'divergence'*]
    
    Attributes
    ==========
    dataframe : Geopandas 
        copy of original dataframe with added columns after update_data() call:
            - distortion_index : the mean focal distance calculated with the 'last passage' approach 
            (or the last time the divergence passes a threshold before converging to 0 throughout the aggregation process). See documentation and Olteanu and al.(2019).
            - min_index : the mean focal distance calculated with the **first passage** approach.
            - delta_index : the difference of min and max indices. This index corresponds to the enveloppe of the individual trajectory. 
            - expected_divergence : the expected value of divergence over the choice of aggregation level centered on the local unit. 
   
    size : integer
        size of relevant dataframe (filetering nan values and null population count)
    
    shape : shapely Polygon
        overall city shape
   
    coodinates : GeoSeries
        series of geometric objects : Polygon of statistical units used. 
    
    groups : list of strings
        names of variables used for population categories. 
    
    demography : Demographic class
        global demographic statistics, population numpy array
    
    sup_distortion : float
        distortion index of the theoretically most segregated city given global statistics. See function ``theoretical_max_distortion``.
    
    Dependant Attributes
    ====================
    structure : Neighbourhood class
        dependancy: set_neighbourhood()
        the neighbouring structure in the city based of the path given by the user. 
    
    divergence_data : list of LocalDivergenceProfile class
        dependancy: set_profiles()
        the divergence profile and distortion indices (``distortion_index``, ``min_index``,``expected_divergence``) for all units.
    
    hindex : float
        dependancy: set_profiles()
        the *H-index* or entropy index descibed by Reardon(2002). 
    
    ehindex : float
        dependancy: set_profiles()
        the mean expected_divergence over all units. 
    
    mean_distortion_index : float
        dependancy: set_profiles()
        the mean distortion_index over all units. 
        
    
    
    Example
    =======
        

    '''
    def no_null(geodata,groups):
        '''find null values on total of all group counts and exclude them as they will drive the KL divergence to nan values.'''
        
        geocopy=geodata.copy()
        geocopy['total']=geocopy[groups].sum(axis=1)
        
        print(str(len(geocopy[geocopy['total']==0]) )+' spatial units have been left out because of null values over all groups. Check your data or continue.')
        geocopy=geocopy[geocopy['total']>0]
        return(geocopy)
        
    def __init__(self,geodata,groups):
        
        #initialise the attributes
        #get rid of the columns that could cause a problem when computing divergence. 
        self.dataframe=DivergenceProfiles.no_null(geodata,groups)
        
        #make sure the indexing is in order (after data cleaning and filtering rows)
        self.dataframe.reset_index(drop=True,inplace=True)
    
        self.size=len(self.dataframe)
        #the aggregated polygon of the whole city 
        self.shape=cascaded_union(self.dataframe.geometry)
       
        #but would be of use if graphs (dictionary) replace the numpy array structure of the population data : future update for graph distance instead of crow distance. 

        self.groups=groups
        #the polygon objects that make up the whole city.
        self.coordinates=self.dataframe.geometry
        #get the global demographic statistics with the class Demographics described above. 
        self.demography=Demographics(self.dataframe,self.groups)
        #calculate the maximum index possible as normalisation coefficient. 
        self.sup_distortion=theoretical_max_distortion(self.demography)
        #initialise the attributes that will be created by calling class methods. 
        self.divergence_data=None
        self.structure=None
        self.hindex=None
        self.ehindex=None
        self.mean_distortion_index=None
        
   
    def set_neighbourhood(self,path='euclidean',marginal_group='None'):
        '''
        This method will give the particular neighbourhood structure of the dataframe based on geographic distance, 
        bilateral *KL-divergence* between units, or demographic distance. 
        The neighbourhood matrix used represents for each unit the ordered indices of all the other units including itself.
        
        Parameters 
        ----------
        path : string
            the type of difference measure, 
        marginal_group : string
            variable in groups with which to binarise the population, marginal_group vs other.
                     
        Returns
        -------
        updated population information if marginal group given, 
        structure attribute : Neighbourhood class
        '''
        
        
        if marginal_group!='None':
            #segment the population accordingly with specific group and all the others
            others='not_'+marginal_group
            new_groups=[marginal_group,others]
            #make a copy of the name list so as not to delete the group name from original list
            other_population=self.groups.copy()
            other_population.remove(marginal_group)
            #add a column to the original dataframe through a view with loc.
            self.dataframe.loc[:,others]=self.dataframe[other_population].sum(axis=1)
            #create a Demographic class for the city for population information
            self.demography=Demographics(self.dataframe,new_groups)
      
            
        #create a Neighborhood class to compute the specific neighbourhood structure type.
        self.structure=Neighbourhood(self.dataframe,self.groups)
        #access the right method of Neighbourhood instance created above to give it a particular structure with the 'path' parameter. 
        if path=='euclidean' :
            self.structure.set_euclidean_neighbours()
        else:
            self.structure.set_neighbourhood_type(path)
            
        
     
        
    def set_profiles(self):
        '''
        This method will compute, when called, the trajectories based on the specific neighbourhood structure for all the spatial units in the dataframe. 
        The aggregated indices are calculated for the global city based on the resulting local indices (*H-index* and *EH-index*). 
        This method is optionnally called by the user, if the latter wishes to compute profiles for specific units only, 
        the costly calculation on the entire city can be avoided. 
                     
        Returns
        -------
        updated values for:
            - divergence_data
            - hindex
            - ehindex
            - mean_distortion_index
        
        '''
        #check if the set_neighbourhood() method has been called first and send an error message.
        if not self.structure:
            raise ValueError('The neighbourhood structure has not yet been defined. Please call the set_neighburhood() method on the DivergenceProfiles class.')
        else:
            #initialise list for receiving divergence indices for each unit. 
            #Unlike other attributes, we do not add it to the __init__ method as we want the user to quickly resolve the problem if he has not called this method.
            self.divergence_data=[]
            for origin in range(self.size):
                #create profile instance for unit origin
                divergence_from_origin = LocalDivergenceProfile(self,origin)
                #compute all indices with calculate_indices() method
                divergence_from_origin.calculate_indices()
                #add the profile and indices in LocalDivergenceProfile structure into the divergence data list
                self.divergence_data.append(divergence_from_origin)
            #compute the global indices over all units once all profiles are given
            self.hindex=sum([self.divergence_data[i].profile[0]for i in range(self.size)])/self.size
            self.ehindex=sum([self.divergence_data[i].expected_divergence for i in range(self.size)])/self.size
            self.mean_distortion_index=sum([self.divergence_data[i].distortion_index_normal for i in range(self.size)])/self.size


    def update_data(self,suffixe=''): 
        '''
        This method will add on the original dataframe attribute (or copy of the original dataframe given as a parameter: check for SettingWithCopy Warning) 
        the columns of segregation indices calculated by the set_profile() method. This method must be called after ``set_profiles()`` and ``set_neighbourhood()``. 
        Parameters
        ----------
        suffixe : string
            optional suffixe to add to column name so as noit to overwrite previous index columns 
            (enabling comparison between indices resulting from different neighbourhood structures.)
        Returns 
        -------
        Upadated dataframe with columns "meax_index", "distortion_index_normal", "delta_index", "expected_divergence".
        
        '''
        #check if the other methods have been called.
        if not self.divergence_data:
            #raise error if other methods have not yet been called.
            raise ValueError('No value has been attributed to self.divergence_data. Call set_neighbourhood() and set_profiles() before updating.')
        else:
            #add columns for each index
            self.dataframe['distortion_index'+suffixe] = pd.Series(
                    [self.divergence_data[i].distortion_index for i in range(self.size)],
                    index = [i for i in range(len(self.dataframe))]
                    )
            self.dataframe['distortion_index_normal'+suffixe]=pd.Series([self.divergence_data[i].distortion_index_normal for i in range(self.size)],
            index=[i for i in range(len(self.dataframe))])
            self.dataframe['min_index'+suffixe]=pd.Series([self.divergence_data[i].min_index for i in range(self.size)],
            index=[i for i in range(len(self.dataframe))])
            self.dataframe['delta_index'+suffixe]=pd.Series([self.divergence_data[i].delta_index for i in range(self.size)],
            index=[i for i in range(len(self.dataframe))])
            self.dataframe['expected_divergence'+suffixe]=pd.Series([self.divergence_data[i].expected_divergence for i in range(self.size)],
            index=[i for i in range(len(self.dataframe))])
                
    def plot_map(self, variable,figure_size=(12,10),text='',color_type='Reds',set_ticks=False):
        '''
        This method will plot the map of a given local index. This could be a segregation measure or any other quantitative variable.
       
        Parameters
        ----------
        variable: string
            the variable (or column) the user would like to see represented on the map
        figure_size: tuple of integers
            the size of the figure
        text: string
            the text that should be displayed.
        color_type: string
            one of the cmap options given by matplotlib.pyplot for plotting maps. Check documation.
        set_ticks: Boolean
            keep the side ticks and bars on the map.
        
        Returns
        -------
        map of the city with a color scale for the quantitative variable. 
        
        '''
        #initialise figure
        f, ax = plt.subplots(1, figsize=figure_size)
        #plot the map given the parameters
        ax = self.dataframe.plot(column=variable,cmap=color_type,legend=True,ax=ax)
        #get rid of useless spines
        ax.spines["top"].set_visible(False)    
        ax.spines["bottom"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.spines["left"].set_visible(False)
        #get rid of ticks
        if set_ticks==False:
            ax.set_xticks([], [])
            ax.set_yticks([], [])
        #lims = plt.axis('equal')
        
    
    def plot_profiles(self,index_list,figure_size=(10,6)):#index_list=[i for i in range(self.size)],figure_size=(10,6)):
        '''
        This method will plot the KL divergence profiles of spatial units over all aggregation levels (the x axis being the aggregated units).
       
        Parameters
        ----------
        index_list: list of integers (location of row in dataframe)
            the units for profile plotting, from 1 to all spatial units.
        figure_size: tuple of integers
            the size of the figure
            default: (10,6)
        
        
        Returns
        -------
        Plots of profiles. These represent the KL divergence of the aggregated units for each aggregation level starting from the local units selected by the user. 
        
        '''
        #initialise figure
        plt.figure(figsize=figure_size) 
        ax = plt.subplot(111)    
        ax.spines["top"].set_visible(False)    
        ax.spines["bottom"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.spines["left"].set_visible(False) 
        #layout of the plot, manage ticks and labels
        ax.get_xaxis().tick_bottom()    
        ax.get_yaxis().tick_left() 
        
        plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                        labelbottom="on", left="off", right="off", labelleft="on")
        plt.xticks(fontsize=10) 
        plt.yticks(fontsize=10)    
        plt.xticks(fontsize=10)
        #plot the profiles given the parameters
        for i in index_list:
            plt.plot(self.divergence_data[i].profile,linewidth=1)
            
    def plot_distribution(self,variable,histogram=True,KDE=True,figure_size=(10,6),bins=20,color='red',):
        ''' 
        This method will plot the distribution of a given variable over all units (from the ones computed previously).
       
        Parameters
        ----------
        variable: string
            the variable (or column) whose distribution should be analysed.
        histogram: Boolean
            for the pyplot.plot parameters, show histogram.
        KDE: Boolean
            for the pyplot.plot parameters, show density curve from KDE method. 
        figure_size: tuple of integers
            the size of the figure
        bins: integer
            number of bins for the histogram.
        color: string
            one of the cmap options given by matplotlib.pyplot for plots.
        
        
        Returns
        -------
        Histogram and density plot for the particular variable in the city. 
        '''
        plt.figure(figsize=figure_size)    
        
        # Remove the plot frame lines.    
        ax = plt.subplot(111)    
        ax.spines["top"].set_visible(False)    
        ax.spines["bottom"].set_visible(True)    
        ax.spines["right"].set_visible(False)    
        ax.spines["left"].set_visible(True)    
        #ax.get_xaxis().tick_bottom()    
        #ax.get_yaxis().tick_left() 
        ax.tick_params(axis="both", which="both", bottom="off", top="off",    
                        labelbottom="off", left="off", right="off", labelleft="off")
        #actually plot the distribution with the kde method for interpolation of density function. 
        sns.distplot(self.dataframe[variable],hist=histogram,kde=KDE,bins=20, color = 'red',
                     hist_kws={'edgecolor': 'white'},kde_kws={'linewidth': 4})   
        # Plot formatting
        plt.title('Density Plot of Distortion Index')
        plt.xlabel('distortion')
        plt.ylabel('density')
                 

    def save_dataframe(self,file_name):
        ''' Save updated dataframe with path and file name.'''
        self.dataframe.to_file(file_name + '.shp')
        
        
        
        
        
        
