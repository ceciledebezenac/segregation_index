#!/usr/bin/env python
# coding: utf-8

# In[42]:


from divergence import segregation_distortion as seg
from random_grid import random_grid as rdg
import geopandas as gdp
import pandas
import pickle as pck
import unittest
import random as rd


# In[43]:


#access the data for the test
f = open('fixatures/fixature_python_object','rb')
distortion_ref = pck.load(f)
f.close()
data_test=gdp.read_file('fixatures/fixature_random_data.shp')
data_test.rename(columns={'Population':'Population_0','Populati_1':'Population_1'},inplace=True)


# In[44]:


distortion_test=seg.DivergenceProfiles(data_test,['Population_0','Population_1'])
#create structure by calling the set_neighborhood method.
distortion_test.set_neighbourhood()
#compute the profiles and the indices.
distortion_test.set_profiles()
# In[45]:


class TestCity(unittest.TestCase):
    
    def test_initial_attributes(self):
        '''Test the creation of the city framework using DivergenceProfiles, 
        checking for the existance of initial attributes and their similarity to the reference object uploaded.'''
        self.assertEqual(distortion_test.size,distortion_ref.size)
        self.assertEqual(distortion_test.shape,distortion_ref.shape)
        self.assertEqual(distortion_test.sup_distortion,distortion_ref.sup_distortion)
        #select random indices to test:
        rand_1=rd.randint(0,24)
        rand_2=rd.randint(0,1)
        rand_3=rd.randint(0,24)
        self.assertEqual(distortion_test.demography.population_matrix[rand_1,rand_2],
                    distortion_ref.demography.population_matrix[rand_1,rand_2])
        self.assertEqual(distortion_test.coordinates[rand_3],distortion_ref.coordinates[rand_3])
    
    def test_neighbourhood(self):
        '''Test the neighbourhood structure created using the Neighbourhood class from the neighbourhood module.
        '''
        
        #check if structure is similar to reference.
        self.assertTrue(distortion_test.structure)
        #select random indices to test:
        rand_1=rd.randint(0,24)
        rand_2=rd.randint(0,24)
        rand_3=rd.randint(0,24)
        rand_4=rd.randint(0,24)
        self.assertEqual(distortion_test.structure.adjacency_matrix[rand_1,rand_2],
                         distortion_ref.structure.adjacency_matrix[rand_1,rand_2])
        self.assertEqual(distortion_test.structure.distance_matrix[rand_3,rand_4],
                         distortion_ref.structure.distance_matrix[rand_3,rand_4])
    
    def test_profiles(self):
        '''Test for trends in the computed profiles and for the specific convergence to 0 at the highest aggregate level.'''
        
        #check for similarity to reference 
        self.assertEqual(distortion_test.hindex,distortion_ref.hindex)
        self.assertEqual(distortion_test.ehindex,distortion_ref.ehindex)
        #check for convergence to 0
        for i in range(distortion_test.size):
            self.assertEqual(distortion_test.divergence_data[i].profile[distortion_test.size-1],0)
            
    def test_local_indices(self):
        '''Test the value of the distortion index for a random cell and compare to reference. '''
        #update dataframe
        distortion_test.update_data()
        #generate random integer 
        rand_index=rd.randint(0,24)
        #check existance and similarity
        self.assertEqual(distortion_test.dataframe.distortion_index[rand_index],
                         distortion_ref.dataframe.distortion_index[rand_index])
        

if __name__ == '__main__':
    unittest.main()




