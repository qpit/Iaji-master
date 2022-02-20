#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 10:36:34 2020

@author: jiedz
"""
#%%
#mports
import h5py
import numpy

#%%

class h5:
    """
    This class defines an h5 manager object object, which helps manipulating .h5 files, performing tasks
    such as copying the group structure of an .h5 file onto a new .h5 file, and making it easier to store data in an already existing dataset.
    """
    
    def __init__(self, file_path=None):
        if file_path:
            self.file = h5py.File(file_path, "r+")
        return
    
        
    def store(self, dataset, data):
        """
        This function stores an arbitrary object into a dataset of a .h5 file.
    
        Inputs
        ----------
        dataset: h5py Dataset object
            Dataset into which the data are stored 
        data: arbitrary object
            Object to be stored
    
        Outputs
        -------
        None.
    
        """
        name = dataset.name #dataset name
        group = dataset.parent #parent group
        del group[name]
        group.create_dataset(name=name, data=data, shape=numpy.asarray(data).shape, dtype=numpy.asarray(data).dtype)
            
    def copyStructure(self, destination_file, source_file=None):
        """
        This function copies the structure of an input .h5 file into a new 
        .h5 file, with empty datasets
        
        Inputs
        -----------------
        
        source_file: h5py.File object
            Source .h5 file whose structure has to be copied. If not specified 
            
        destination_file: h5py.File object    
            Destination .h5 file
        
        Outputs
        ------------------
        
        None
        """
        if not(source_file):
            if not(self.file):
                raise ValueError("Either the input 'source_file' or the internal property 'file' must be different than 'None'.")
            source_file = self.file
        if type(source_file)==h5py._hl.files.File:
            #Make sure that the source and destination files are open
            if not(source_file and destination_file):
                raise Exception("\nThe source file must be open.")
        #Run the function until a dataset is found
        if type(source_file) != h5py._hl.dataset.Dataset:
            keys = [key for key in source_file.keys()]
            for key in keys:
                destination_file.create_group(name=key)
                #Recursively create the group structure
                self.copyStructure(source_file=source_file[key], destination_file=destination_file[key])
        else:
            #Dataset found
            #Delete the last created group 
            parent_group = destination_file.parent
            dataset_name = source_file.name.split('/')[-1]
            del parent_group[dataset_name]
            #Replace it with a new empty dataset
            value = None
            parent_group.create_dataset(name=dataset_name, data=value, dtype=numpy.dtype(None)) 
     
    def dictionaryToFile(self, dictionary, destination_file):
        """
        this function save an input dictionary of numpy arrays into a .h5 file.
        
        INPUTS
        --------------
        dictionary : dict
            Input dictionary
        
        destination_file : h5py.File object    
            Destination .h5 file
        
        OUTPUTS
        --------------
        None
        """
        #create an empty output file
        if type(dictionary) is dict:
            keys_list = list(dictionary.keys())
            values_list = list(dictionary.values())
            #Create groups named as the keys, with content given by the items
            for j in range(len(keys_list)):
                key = keys_list[j]
                value = values_list[j]
                #Create group
                destination_file.create_group(name=key)
                #Recursively repeat the procedure
                self.dictionaryToFile(dictionary=value, destination_file=destination_file[key])
        else:
            #We found the bottom of the dictionary: save data into a dataset
            print(dictionary)
            destination_file.create_dataset(name='data', data=numpy.array([dictionary]))
        
            
            
        