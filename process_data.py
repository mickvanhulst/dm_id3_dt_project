#import packages
import numpy as np
import pandas as pd

class Data(object):

    '''
    This class can be used to process the data so it can be used with several algorithms. 
    Once this class is  initialized it will prepare the dataset. 
    - 'data' consists of a Pandas dataframe which represents your dataset
    - 'Features' represents a list of columns.
    - 'cols_to_init' a list of string values which are initialized as new columns (values will be 
    set to numpy's version of 'not a number'/NaN)
    - 'test_size' size of the test dataset (train_data will be determined by calculating 1-test_size)
    - 'class_col' represents the column which will be used (optional, only used in supervised learning)
    '''

    def __init__(self, data, features, cols_to_init, test_size=0.6, class_col=None):
        self.data = data
        self.test_size = test_size
        self.features = features
        #self.class_col = class_col
        self.cols_to_init = cols_to_init
        self.train_data, self.test_data = self.__process()

    def __process(self):
        for col in self.cols_to_init:
            self.data[col] = np.nan

        # OPTIMIZE THIS FUNCTION. BETTER RANDOM SAMPLING IS PROBABLY BETTER.
        test_data = self.data.sample(frac=(self.test_size))
        train_data = self.data[~self.data.index.isin(test_data.index.values)]

        return train_data, test_data