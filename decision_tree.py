import numpy as np
import pandas as pd
#import pydot_ng
import operator
#from PIL import Image

# Import class
from process_data import Data 

# Turn off annoying warning (Link: http://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas)
pd.options.mode.chained_assignment = None

class id3_decision_tree(object):

    def __init__(self, data, classify_label):
        self.data = data.data
        self.test_data = data.test_data
        self.train_data = data.train_data
        self.features = data.features
        self.class_col = data.class_col
        self.classify_label = classify_label
        self.accuracy = None
        self.pruning_type = 'post'
        self.result = self.__build_tree(self.train_data, self.features)

    def __entropy_formula(self, freq, data):
        # -probability * log(probability)
        return -(freq[1]/len(data)) * np.log2(freq[1]/len(data))
    
    def __entropy(self, data):
        # Calculate the frequency of each of the values in the target attr
        val_freq = data[self.class_col].value_counts()

        # Calculate the entropy of the data for the target attribute
        entropy = sum([self.__entropy_formula(freq, data) for freq in val_freq.iteritems()])
            
        return entropy

    def __information_gain(self, data, feature):
        # Init entropy of subset
        entropy_sub = 0.0

        # Calculates the frequency of each of the values in the target attribute
        val_freq = data[feature].value_counts()
       
        # for each value in subset multiply probability with entropy of that subset
        for val in val_freq.iteritems():
            val_prob = val[1] /  len(data.index)
            data_subset = data[data[feature] == val[0]]
            entropy_sub += val_prob * self.__entropy(data_subset)

        return (self.__entropy(data) - entropy_sub)

    def __build_tree(self, data, features, default_class=None, prev_information_gain=-1):
        '''
        pre-prune: Grow the tree until the information gain does not increase anymore, then stop and return highest class.

        post-prune keep growing until only one type of class remains. Then prune afterwards to decrease the error rate.

        To do:
        - Use chi squared test
        - Add postpruning step
        - Create an upper function which runs build_tree so that we can optimize afterwards (postpruning)
        '''
        # Choose best feature to split on.
        gain_dict = {feature : self.__information_gain(data, feature) for feature in features}
        best_feat, best_information_gain = max(gain_dict.items(), key=operator.itemgetter(1))
         
        # Determine occurances per class
        cnt = data.groupby(self.class_col).size()
        print(len(cnt))
        # Set class which occurs most
        dominant_class = cnt[max(cnt) == cnt.values].index[0]
        
        if self.pruning_type == 'pre': 
            # Test if node improves impurity measure of previous split.
            if prev_information_gain >= best_information_gain:
                # No improvement, so we quit and return the class which occurs most in our current dataset.
                return dominant_class
            else:
                prev_information_gain = best_information_gain
        elif self.pruning_type == 'post':
            # If amount of classes in remaining dataset is one, return class.
            if len(cnt) == 1:
               # No improvement, so we quit and return the class which occurs most in our current dataset.
               return dominant_class 
        
        # Init empty tree
        result = {best_feat:{}}
        remaining_features = [i for i in features if i != best_feat]
        print(len(remaining_features))

        # Create branch for each value in best feature.
        for attr_val in data[best_feat].unique():
            data_subset = data[data[best_feat] == attr_val]
            subtree = self.__build_tree(data_subset,
                        remaining_features,
                        default_class,
                        prev_information_gain)

            result[best_feat][attr_val] = subtree

        return result

    def classify(self):
        #Classify
        self.test_data[self.classify_label] = self.test_data.apply(self.__classify_loop, axis=1, args=(self.result, 'None_found'))
        
        # Calculate accuracy
        self.accuracy = len(self.test_data[self.test_data[self.class_col] == 
            self.test_data[self.classify_label]].index) / len(self.test_data.index)

    def __classify_loop(self, instance, tree, default=None):
        '''
        Recursively checks if the result is a dict or not. If it's not a dict, then it's a leaf. 
        
        '''
        attribute = list(tree.keys())[0]
        
        if instance[attribute] in tree[attribute].keys():
            result = tree[attribute][instance[attribute]]
            if isinstance(result, dict): # if it's a dict, then this is a tree.
                return self.__classify_loop(instance, result)
            else:
                return result # this is a label
        else:
            return default