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

    def __init__(self, train_data, features, class_col, pred_label, 
                    pruning_type='pre', stop_type='un_class', max_depth=-1):#data, classify_label):
        self.train_data = train_data
        self.features = features
        self.class_col = class_col
        self.pred_label = pred_label
        self.pruning_type = pruning_type
        self.stop_type = stop_type
        self.max_depth = max_depth
        self.result = self.__build_tree(train_data, self.features)

    def __entropy_formula(self, freq, data):
        # -probability * log(probability)
        return -(freq[1]/len(data)) * np.log2(freq[1]/len(data))
    
    def __entropy(self, data):
        # Calculate the frequency of each of the values in the target attr
        val_freq = data['class'].value_counts()
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
    def __pruning(self, cnt, prev_information_gain, best_information_gain):
        # returns true if stop, returns false if continue.
        # Test if node improves impurity measure of previous split.
        if self.pruning_type == 'post':
            if len(cnt) == 1:
                return True
        
        else:
            # Pre-pruning
            if prev_information_gain >= best_information_gain:
                # No improvement, dso we quit and return the class which occurs most in our current dataset.
                return True
            else:
                prev_information_gain = best_information_gain

    def __build_tree(self, data, features, default_class=None, prev_information_gain=-1):
        '''
        Pre-prune: Grow the tree until the information gain does not increase anymore, then stop and return highest class.
        Several options for pre-pruning:
            * Stop when unique amount of classes is one.
            * Stop after a certain tree depth.
            * Stop when information gain doesn't increase at a possible split.
            * Chi-squared test.
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
        
        # Set class which occurs most
        dominant_class = cnt[max(cnt) == cnt.values].index[0]
        stop = self.__pruning(cnt, prev_information_gain, best_information_gain)
    
        if stop: 
            return dominant_class
        
        # Init empty tree
        result = {best_feat:{}}
        remaining_features = [i for i in features if i != best_feat]
        
        # Create branch for each value in best feature.
        for attr_val in data[best_feat].unique():
            data_subset = data[data[best_feat] == attr_val]
            subtree = self.__build_tree(data_subset,
                        remaining_features,
                        default_class,
                        prev_information_gain)

            result[best_feat][attr_val] = subtree

        return result

    def classify(self, test_data):
        #Classify
        test_data[self.pred_label] = test_data.apply(self.__classify_loop, axis=1, args=(self.result, 'None_found'))
        
        # Calculate accuracy
        accuracy = len(test_data[test_data[self.class_col] == 
            test_data[self.pred_label]].index) / len(test_data.index)

        return accuracy, test_data

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