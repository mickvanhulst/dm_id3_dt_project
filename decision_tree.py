import numpy as np
import pandas as pd
from TreeGraph import *
import operator
#from PIL import Image

# Turn off annoying warning (Link: http://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas)
pd.options.mode.chained_assignment = None

class ID3DecisionTree(object):

    def __init__(self, train_data, val_data ,features, class_col, pred_label, 
                    pruning_type='pre', max_depth=None):
        self.train_data = train_data
        self.val_data = val_data
        self.features = features
        self.class_col = class_col
        self.pred_label = pred_label
        self.pruning_type = pruning_type

        # Verify if max-depth is positive, else change it to one
        if (max_depth is not None) and (max_depth < 1):
            print('Illigal max-depth detected, changed to 1`.')
            self.max_depth = 1
        else:
            self.max_depth = max_depth        
        
        # Recursively build the tree. 
        self.result = self.__build_tree(train_data, self.features)

        # Check if we're applying post-pruning, if so, use YOERI's function
        if self.pruning_type == 'post':
            print('call function')

    def create_tree_graph(self):
        self.tree_graph = TreeGraph(self.result, self.features)

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
    def __pruning(self, cnt, prev_information_gain, best_information_gain):
        # returns true if stop, returns false if continue.
        # Test if node improves impurity measure of previous split.
        if self.pruning_type == 'post':
            if len(cnt) == 1:
                return True
            else: 
                return False
        else:
            # Pre-pruning
            if prev_information_gain >= best_information_gain:
                # If we want to stop pruning if IG doesn't improve.
                return True
            else:
                return False

    def __build_tree(self, data, features, default_class=None, prev_information_gain=-1, 
            tree_depth=0):
        '''
        General:
            * Error estimation SSE.
            * Process schema for different inputs user (user can input error method etc.). 
        Pre-prune: Grow the tree until the information gain does not increase anymore, then stop and return highest class.
        Several options for pre-pruning:
            * Stop when unique amount of classes is one. *
            * Stop after a certain tree depth. *
            * Stop when information gain doesn't increase at a possible split. *
        post-prune: keep growing until only one type of class remains. Then prune afterwards to decrease the error rate.
        Several options for post-pruning:
            * Stop when unique amount of classes is one. *
            * Stop after a certain tree depth.*
            * Implement post-pruning after growing full tree.
            * Chi-squared test.
        '''
        # Choose best feature to split on.
        gain_dict = {feature : self.__information_gain(data, feature) for feature in features}
        best_feat, best_information_gain = max(gain_dict.items(), key=operator.itemgetter(1))
        # Determine occurances per class
        cnt = data.groupby(self.class_col).size()
        
        # Set class which occurs most
        dominant_class = cnt[max(cnt) == cnt.values].index[0]
        stop = self.__pruning(cnt, prev_information_gain, best_information_gain)

        if stop or ((tree_depth == self.max_depth) and self.max_depth > 0): 
            # Return dominant class if we have to stop because of pruning
            # or we have to stop because the max depth is reached.
            return dominant_class
        else:
            prev_information_gain = best_information_gain

        # Init empty tree
        result = {best_feat:{}}
        remaining_features = [i for i in features if i != best_feat]
        
        # Increase tree depth.
        tree_depth += 1
        
        # Create branch for each value in best feature.
        for attr_val in data[best_feat].unique():
            data_subset = data[data[best_feat] == attr_val]
            subtree = self.__build_tree(data_subset,
                        remaining_features,
                        default_class,
                        prev_information_gain, 
                        tree_depth)
        
            result[best_feat][attr_val] = subtree

        return result

    def create_visualization_file(self, filename='Tree', format='png'):
        self.tree_graph.export_image(filename, format)

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