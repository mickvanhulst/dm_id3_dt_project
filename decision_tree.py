import numpy as np
import pandas as pd
from TreeGraph import *
from post_pruning import TreePostPruner
import operator
#from PIL import Image

# Turn off annoying warning (Link: http://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas)
pd.options.mode.chained_assignment = None

class ID3DecisionTree(object):

    def __init__(self, train_data, val_data, features, class_col, pred_label, 
                    pruning_type='pre', max_depth=None, conf_interval=.75):
        '''
        This function initializes the ID3 Decision Tree algorithm
        '''
        self.train_data = train_data
        self.features = features
        self.class_col = class_col
        self.pred_label = pred_label
        self.pruning_type = pruning_type

        # Verify if the user's input of max-depth is positive.
        if (max_depth is not None) and (max_depth < 1):
            print('Illigal max-depth detected, changed to 1`.')
            self.max_depth = 1
        else:
            self.max_depth = max_depth        
        
        # Recursively build the tree. 
        self.result = self.__build_tree(train_data, self.features)

        # Apply post-pruning
        if self.pruning_type == 'post':
            pruner = TreePostPruner(self.result, val_data, conf_interval)
            self.result = pruner.get_tree()

    def __apply_entropy_formula(self, freq, data):
        '''
        This function applies the Entropy formula to a specific target attribute.
        '''
        return -(freq[1]/len(data)) * np.log2(freq[1]/len(data))
    
    def __entropy(self, data):
        '''
        This function sums and returns the entropy for each of the values in the target attribute.
        '''
        val_freq = data[self.class_col].value_counts()
        entropy = sum([self.__apply_entropy_formula(freq, data) for freq in val_freq.iteritems()])
        return entropy

    def __information_gain(self, data, feature):
        '''
        This function calculates the information gain for a specific feature by applying
        the corresponding formula. Based on the information gain of the leftover
        features, the algorithm decides what feature we should split on or
        if we should prune (in the case of pre-pruning).
        '''
        entropy_sub = 0.0
        val_freq = data[feature].value_counts()
        
        for val in val_freq.iteritems():
            val_prob = val[1] /  len(data.index)
            data_subset = data[data[feature] == val[0]]
            entropy_sub += val_prob * self.__entropy(data_subset)

        return (self.__entropy(data) - entropy_sub)
    
    def __pruning(self, cnt, prev_information_gain, best_information_gain):
        '''
        This function checks whether we're using post- or pre-pruning and 
        decided whether or not we should prune. In the case of post-pruning,
        we only prune once the amount of classes is one. In the case of pre-pruning,
        we prune if the information gain hasn't improved by performing this
        next split.
        '''
        if self.pruning_type == 'post':
            if len(cnt) == 1:
                return True
            else: 
                return False
        else:
            if prev_information_gain >= best_information_gain:
                return True
            else:
                return False

    def __build_tree(self, data, features, default_class=None, prev_information_gain=-1, 
            tree_depth=0):
        '''
        This recursive function is applied to the mentioned features and results in a tree. Each recursion is called
        with a filtered list of features and dataset. Based on this leftover information the information gain is calculated.
        Based on the information gain and the mode that the user is applying (i.e. post-, pre-pruning and also the 
        maximal tree-depth) the algorithm decided what actions to take and whether or not to prune or not.

        Steps:
        1. Choose best feature to split on.
        2. Determine dominant class
        3. Decide whether to prune or not.
        4. Initialize a new dict (subtree).
        5. Create branch for each value of the chosen feature to split on and recursively call this function.
        '''
        
        gain_dict = {feature : self.__information_gain(data, feature) for feature in features}
        best_feat, best_information_gain = max(gain_dict.items(), key=operator.itemgetter(1))
        cnt = data.groupby(self.class_col).size()
        
        dominant_class = cnt[max(cnt) == cnt.values].index[0]
        stop = self.__pruning(cnt, prev_information_gain, best_information_gain)

        if stop or ((tree_depth == self.max_depth) and self.max_depth > 0): 
            return dominant_class
        else:
            prev_information_gain = best_information_gain

        result = {best_feat:{}}
        remaining_features = [i for i in features if i != best_feat]
        tree_depth += 1
        
        for attr_val in data[best_feat].unique():
            data_subset = data[data[best_feat] == attr_val]
            subtree = self.__build_tree(data_subset,
                        remaining_features,
                        default_class,
                        prev_information_gain, 
                        tree_depth)
        
            result[best_feat][attr_val] = subtree

        return result

    def classify(self, test_data):
        '''
        This function applies the classify_loop function to all the test data that the user has
        given as the input. The accuracy is calculated by comparing the predicted class
        to the actual class (i.e. n correctly predicted classes / total n of rows = accuracy).
        '''
        test_data[self.pred_label] = test_data.apply(self.__classify_loop, axis=1, args=(self.result, 'None_found'))
        
        accuracy = len(test_data[test_data[self.class_col] == 
            test_data[self.pred_label]].index) / len(test_data.index)

        return accuracy, test_data

    def __classify_loop(self, instance, tree, default=None):
        '''
        This function recursively goes through the tree with the input test data row. 
        If the instance is a dict, then we continue, if it's not a dict, then it's a leaf (i.e. the class.).
        '''
        attribute = list(tree.keys())[0]
        
        if instance[attribute] in tree[attribute].keys():
            result = tree[attribute][instance[attribute]]
            if isinstance(result, dict):
                return self.__classify_loop(instance, result)
            else:
                return result
        else:
            return default

    def create_tree_graph(self):
        '''
        This function initializes the tree graph class, which enables
        the user to create a visualization of the decision tree.
        '''
        self.tree_graph = TreeGraph(self.result, self.features)

    def create_visualization_file(self, filename='Tree'):
        '''
        This function uses the export_image function of the tree_graph class to
        export the visualization of the tree to a image in PNG format.
        '''
        self.tree_graph.export_image(filename, 'png')
