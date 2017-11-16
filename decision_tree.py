import numpy as np
import pandas as pd
#import pydot_ng
import operator
from PIL import Image

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

    def __build_tree(self, data, features, default_class=None):
        # Determine occurances per class 
        cnt = data.groupby(self.class_col).size()

        if len(cnt) == 1:
            # If len of cnt is one, one class remains (leaf).
            return cnt.index[0]
        elif data.empty or (len(features) == 0):
            # Empty dataset and/or no feature names to split on, return default class
            return default_class
        else:
            # Set default class to majority value
            default_class = cnt[max(cnt) == cnt.values].index[0]

            # Choose best feature to split on.
            gain_dict = {feature : self.__information_gain(data, feature) for feature in features}
            best_feat = max(gain_dict.items(), key=operator.itemgetter(1))[0]

            # Init empty tree
            result = {best_feat:{}}
            remaining_features = [i for i in features if i != best_feat]

            # Create branch for each value in best feature.
            for attr_val in data[best_feat].unique():
                data_subset = data[data[best_feat] == attr_val]
                subtree = self.__build_tree(data_subset,
                            remaining_features,
                            default_class)

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
    ## ------------------------ VISUALIZATION ------------------------------
    def __walk_dictionary(self, graph, dictionary, parent_node=None):
        '''
        Recursive plotting function for the decision tree stored as a dictionary
        '''

        for k in dictionary.keys():

            if parent_node is not None:

                from_name = parent_node.get_name().replace("\"", "") + '_' + str(k)
                from_label = str(k)

                node_from = pydot_ng.Node(from_name, label=from_label)
                graph.add_node(node_from)

                graph.add_edge( pydot_ng.Edge(parent_node, node_from) )

                if isinstance(dictionary[k], dict): # if interim node


                    self.__walk_dictionary(graph, dictionary[k], node_from)

                else: # if leaf node
                    to_name = str(k) + '_' + str(dictionary[k]) # unique name
                    to_label = str(dictionary[k])

                    node_to = pydot_ng.Node(to_name, label=to_label, shape='box')
                    graph.add_node(node_to)

                    graph.add_edge(pydot_ng.Edge(node_from, node_to))

            else:

                from_name =  str(k)
                from_label = str(k)

                node_from = pydot_ng.Node(from_name, label=from_label)
                self.__walk_dictionary(graph, dictionary[k], node_from)


    def draw_tree(self, image_name, open_image=False):
        # draw tree
        graph = pydot_ng.Dot(graph_type='graph')
        self.__walk_dictionary(graph, self.result)
        graph.write_png('./results/' + image_name +'.png')

        # Open image 
        if open_image:
            img = Image.open('./results/' + image_name +'.png')
            img.show()

def main():
    #load dataset
    df = pd.read_csv('./data/mushrooms.csv')

    # Optional to remove the 'odor' column as it is a dominant feature.
    #df = df.drop('odor', axis=1)
    class_label = 'class'
    features = [x for x in df.columns if x != class_label]
    classify_label = 'classify'

    # Create data class
    data = Data(df, features, [classify_label], (2/3), class_label)
    
    # build tree & classify data
    tree = id3_decision_tree(data, classify_label)
    tree.classify()
    print(tree.accuracy)
    print('----------------------- RESULT -----------------------')
    #print(tree.test_data)

    # Draw tree
    #tree.draw_tree('tree', True)

if __name__ == '__main__':
    main()