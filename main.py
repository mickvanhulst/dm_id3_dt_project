import numpy as np
import pandas as pd
import operator
import os
# Import classes
from TreeGraph import *
from decision_tree import ID3DecisionTree

def main():
    '''
    This function is the main function of the algorithm. The algorithm uses the following inputs from the user:
    1. Dataset
    2. If there's a dominant feature, then we can decide to remove the feature (in this case 'odor' is dominant)
    3. The name of the label which has to be used as the class label
    4. The features/columns of the datasets which we use to build the tree.

    Using these inputs, the tree is created and the test set is classified. Afterwards this function
    returns the resulting tree in the form of a dictionary and the accuracy. The user will also find a 
    PNG file in the same folder as this file which is a drawing of the generated tree. To create this visualization
    one needs to follow the steps in the README.md file.
    '''
    df = pd.read_csv('./data/mushrooms.csv')
    df = df.drop('odor', axis=1)
    class_label = 'class'
    features = [x for x in df.columns if x != class_label]
    pred_column = 'classify'
    train_data, val_data, test_data = np.split(df.sample(frac=1), [int(.4*len(df)), int(.7*len(df))])
    
    tree = ID3DecisionTree(train_data, val_data, features, class_label, pred_column
        , pruning_type='pre')
    accuracy, classified_test_data = tree.classify(test_data)
    
    print('----------------------- RESULT -----------------------')
    print(tree.result)
    print('The accuracy of the tree is: ' + str(accuracy))
        
    # To create a visualization we need to append Graphviz to our path
    path = 'C:/Program Files (x86)/Graphviz2.38/bin/'
    os.environ["PATH"] += os.pathsep + str(path)
    tree.create_tree_graph()
    tree.create_visualization_file()

if __name__ == '__main__':
    main()