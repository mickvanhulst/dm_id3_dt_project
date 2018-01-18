import numpy as np
import pandas as pd
import operator
import os
# Import classes
from TreeGraph import *
from decision_tree import ID3DecisionTree

def main():
    # Create path for visualization
    #load dataset
    df = pd.read_csv('./data/mushrooms.csv')
    # Optional to remove the 'odor' column as it is a dominant feature.
    df = df.drop('odor', axis=1)
    class_label = 'class'
    features = [x for x in df.columns if x != class_label]
    print(features)
    pred_column = 'classify'
    # Split data in train/test/validation (40, 30, 30).
    train_data, val_data, test_data = np.split(df.sample(frac=1), [int(.4*len(df)), int(.7*len(df))])
    
    #data = Data(df, features, [pred_column], test_size=(2/3))
    tree = ID3DecisionTree(train_data, val_data, features, class_label, pred_column
        , pruning_type='post')
    accuracy, classified_test_data = tree.classify(test_data)
    #print('----------------------- RESULT -----------------------')
    
    #print(tree.result)
    #print('The accuracy of the tree is: ' + str(accuracy))
    
    
    # To create a visualization we need to append Graphviz to our path
    path = 'C:/Program Files (x86)/Graphviz2.38/bin/'
    os.environ["PATH"] += os.pathsep + str(path)

    #tree.create_tree_graph()
    #tree.create_visualization_file()

if __name__ == '__main__':
    main()