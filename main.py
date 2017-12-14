import numpy as np
import pandas as pd
import operator
import os
# Import classes
from process_data import Data 
from TreeGraph import *
from decision_tree import ID3DecisionTree

def setup_path():
    valid_input = False
    while not valid_input:
        text = input("Please enter the path to your Graphviz folder: ")
        if os.path.exists(os.path.dirname(text)):
            os.environ["PATH"] += os.pathsep + str(text)
            valid_input = True
        else:
            print("Path does not exit, try again.")

def main():
    # Create path for visualization
    #setup_path()
    #load dataset
    df = pd.read_csv('./data/mushrooms.csv')
    # Optional to remove the 'odor' column as it is a dominant feature.
    df = df.drop('odor', axis=1)
    class_label = 'class'
    features = [x for x in df.columns if x != class_label]
    pred_column = 'classify'

    # Split data in train/test/validation.
    data = Data(df, features, [pred_column], test_size=(2/3))
    tree = ID3DecisionTree(data.train_data, data.features, class_label, pred_column
        , pruning_type='post', stop_type='ig')
    accuracy, classified_test_data = tree.classify(data.test_data)
    print(accuracy)
    print('----------------------- RESULT -----------------------')
    tree.chi_squared_test(classified_test_data) 
    
    #print(tree.result)

    #tree.create_tree_graph()
    #tree.create_visualization_file()

if __name__ == '__main__':
    main()