import numpy as np
import pandas as pd
#import pydot_ng
import operator
#from PIL import Image
# Import classes
from process_data import Data 
from decision_tree import id3_decision_tree

def main():
    #load dataset
    df = pd.read_csv('./data/mushrooms.csv')

    # Optional to remove the 'odor' column as it is a dominant feature.
    df = df.drop('odor', axis=1)
    class_label = 'class'
    features = [x for x in df.columns if x != class_label]
    pred_column = 'classify'

    # Split data in train/test/validation.
    # MAKE SURE THAT WE NEED TO ENTER THE TRAIN DATA FIRST TO CREATE THE TREE
    # AFTERWARDS TRAIN THE TREE BY ENTERING THE TEST_DATA.
    
    # Create data class
    data = Data(df, features, [pred_column], test_size=(2/3))
    tree = id3_decision_tree(data.train_data, data.features, class_label, pred_column
        , pruning_type='post', stop_type='ig', max_depth=2)
    print(tree.result) 
    #accuracy, classified_test_data = tree.classify(data.test_data)
    #print(accuracy)
    #print('----------------------- RESULT -----------------------')
    #print(tree.result)
    #print(tree.result.keys())
    

    # Draw tree (name, object)
    #tree.draw_tree('tree', True)

if __name__ == '__main__':
    main()