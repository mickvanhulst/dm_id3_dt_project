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
    classify_label = 'classify'

    # Create data class
    data = Data(df, features, [classify_label], (2/3), class_label)
    
    # build tree & classify data
    tree = id3_decision_tree(data, classify_label)
    tree.classify()
    print(tree.accuracy)
    print('----------------------- RESULT -----------------------')
    print(tree.result)

    # Draw tree (name, object)
    #tree.draw_tree('tree', True)

if __name__ == '__main__':
    main()