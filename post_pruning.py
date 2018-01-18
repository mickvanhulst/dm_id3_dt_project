import pandas as pd
import math
import scipy.stats as st
import operator
import numpy as np


class TreePostPruner:

    def __init__(self, original_tree, validation_data, conf_level):
        '''
        This function initializes the post-pruning algorithm that is used with the
        ID3 Decision Tree algorithm.
        '''
        self.original_tree = original_tree
        self.validation_data = validation_data
        self.conf_level = conf_level
        self.validation_tree = self.__construct_validation_tree(original_tree)
        self.__validate_tree()
        self.pruned_tree = self.__prune_tree(self.validation_tree)

    def __construct_validation_tree(self, original_tree):
        '''
        This function constructs the validation tree. The validation tree is based on tree which is given as an input
        to this class. The validation tree 
        '''
        validation_tree = dict()
        first_node = list(original_tree.keys())[0]
        validation_tree[first_node] = self.__add_subtree(
            original_tree[first_node])
        return validation_tree

    def __add_subtree(self, tree):
        subtree = dict()
        for key, value in tree.items():
            if isinstance(value, dict):
                subtree[key] = self.__add_subtree(value)
            else:
                subtree[key] = (value, 0, 0)
        return subtree

    def __validate_tree(self):
        indices = self.validation_data.index.values.tolist()

        for index in indices:
            row = pd.DataFrame(self.validation_data.loc[index]).T
            self.__validate(row, self.validation_tree)

    def __validate(self, instance, actual_tree):
        first_attribute = str(list(actual_tree.keys())[0])
        attribute_value = str(instance[first_attribute].iloc[0])
        child_attributes = list(actual_tree[first_attribute].keys())

        if attribute_value in child_attributes:
            child = actual_tree[first_attribute][attribute_value]

            if isinstance(child, dict):  # If the child is also a tree (subtree)
                actual_tree[first_attribute][attribute_value] = self.__validate(
                    instance, child)
            else:
                predicted_class = child[0]
                actual_class = instance['class'].iloc[0]
                actual_good_rate = actual_tree[first_attribute][attribute_value][1]
                actual_bad_rate = actual_tree[first_attribute][attribute_value][2]

                if actual_class == predicted_class:
                    actual_good_rate += 1
                else:
                    actual_bad_rate += 1

                actual_tree[first_attribute][attribute_value] = (
                    predicted_class, actual_good_rate, actual_bad_rate)

            return actual_tree

    def __prune_tree(self, tree):

        if isinstance(tree, dict):

            children = list()
            new_tree = dict()
            class_instances = dict()

            for key, value in tree.items():

                if isinstance(value, dict):
                    node = value
                    node_error, good_instances, number_of_instances, tree, preferable_class = self.__prune_tree(
                        node)
                    node_data = (number_of_instances, good_instances,
                                 node_error, key, node, preferable_class)
                    children.append(node_data)
                else:
                    leaf = value
                    good_instances = leaf[1]
                    bad_instances = leaf[2]
                    number_of_instances = good_instances + bad_instances

                    if number_of_instances == 0:
                        leaf_error = 0
                    else:
                        leaf_error = self.__error_estimation(
                            good_instances / number_of_instances, number_of_instances, self.conf_level)

                    pred_class = leaf[0]
                    leaf_data = (number_of_instances, good_instances, leaf_error, key, leaf,
                                 dict([(pred_class, number_of_instances)]))
                    children.append(leaf_data)

            total_number_of_instances = 0
            total_number_of_good_instances = 0

            for child in children:
                for key, value in child[5].items():
                    if key in class_instances:
                        class_instances[key] += value
                    else:
                        class_instances[key] = value

                total_number_of_instances += child[0]
                total_number_of_good_instances += child[1]

            error = 0

            for child in children:
                child_error = child[2]
                child_good_instances = child[1]

                error += child_good_instances / total_number_of_instances * child_error

            for child in children:

                if isinstance(child[4], dict):
                    if not (child[2] > error):
                        preferable_class = max(
                            class_instances.items(), key=operator.itemgetter(1))[0]
                        new_tree[child[3]] = (
                            preferable_class, child[1], child[0] - child[1])
                    else:
                        new_tree[child[3]] = child[4]

                else:
                    new_tree[child[3]] = child[4]

            return error, total_number_of_good_instances, total_number_of_instances, new_tree, class_instances

    def __error_estimation(self, f, n, confidence_level):
        """
        This function calculates the error estimation of a leaf.
        :param f: error on the training data.
        :param n: number of instances covered by the leaf.
        :param confidence_level: z from normal distribution
        :return: error estimation.
        """
        z = st.norm.ppf(
            confidence_level)  # Convert confidence level to z-score.

        # Calculate error estimate
        error_estimate = 0.0
        error_estimate += f
        error_estimate += (math.pow(z, 2) / (2 * n))
        error_estimate += (z * (math.sqrt((f / n) - (math.pow(f, 2) /
                                                     n) + (math.pow(z, 2) / (4 * math.pow(n, 2))))))
        error_estimate = error_estimate / (1 + (math.pow(z, 2) / n))

        return error_estimate  # Return error estimate.

    def __sanitize_tree(self, tree):

        sub_tree = dict()
        for key, value in tree.items():
            if isinstance(value, tuple):
                sub_tree[key] = value[0]
            else:
                sub_tree[key] = self.__sanitize_tree(value)
        return sub_tree

    def get_tree(self):
        return self.__sanitize_tree(self.validation_tree)
