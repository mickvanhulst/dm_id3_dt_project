import pydot


class TreeGraph:
    """ An tool to create and export graphs from built (decision) trees.

    Attributes:
        tree: A dictionary representing a tree.
        features: A list with all features in the tree.
    """

    def __init__(self, tree, features):
        """Return a TreeGraph object created with given input"""
        self.tree = tree
        self.features = features
        self.__graph = pydot.Dot(graph_type='graph', strict=False)
        self.__graph_node_count = 0
        self.__build_tree()

    def __build_tree(self):
        """Build a graph using the tree dictionary."""
        self.__create_nodes_and_edges(self.tree, self.features)

    def __create_nodes_and_edges(self, tree, features, parent_node=None, parent_branch=None):
        """Build a particular part of the tree. This can be a node or a leaf node."""

        for key, value in tree.items():

            if parent_node is None:                                     # First node in the tree.

                self.__graph_node_count += 1                                        # Create node.
                name = str(self.__graph_node_count) + str(key)
                node = pydot.Node(name=name, label=key)
                self.__graph.add_node(node)

                self.__create_nodes_and_edges(value, features, parent_node=name)    # Go to next item in tree.

            elif key in features:                                       # Not the first node

                self.__graph_node_count += 1                                        # Create node.
                name = str(self.__graph_node_count) + str(key)
                node = pydot.Node(name=name, label=key)
                self.__graph.add_node(node)

                edge = pydot.Edge(parent_node, name, label=parent_branch)           # Connect nodes to parent with edge.
                self.__graph.add_edge(edge)

                self.__create_nodes_and_edges(value, features, parent_node=name)    # Go te next item in tree.

            elif not isinstance(value, dict):                           # There is no child, this is a leaf node.

                self.__graph_node_count += 1                                        # Create leaf node.
                name = str(self.__graph_node_count) + str(value)
                node = pydot.Node(name, label=value)
                self.__graph.add_node(node)

                edge = pydot.Edge(parent_node, name, label=key)                     # Connect leaf to parent with edge.
                self.__graph.add_edge(edge)

            else:                                                       # Go to next item in tree.
                self.__create_nodes_and_edges(value, features, parent_node=parent_node, parent_branch=key)

    def export_image(self, file_name="Tree", file_format='png'):
        """Create an image file that displays the tree."""
        if file_format == 'png':
            self.__graph.write_png("{}.png".format(file_name))

