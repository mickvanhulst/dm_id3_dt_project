# ID3 Decision Tree classifier 
This is a project created by the students Yoeri van Bruchem and Mick van Hulst for the course Data Mining at the Radboud University.

# How to
1. Clone the repo on your local pc.
2. Open a terminal and run: 'pip3 install -r requirements.txt' (pip3 is used for Python 3)
3. (optional*)Install [Graphviz](https://graphviz.gitlab.io/download/).
4. (optional) The algorithm uses pre-pruning by default and does not apply a maximum tree depth. The user can change the following parameters:
- pruning_type: can be set to either 'pre' or 'post' for either pre- or post-pruning.
- max_depth: can be set to establish a maximum tree depth (has to be higher than zero). This may only be a natural number. Non-natural numbers are automatically converted. Use 'None' if you do not want
to apply a max tree depth (i.e. this is the default value for the class, so you may also choose to remove the parameter as an input).
4. Run 'main.py' using Python3. 

# Important
*This is if you'd like the algorithm to chart the tree itself. This requires some manual input in the code by the user. The user has to add the path to his GraphViz folder in the code (see variable 'path' in main.py). Keep in mind
that the function to draw the tree is commented by default.