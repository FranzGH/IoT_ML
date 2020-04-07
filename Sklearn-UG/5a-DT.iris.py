# https://towardsdatascience.com/scikit-learn-decision-trees-explained-803f3812290d

# Importing Libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load data and store it into pandas DataFrame objects
# pylint: disable=no-member
iris = load_iris()
X = pd.DataFrame(iris.data[:, :], columns = iris.feature_names[:])
y = pd.DataFrame(iris.target, columns =["Species"])

'''
iris = pd.read_csv('datasets/iris.csv', skiprows=1)
X = pd.DataFrame(iris.values[:,:-1])
y = pd.DataFrame(iris.values[:,-1], columns =["Species")
'''

# Defining and fitting a DecisionTreeClassifier instance
tree = DecisionTreeClassifier(max_depth = 2)
tree.fit(X,y)

# Visualize Decision Tree
from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO
dot_data  = StringIO() # in-memory buffer
# Creates dot file (named tree.dot, actually I use a in-memory buffer)
export_graphviz(
            tree,
            out_file =  dot_data, #"myTreeName.dot",
            feature_names = list(X.columns),
            class_names = iris.target_names,
            filled = True,
            rounded = True)

import pydotplus
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('myTreeName.png',)

# Making a Prediction On a New Sample
sample_one_pred = int(tree.predict([[5, 5, 1, 3]]))
sample_two_pred = int(tree.predict([[5, 5, 2.6, 1.5]]))
print(f"The first sample most likely belongs a {iris.target_names[sample_one_pred]} flower.")
print(f"The second sample most likely belongs a {iris.target_names[sample_two_pred]} flower.")

# Closing remarks
# The real power of decision trees unfolds more so when cultivating many of them — while limiting the way they grow — and
# collecting their individual predictions to form a final conclusion.
# In other words, you grow a forest, and if your forest is random in nature,
# using the concept of bagging and with splitter = "random", we call this a Random Forest.

# splitter: This is how the decision tree searches the features for a split.
# The default value is set to “best”. That is, for each node, the algorithm considers all the features and chooses the best split.
# If you decide to set the splitter parameter to “random,” then a random subset of features will be considered.

# Regularization parameters
# Not limiting the growth of a decision tree may lead to over-fitting.