# https://towardsdatascience.com/scikit-learn-decision-trees-explained-803f3812290d

# Importing Libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load data and store it into pandas DataFrame objects
iris = load_iris()
X = pd.DataFrame(iris.data[:, :], columns = iris.feature_names[:])
y = pd.DataFrame(iris.target, columns =["Species"])

# Defining and fitting a DecisionTreeClassifier instance
tree = DecisionTreeClassifier(max_depth = 2)
tree.fit(X,y)

# Visualize Decision Tree
from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO
dot_data  = StringIO() # in-memory buffer
# Creates dot file named tree.dot
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