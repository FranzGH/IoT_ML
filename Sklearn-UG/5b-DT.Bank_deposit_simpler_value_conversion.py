# https://github.com/suvoooo/Machine_Learning/blob/master/DecsTree/notebooks/Bank_Data_Analysis.ipynb

# References
# https://victorzhou.com/blog/information-gain/ 
# https://towardsdatascience.com/entropy-how-decision-trees-make-decisions-2946b9c18c8 
# https://blog.quantinsti.com/gini-index/ (example with binary features)


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

bank_df  = pd.read_csv('./bank.csv', delimiter=',')

print (bank_df.head(3))
print ("\n")
print ("list of attributes: ", list(bank_df.columns))
print ("\n")
print ("total number of attributes: ", len(list(bank_df.columns))-1)
print ("shape of datafrmae: ", bank_df.shape)



# Converting Categorical Variables to Dummy Variables
#Since the dataframe contains many categorical variables, need to convert them to dummy variabels before we can use them for classification task. Also, as mentioned in the data description 'duration' feature highly affects the target. This feature should be included for benchmark purposes and should be discarded if the intention is to have realistic predictive model.
#Also, I have no clue what the hell actually 'poutcome' feature represents, so for the sake of simplicity, we will drop it too. From the countplot, most of the data falls under the 'unknown' category, so possibly reasonable to drop this variable.

bank_df = bank_df.drop(['duration', 'poutcome', 'month'], axis=1)
print ("now column names after dropping duration: ", list(bank_df.columns))

print ("data types of the features: \n", bank_df.dtypes) # Find the categorical variables

#### Check the Correlation Matrix of the Numerical Variables 

numeric_bank_df = bank_df.select_dtypes(exclude="object")
categorical_bank_df = bank_df.select_dtypes(include="object")

#### Finally convert the categorical variabels ('object type') to dummy variables
# Not so sure what exactly the 'poutcome' does. 

#import the necessary module
from sklearn import preprocessing
# create the Labelencoder object
le = preprocessing.LabelEncoder()
print(categorical_bank_df.columns)


for cat in categorical_bank_df.columns:
    #convert the categorical columns into numeric
    bank_df[cat] = le.fit_transform(bank_df[cat])

labels = bank_df[['deposit']]
print ("check labels: ", labels.head(3))

features = bank_df.drop(['deposit'], axis=1)
print ("check features: ", features.head(3))

####
# Separate Train and Test Data
####

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, stratify=labels)

print ("number of training samples: ", len(X_train))
print ("number of test samples: ", len(y_test))

from sklearn.tree import DecisionTreeClassifier 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

pipe_steps = [('scaler', StandardScaler()), ('decsT', DecisionTreeClassifier())]

check_params = {'decsT__criterion':['gini', 'entropy'], 
               'decsT__max_depth': np.arange(3, 15)}

pipeline = Pipeline(pipe_steps)
print(pipeline)

### I love you so much

from tqdm import tqdm_notebook as tqdm

print ("start fitting the data")
import warnings
warnings.filterwarnings("ignore")


for cv in tqdm(range(3, 6)):
    create_grid = GridSearchCV(pipeline, param_grid=check_params, cv=cv)
    create_grid.fit(X_train, y_train)
    print("score for %d fold CV := %3.2f" %(cv, create_grid.score(X_test, y_test)))
    print ("!!!! best fit parameters from GridSearchCV !!!!")
    print (create_grid.best_params_)

print ("out o' the loop")

# Visualize the Tree

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus # pip install pydotplus


DecsTree = DecisionTreeClassifier(criterion='gini', max_depth=9)
DecsTree.fit(X_train, y_train)


dot_data  = StringIO()

export_graphviz(DecsTree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = features.columns, class_names=['0','1'])


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_png('Bank_DecsT2.png',)
# graph.set_size('"300, 180!"')
# graph.write_png('resized_tree.png')
Image(graph.create_png())


DecsTreeModel = DecisionTreeClassifier(criterion='gini', max_depth=7)
DecsTreeModel.fit(X_train, y_train)

train_score = DecsTreeModel.score(X_train, y_train)
print ("score on the training data: ", train_score)
print ("\n")

test_score = DecsTreeModel.score(X_test, y_test)
print ("score on the test data: ", test_score)

import sys
sys.exit()

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus


DecsTreeCheck = DecisionTreeClassifier(criterion='gini', max_depth=4)
DecsTree.fit(X_train, y_train)


dot_data  = StringIO()

export_graphviz(DecsTree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = col_names_list, class_names=['0','1'])


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_png('Bank_DecsT_check.png',)
Image(graph.create_png())

###
# Drop the Month Column and Try Again (Just for the Sake of the Decision Tree Visualization)
###
#bank_df_new = bank_df.drop(['duration', 'poutcome', 'month'], axis=1)
bank_df_new = bank_df.drop(['month'], axis=1)
print ("now column names after dropping duration: ", list(bank_df_new.columns))
print ("data types of the features: \n", bank_df_new.dtypes) # Find the categorical variables
cat_vars_new = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact']


bank_df_new_dummies = pd.get_dummies(bank_df_new, columns=cat_vars_new)
print ("check the column names: ", bank_df_new_dummies.columns.tolist())
print ("total number of columns: ", len(bank_df_new_dummies.columns.tolist()))

bank_df_new_dummies['deposit'] = bank_df_new_dummies['deposit'].map({'yes':1, 'no': 0})

print (bank_df_new_dummies['deposit'].value_counts())

labels_new = bank_df_new_dummies[['deposit']]
print ("check labels: ", labels_new.head(3))

features_new = bank_df_new_dummies.drop(['deposit'], axis=1)
col_names_list_new = list(features_new.columns.values)
print ("features in a list: ", col_names_list_new)


from sklearn.model_selection import train_test_split

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(features_new, labels_new, 
                                                                    test_size=0.25, stratify=labels_new)

print ("number of training samples: ", len(X_train_new))
print ("number of test samples: ", len(y_test_new))

pipe_steps = [('scaler', StandardScaler()), ('decsT', DecisionTreeClassifier())]

check_params = {'decsT__criterion':['gini', 'entropy'], 
               'decsT__max_depth': np.arange(3, 15)}

pipeline = Pipeline(pipe_steps)
print(pipeline)


print ("start fitting the data")
import warnings
warnings.filterwarnings("ignore")


for cv in tqdm(range(3, 6)):
    create_grid = GridSearchCV(pipeline, param_grid=check_params, cv=cv)
    create_grid.fit(X_train_new, y_train_new)
    print("score for %d fold CV := %3.2f" %(cv, create_grid.score(X_test_new, y_test_new)))
    print ("!!!! best fit parameters from GridSearchCV !!!!")
    print (create_grid.best_params_)

print ("out o' the loop")

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus


DecsTree = DecisionTreeClassifier(criterion='gini', max_depth=6)
DecsTree.fit(X_train_new, y_train_new)


dot_data  = StringIO() # in-memory buffer

export_graphviz(DecsTree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = col_names_list_new, class_names=['0','1'])


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_png('Bank_DecsT_new.png',)
# graph.set_size('"300, 180!"')
# graph.write_png('resized_tree.png')
Image(graph.create_png())

# Learn to Plot Feature Importance (Muller's Book)
# To summarize the workings of a complicated Decision Tree, most commonly we use _featureimportance. It is a number between 0 and 1, where 0 means the feature is not used at all to 1 implying "perfectly predicts the target". We will also check the score on train and test data.

DecsTreeModel = DecisionTreeClassifier(criterion='gini', max_depth=6)
DecsTreeModel.fit(X_train_new, y_train_new)


train_score = DecsTreeModel.score(X_train_new, y_train_new)
print ("score on the training data: ", train_score)
print ("\n")

test_score = DecsTreeModel.score(X_test_new, y_test_new)
print ("score on the test data: ", test_score)

n_features = len(col_names_list_new)

sns.set(style="whitegrid")

fig = plt.figure(figsize=(15, 11))
fig.tight_layout()
plt.bar(range(n_features), DecsTreeModel.feature_importances_, color="magenta", align="center", alpha=0.6)
plt.xticks(np.arange(n_features), col_names_list_new, rotation=80, fontsize=11)
plt.xlabel("Features", fontsize=14)
plt.ylabel("Feature Importance", fontsize=14)
plt.savefig("Feature_Importance.png", dpi=300, bbox_inches='tight')# xticks are not clipped with 'bbox'


