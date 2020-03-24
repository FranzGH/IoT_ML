# https://github.com/suvoooo/Machine_Learning/blob/master/DecsTree/notebooks/Bank_Data_Analysis.ipynb

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

### dataset is biased more towards no .. 

sns.countplot(x = 'deposit', data=bank_df, palette='hls')
plt.xlabel('Deposit', fontsize=13)
plt.ylabel('Count', fontsize=12)
plt.show()
print("total number of no : ", len(bank_df[bank_df['deposit']=='no']))
print("total number of yes : ", len(bank_df[bank_df['deposit']=='yes']))

# poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
sns.countplot(x='poutcome', data=bank_df, palette='summer_r')
plt.xlabel('Poutcome', fontsize=13)
plt.ylabel('Count', fontsize=12)

# depo is the target variable
depo = ['yes', 'no']
sel_list = ['age', 'balance']

yes_depo = bank_df[bank_df['deposit']=='yes']
no_depo = bank_df[bank_df['deposit']=='no']

h_age_yes = list(yes_depo['age'])
h_age_no = list(no_depo['age'])

h_bal_yes = list(yes_depo['balance']/1000)
h_bal_no = list(no_depo['balance']/1000)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)

plt.hist([h_age_no, h_age_yes], density=True, label=depo, alpha=0.6, bins=range(5, 95), linewidth=1.3)
plt.xlabel('Age', fontsize=12)
plt.legend(fontsize=12)

plt.subplot(1, 2, 2)
plt.hist([h_bal_yes, h_bal_no], density=True, label=depo, alpha=0.6,bins=range(-11, 30), linewidth=1.3)
plt.xlabel(r'Balance $\left(\times 1000\right)$', fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# Use Seaborn Categorical Plot which creates figure-level interface for drawing categorical plots onto a FacetGrid.
# plt.figure(figsize=(10, 5))
sns.set(font_scale=1.2)

g1 = sns.catplot(x='education', y='balance', hue='deposit', data=bank_df, kind='boxen')
plt.yscale('log')
plt.xlabel('Education')
g1.set_xticklabels(rotation=45)
g2 = sns.catplot(x='marital', hue='deposit', data=bank_df, kind='count')
plt.xlabel('Marital')
g3 = sns.catplot(x='housing', kind='count', hue='deposit', data=bank_df)

plt.figure(figsize=(15, 7))
# sns.set(font_scale=1.2)
g4 = sns.catplot(x='job', y='balance', hue='deposit', kind='boxen', data=bank_df, 
                height=5, aspect=3)
g4.set_xticklabels(rotation=60, fontsize=14)

plt.xlabel('Jobs', fontsize=13)
plt.ylabel('Balance', fontsize=13)
plt.yscale('log')
plt.ylim(1, 9e4)
plt.show()

# Converting Categorical Variables to Dummy Variables
#Since the dataframe contains many categorical variables, need to convert them to dummy variabels before we can use them for classification task. Also, as mentioned in the data description 'duration' feature highly affects the target. This feature should be included for benchmark purposes and should be discarded if the intention is to have realistic predictive model.
#Also, I have no clue what the hell actually 'poutcome' feature represents, so for the sake of simplicity, we will drop it too. From the countplot, most of the data falls under the 'unknown' category, so possibly reasonable to drop this variable.

bank_df = bank_df.drop(['duration', 'poutcome'], axis=1)
print ("now column names after dropping duration: ", list(bank_df.columns))

print ("data types of the features: \n", bank_df.dtypes) # Find the categorical variables

#### Check the Correlation Matrix of the Numerical Variables 

numeric_bank_df = bank_df.select_dtypes(exclude="object")
# categorical_df = df.select_dtypes(include="object")

corr_numeric = numeric_bank_df.corr()


sns.heatmap(corr_numeric, cbar=True, cmap="RdBu_r")
plt.title("Correlation Matrix", fontsize=16,)
plt.show() # very low correlation among the numeric variables, i.e. they all play important results

#### Finally convert the categorical variabels ('object type') to dummy variables
# Not so sure what exactly the 'poutcome' does. 

cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month']


bank_df_dummies = pd.get_dummies(bank_df, columns=cat_vars)

print ("check the column names: ", bank_df_dummies.columns.tolist())
print ("\n")
print ("total number of columns: ", len(bank_df_dummies.columns.tolist()))

# Since Deposit is the label so we can now prepare the training and test data for further analysis

## Turn yes/no in deposit to 1/0 

print (bank_df_dummies['deposit'].value_counts())

bank_df_dummies['deposit'] = bank_df_dummies['deposit'].map({'yes':1, 'no': 0})

labels = bank_df_dummies[['deposit']]
print ("check labels: ", labels.head(3))

features = bank_df_dummies.drop(['deposit'], axis=1)

print ("features data type: ", features.dtypes)

col_names_list = list(features.columns.values)
print ("features in a list: ", col_names_list)
print ("\n")
print ("number of features: ", len(col_names_list))

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
                special_characters=True,feature_names = col_names_list, class_names=['0','1'])


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_png('Bank_DecsT.png',)
# graph.set_size('"300, 180!"')
# graph.write_png('resized_tree.png')
Image(graph.create_png())

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

##########
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


# References
# https://victorzhou.com/blog/information-gain/ 
# https://blog.quantinsti.com/gini-index/ (example with binary features)
# https://towardsdatascience.com/entropy-how-decision-trees-make-decisions-2946b9c18c8 