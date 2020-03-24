# https://seaborn.pydata.org/generated/seaborn.countplot.html

# import the seaborn module
import seaborn as sns
sns.set(style="darkgrid")
'''
titanic = sns.load_dataset("titanic")
#reads online
ax = sns.countplot(x="class", data=titanic)
plotplt.show()
'''

import pandas as pd
# import the matplotlib module
import matplotlib.pyplot as plt
titanic = pd.read_csv("Titanic.csv",sep='\t')
sns.countplot('Pclass',data=titanic)
plt.show()

# Show value counts for two categorical variables:
ax = sns.countplot(x="Pclass", hue="Sex", data=titanic)
plt.show()

ax = sns.countplot(y="Pclass", hue="Sex", data=titanic)
plt.show()

ax = sns.countplot(x="Sex", hue="Survived", data=titanic, palette="Set3")
plt.show()

ax = sns.countplot(x="Survived", data=titanic,
                   facecolor=(0, 0, 0, 0),
                   linewidth=5,
                   edgecolor=sns.color_palette("dark", 3))
plt.show()

# Introducing a third category
g = sns.catplot(x="Pclass", hue="Sex", col="Survived",
                data=titanic, kind="count",
                height=4, aspect=.7)
plt.show()