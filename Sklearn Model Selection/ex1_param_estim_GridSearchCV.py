# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

print(__doc__)

# Loading the Digits dataset
digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images) # 1797 images
X = digits.images.reshape((n_samples, -1)) # ahape: (1797, 64)
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print(f"# Tuning hyper-parameters for {score}")
    print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring=f'{score}_macro', cv=5
        # For integer/None inputs, if the estimator is a classifier and y is either binary or multiclass,
        # StratifiedKFold is used.
        # In all other cases, KFold is used.
    )
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score'] # For each candidate configuration
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print(f"{mean:.3f} (+/-{std * 2:.03f}) for {params}")
        # print("%0.3f (+/-%0.03f) for %r"  % (mean, std * 2, params))
        # http://zetcode.com/python/fstring/
        # https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule (within 2 standard deviations)
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    print(f'Best classifier\'s params: {clf.best_params_}')
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.