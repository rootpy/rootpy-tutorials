from root_numpy import root2rec
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

# read the sample
sample = root2rec('sample.root')
y = sample['label']
X = np.vstack(sample[['a', 'b']]).T

clf_params = {
    "base_estimator__min_samples_leaf": range(10, 100, 10),
    "n_estimators": range(1, 100, 20)}

cv = KFold(y.shape[0], n_folds=2)
clf = AdaBoostClassifier(DecisionTreeClassifier())
grid_search = GridSearchCV(clf, clf_params, cv=cv, n_jobs=-1)
grid_search.fit(X, y)

import pickle
with open('sklearn_bdt.pickle', 'w') as f:
    pickle.dump(grid_search.best_estimator_, f)
