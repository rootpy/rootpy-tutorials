from root_numpy import root2rec
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

# read the sample
sample = root2rec('sample.root')
y = sample['label']

X = np.vstack([sample[var] for var in ['a', 'b']]).T

dt = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=150)
bdt = AdaBoostClassifier(dt,
        algorithm='SAMME',
        n_estimators=850,
        learning_rate=0.5)

bdt.fit(X, y)
with open('sklearn_bdt.pickle', 'w') as f:
    pickle.dump(bdt, f)
