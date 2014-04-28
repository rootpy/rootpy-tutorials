from root_numpy import root2rec, rec2array
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

# define the classifier
dt = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=150)
bdt = AdaBoostClassifier(dt,
    algorithm='SAMME',
    n_estimators=850,
    learning_rate=0.5)

# read the dataset
sample = root2rec('sample.root')
X = rec2array(sample, fields=['a', 'b'])
y = sample['label']

# train the classifier
bdt.fit(X, y)

# save the classifier
with open('sklearn_bdt.pickle', 'w') as f:
    pickle.dump(bdt, f)
