from root_numpy import root2rec
import numpy as np
import pylab as pl
import pickle

# read the sample
sample = root2rec('sample.root')
y = sample['label']
X = np.vstack([sample[var] for var in ['a', 'b']]).T

with open('sklearn_bdt.pickle', 'r') as f:
    bdt = pickle.load(f)

plot_colors = "br"
plot_step = 0.02
class_names = "AB"

pl.figure(figsize=(10, 5))

# Plot the decision boundaries
pl.subplot(121)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = pl.contourf(xx, yy, Z, cmap=pl.cm.Paired)
pl.axis("tight")

# Plot the training points
for i, n, c in zip(range(2), class_names, plot_colors):
    idx = np.where(y == i)
    pl.scatter(X[idx, 0], X[idx, 1],
               c=c, cmap=pl.cm.Paired,
               label="Class %s" % n,
               s=3,
               linewidth=0)
pl.xlim(x_min, x_max)
pl.ylim(y_min, y_max)
pl.legend(loc='upper right')
pl.xlabel("Decision Boundary")

# Plot the two-class decision scores
twoclass_output = bdt.decision_function(X)
plot_range = (twoclass_output.min(), twoclass_output.max())
pl.subplot(122)
for i, n, c in zip(range(2), class_names, plot_colors):
    pl.hist(twoclass_output[y == i],
            bins=20,
            range=plot_range,
            facecolor=c,
            label='Class %s' % n,
            alpha=.5)
x1, x2, y1, y2 = pl.axis()
pl.axis((x1, x2, y1, y2 * 1.2))
pl.legend(loc='upper right')
pl.ylabel('Samples')
pl.xlabel('Decision Scores')

pl.subplots_adjust(wspace=0.25)
pl.show()
