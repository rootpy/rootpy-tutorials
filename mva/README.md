# TMVA vs. scikit-learn

## Introduction

Most high-energy physicists use for machine learning the [TMVA](http://tmva.sourceforge.net/)
package that is integrated in [ROOT](http://root.cern.ch/).

Since [rootpy](http://rootpy.org/) makes it easy to convert ROOT TTrees to numpy arrays,
it is easily possible to use Python machine learning packages
such as [scikit-learn](http://scikit-learn.org/)

Here we show an example of running a similar boosted decision tree (BDT) training and evaluation
with TMVA and scikit-learn.


## Howto

* Execute `python create_sample.py` to generate the `sample.root` file, which contains
a `sample` TTree with some simulated signal and background events.
* Run `python tmva_train.py` to perform the BDT training with TMVA.
This will generate the `tmva_output.root` file and `weights` folder.
* Run `python tmva_read.py` ... TODO: doesn't work at the moment.
* Run `python sklearn_train.py` to perform the BDT training with scikit-learn.
This will generate the `sklearn_bdt.pickle` file.
* Run `python sklearn_plot.py` to illustrate the scikit-learn classification.

TODO: compare classification results from TMVA and scikit-learn.
One possibility would be to evaluate the classifiers on a 2D grid and save the images
(as ROOT TH2s or picke numpy arrays), then overplot the decision boundary contours.
Another possibility would be to time the training and evaluation to see who is faster.
 