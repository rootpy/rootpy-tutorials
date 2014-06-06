# TMVA vs. scikit-learn

## Introduction

Most high energy physicists use the [TMVA](http://tmva.sourceforge.net/)
package that is integrated in [ROOT](http://root.cern.ch/) for machine
learning.

Since [root_numpy](http://rootpy.github.io/root_numpy/) makes it easy to
convert ROOT TTrees to numpy arrays, it is easily possible to use Python
machine learning packages such as [scikit-learn](http://scikit-learn.org/)

Here we show an example of running a similar boosted decision tree (BDT)
training and evaluation with TMVA and scikit-learn.

## Usage

* Execute `python create_sample.py` to generate the `sample.root` file, which
  contains a `sample` TTree with some simulated signal and background events.
* Run `python tmva_train.py` to perform the BDT training with TMVA.  This will
  generate the `tmva_output.root` file and `weights` folder.
* Run `python tmva_read.py` ... TODO: doesn't work at the moment.
* Run `python sklearn_train.py` to perform the BDT training with scikit-learn.
  This will generate the `sklearn_bdt.pickle` file.
* Run `python sklearn_plot.py` to illustrate the scikit-learn classification.

## TODO

* Compare classification results from TMVA and scikit-learn.
* Evaluate the classifiers on a 2D grid and then plot the decision boundary
  contours.
* Compare training and evaluation times

Also see this [blog post](http://betatim.github.io/posts/matching-machine-learning/) 
comparing TMVA and scikit-learn.
