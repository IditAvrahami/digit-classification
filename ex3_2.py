# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:07:16 2022

@author: valeri
"""

"""
==============================================================
Restricted Boltzmann Machine features for digit classification
==============================================================

For greyscale image data where pixel values can be interpreted as degrees of
blackness on a white background, like handwritten digit recognition, the
Bernoulli Restricted Boltzmann machine model (:class:`BernoulliRBM
<sklearn.neural_network.BernoulliRBM>`) can perform effective non-linear
feature extraction.

"""

# Authors: Yann N. Dauphin, Vlad Niculae, Gabriel Synnaeve
# License: BSD

# %%
# Generate data
# -------------
#
# In order to learn good latent representations from a small dataset, we
# artificially generate more labeled data by perturbing the training data with
# linear shifts of 1 pixel in each direction.

import numpy as np

from scipy.ndimage import convolve

from sklearn import datasets
from sklearn.preprocessing import minmax_scale

from sklearn.model_selection import train_test_split


def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
    ]

    def shift(x, w):
        return convolve(x.reshape((8, 8)), mode="constant", weights=w).ravel()

    X = np.concatenate(
        [X] + [np.apply_along_axis(shift, 1, X, vector) for vector in direction_vectors]
    )
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y


X, y = datasets.load_digits(return_X_y=True)
X = np.asarray(X, "float32")
X, Y = nudge_dataset(X, y)
X = minmax_scale(X, feature_range=(0, 1))  # 0-1 scaling

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# %%
# Models definition
# -----------------
#
# We build a classification pipeline with a BernoulliRBM feature extractor and
# a :class:`LogisticRegression <sklearn.linear_model.LogisticRegression>`
# classifier.

from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

logistic = linear_model.LogisticRegression(solver="newton-cg", tol=1)
rbm = BernoulliRBM(random_state=0, verbose=True)

rbm_features_classifier = Pipeline(steps=[("rbm", rbm), ("logistic", logistic)])

# %%
# Training
# --------
#
# The hyperparameters of the entire model (learning rate, hidden layer size,
# regularization) were optimized by grid search, but the search is not
# reproduced here because of runtime constraints.

from sklearn.base import clone
from sklearn import metrics
from time import perf_counter
import matplotlib.pyplot as plt
results_arr = []

def components_test():    
    for i in (p**2 for p in range(2,21)):
        rbm.learning_rate = 0.06
        rbm.n_iter = 10

        # More components tend to give better prediction performance, but larger
        # fitting time
        rbm.n_components = i
        logistic.C = 6000

        # Training RBM-Logistic Pipeline
        t1_start = perf_counter()
        rbm_features_classifier.fit(X_train, Y_train)
        t1_stop = perf_counter()
        Y_pred = rbm_features_classifier.predict(X_test)
        avg = metrics.precision_recall_fscore_support(Y_test, Y_pred,average='micro')
        results = [int(i),avg[0], t1_stop-t1_start]
        results_arr.append(results)
        plt.figure(figsize=(4.2, 4))
        squre = np.sqrt(i)

def dispaly_2d_plot(x,y,x_label,y_label,title,line,y_line=None):
    if line:
        plt.axhline(y = y_line, color = 'r', linestyle = '-')
        
    plt.plot(x, y, color="blue", linewidth=1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.suptitle(title, fontsize=16)
    plt.show()



#raw pixel :
raw_pixel_classifier = clone(logistic)
raw_pixel_classifier.C = 100.0
raw_pixel_classifier.fit(X_train, Y_train)
Y_pred = raw_pixel_classifier.predict(X_test)
avg = metrics.precision_recall_fscore_support(Y_test, Y_pred,average='micro')
precision_raw_pixel = avg[0]


#gettign data from our results array
# Training the Logistic regression classifier directly on the pixel
components_test()
results_arr= np.array(results_arr)
num_of_comp = results_arr[:, 0] 
precision_avg = results_arr[:,1] 
time = results_arr[:, 2] 

dispaly_2d_plot(num_of_comp, precision_avg, "components number", "precision average", "precision average on components number ",True,precision_raw_pixel)
dispaly_2d_plot(num_of_comp, time, "components number", "precision average", "precision average on time per run ",False)


plt.figure(figsize=(4.2, 4))
#displaying 20*20 components extraction
for i, comp in enumerate(rbm.components_):
    plt.subplot(20, 20, i + 1)
    plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r, interpolation="nearest")
    plt.xticks(())
    plt.yticks(())
plt.suptitle("20*20 components extracted by RBM", fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
plt.show()


print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(rbm.transform(X_train)))
print(np.shape(rbm.intercept_hidden_))

