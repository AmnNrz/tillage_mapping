# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: tillenv
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


cm = np.array([[89379,   486,   151],
       [  826, 56277,   136],
       [  413,   238, 26898]])
cm

# Plot confusion matrix for the best model on the test set
labels = ['0-15%', '16-30%', '>30%']
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
       xticklabels=labels, yticklabels=labels, xlabel='Predicted Class', ylabel='True Class')
plt.show()

# +
import numpy as np
import matplotlib.pyplot as plt

# Sample data for cm
# cm = np.array([[5, 2, 1], [1, 7, 1], [0, 2, 8]])

labels = ['0-15%', '16-30%', '>30%']
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=labels,
       yticklabels=labels,
       xlabel='Predicted Class',
       ylabel='True Class')

# Display the numbers on the matrix
thresh = cm.max() / 2.  # this is for the text color, white if the cell color is dark, and black otherwise.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

plt.show()

