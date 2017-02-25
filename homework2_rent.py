#!/usr/bin/env python
"""A boilerplate script to be customized for data projects.

This script-level docstring will double as the description when the script is
called with the --help or -h option.

"""

import numpy  as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model    import Ridge


# Import Data from URL
url  = 'https://ndownloader.figshare.com/files/7586326'
data = pd.read_csv(url)



X = data.drop('uf17', axis=1)
y = data['uf17']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)




estimator = Ridge()




def score_rent(estimator=estimator):
  """Returns the R^2
  
  """
  return(np.mean(cross_val_score(estimator, X_train, y_train, cv=5)))

def predict_rent():
  """Returns your test data, the true labels and your predicted labels (all as numpy arrays)
  
  """
  pass









def main():
  pass

if __name__ == "__main__":
    main()