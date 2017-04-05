# GitHub Repository for Homework 2

### COMS 4995: Applied Machine Learning, Spring 2017



##### Group Members:

- [Daryl Kang](https://github.com/darylkang) (dk2896)
- [Weiwei Wang](https://github.com/WeiweiVivianWang) (ww2439)

##### Files:



- [data](./data)

*****

#### Model Performance

Test Set R<sup>2</sup>: 0.538


#### Overview

This model is an implementation of Ridge Regression to predict the monthly rent of an apartment in New York City. The data was obtained from the <a href="https://www.census.gov/housing/nychvs/data/2014/nychvs14.html">2014 New York City Housing and Vacancy Survey (NYCHVS)</a>, where only features relevant to pricing an apartment that is not currently rented were selected. This model makes the simplifying assumption that the market rate is static; i.e., the rent for a new tenant will be the same as that for a current tenant.

#### Data Preprocessing

The data was filtered to remove all observations where the response variable (`UF17`) is missing. Then, the data was randomly split into a training and test subset. Missing values in the data were then recoded as `NaN`s. Next, all features that were specific to the tenant—or were leaking information about the response variable—were removed. This reduced the number of features from 196 to 73. Missing values were then imputed based on the median for continuous values and the mode for categorical values. Finally, the categorical features were encoded as one-hot.

#### Model Selection

The model of choice for this assignment is the Ridge Regression because the inclusion of a penalty term `alpha` is able to reduce overfitting. Grid search was performed to determine the ideal value for `alpha`, which turns out to be `0.01` in this implementation. The model was also cross-validated using 10 iterations of shuffle-split cross-validation on the training set. Shuffle-split was chosen because it randomly shuffles the data before splitting it into the training and validation subsets. This led to an accuracy of 53.8% on the test set. The script `test_rent.py` ensures that the model is at least 50% accurate.
