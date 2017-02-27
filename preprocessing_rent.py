#!/usr/bin/env python
"""preprocessing_rent.py

1) Fetch Data from URL
2) Filter Rows where y = NA
3) Split Data into X and y
4) Split Data into Training and Test Subsets
5) Recode Missing Values
6) Select Relevant Features
7) Impute Missing Values
8) Encode Categorical Features as One-Hot
9) Convert Data from Pandas DataFrame to NumPy Array

"""

import numpy  as np
import pandas as pd

from sklearn.model_selection import train_test_split

# Fetch Data Using URL
NYCHVS = pd.read_csv('https://ndownloader.figshare.com/files/7586326')

# Remove Observations with Null Response Variable
NYCHVS = NYCHVS[NYCHVS.uf17 != 99999]

# Split Data into X and y
X = NYCHVS.drop('uf17', axis=1)
y = NYCHVS['uf17']

# Define Parameters
random_state = 26
test_size = 0.25

# Split Data into Random Training and Test Subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                    random_state=random_state)

# Save the Raw Test Set
X_test_raw = X_test

# Recode Missing Values
def recode_NA(X):
    """Returns X with missing values coded as NaNs"""
    # Index of Features where NA = 8
    j = []
    j.extend(range(2, 27))
    j.extend(range(75, 79))
    j.extend([99])
    j.extend(range(101, 104))
    j.extend(range(106, 112))
    j.extend(range(116, 129))
    for i in j:
        X.iloc[:, i].replace(to_replace=8, value=np.nan, inplace=True)
    
    # Index of Features where NA = 3 and 8
    j = []
    j.extend(range(27, 30))
    j.extend([66, 69, 70, 71, 72])
    for i in j:
        X.iloc[:, i].replace(to_replace=[3, 8], value=np.nan, inplace=True)
    
    # Feature where NA = 4 and 8
    X.iloc[:, 100].replace(to_replace=[4, 8], value=np.nan, inplace=True)
    
    # Index of Features where NA = 5 and 8
    j = []
    j.extend(range(104, 106))
    for i in j:
        X.iloc[:, i].replace(to_replace=[5, 8], value=np.nan, inplace=True)
    
    # Feature where NA = 7 and 8
    X.iloc[:, 91].replace(to_replace=[7, 8], value=np.nan, inplace=True)
    
    # Feature where NA = 10, 11, and 12
    X.iloc[:, 125].replace(to_replace=[10, 11, 12], value=np.nan, inplace=True)

    # Index of Features where NA = 9999
    j = []
    j.extend([81, 83, 84, 86])
    for i in j:
        X.iloc[:, i].replace(to_replace=9999, value=np.nan, inplace=True)
    
    # Feature where NA = 99999
    X.iloc[:, 88].replace(to_replace=99999, value=np.nan, inplace=True)
    
    return X

# Remove Irrelevant Features 
def select_features(X):
    """Returns a subset of X
    
    Select features that are relevant to pricing an apartment that is not
    currently rented.
    
    Remove features that are tenant-specific, are leaking information, or have
    a variance of zero.
    
    """
    # Convert All Features to Numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # Index of Features to Be Removed
    j = []
    
    # Tenant Specific Features
    j.extend(range(30, 51))
    j.extend([80, 82, 85, 87, 89, 90])
    j.extend(range(92, 97))
    j.extend([98])
    j.extend(range(112, 116))
    j.extend(range(117, 124))
    j.extend([126])
    j.extend(range(129, 135))
    j.extend(range(139, 141))
    j.extend(range(144, 161))
    j.extend(range(162, 196))
    
    # Features Leaking Information
    j.extend([97])
    j.extend(range(141, 144))

    # Remove Selected Features
    X.drop(X.columns[j], axis=1, inplace=True)
    
    # Remove Features with Zero Variance
    X = X.loc[:, X.var(skipna=True) != 0]
    
    return X

# Impute Missing Values
def impute_NA(X):
    """Returns X with missing values imputed using the median and mode"""
    #List of column names for continuous variables
    list_con = ["uf12","uf13","uf14","uf15","uf16"]
    
    #List of column names for categorical variables
    list_cat = ['boro', 'uf1_1', 'uf1_2', 'uf1_3', 'uf1_4', 'uf1_5', 'uf1_6',
       'uf1_7', 'uf1_8', 'uf1_9', 'uf1_10', 'uf1_11', 'uf1_12', 'uf1_13',
       'uf1_14', 'uf1_15', 'uf1_16', 'uf1_35', 'uf1_17', 'uf1_18',
       'uf1_19', 'uf1_20', 'uf1_21', 'uf1_22', 'sc23', 'sc24', 'sc36',
       'sc37', 'sc38', 'uf48', 'sc147', 'uf11', 'sc149', 'sc173', 'sc171',
       'sc150', 'sc151', 'sc152', 'sc153', 'sc154', 'sc155', 'sc156',
       'sc157', 'sc158', 'sc181',
       'sc186', 'sc197', 'sc198', 'sc187', 'sc188', 'sc571', 'sc189',
       'sc190', 'sc191', 'sc192', 'sc193', 'sc194', 'sc196', 'sc199',
       'new_csr', 'rec15', 'uf23', 'rec21', 'rec62', 'rec64', 'rec54',
       'rec53', 'cd']
       
    #Median imputation for continuous variables
    con_df = X[list_con].fillna(X[list_con].median(), inplace=True)
    
    #Frequent imputation for categorical variables
    cat_df = X[list_cat].apply(lambda x:x.fillna(x.value_counts().index[0]))
   
    #Merge two dataset
    X_no_NaN = pd.concat([con_df, cat_df], axis=1)
    
    return X_no_NaN
    
# One-Hot-Encoding
def encode_OH(X):
    """Returns X with categorical features encoded as one-hot"""

    #List of column names for continuous variables
    list_con = ["uf12","uf13","uf14","uf15","uf16"] 
    
    #List of column names for categorical variables
    list_cat = ['boro', 'uf1_1', 'uf1_2', 'uf1_3', 'uf1_4', 'uf1_5', 'uf1_6',
       'uf1_7', 'uf1_8', 'uf1_9', 'uf1_10', 'uf1_11', 'uf1_12', 'uf1_13',
       'uf1_14', 'uf1_15', 'uf1_16', 'uf1_35', 'uf1_17', 'uf1_18',
       'uf1_19', 'uf1_20', 'uf1_21', 'uf1_22', 'sc23', 'sc24', 'sc36',
       'sc37', 'sc38', 'uf48', 'sc147', 'uf11', 'sc149', 'sc173', 'sc171',
       'sc150', 'sc151', 'sc152', 'sc153', 'sc154', 'sc155', 'sc156',
       'sc157', 'sc158', 'sc181',
       'sc186', 'sc197', 'sc198', 'sc187', 'sc188', 'sc571', 'sc189',
       'sc190', 'sc191', 'sc192', 'sc193', 'sc194', 'sc196', 'sc199',
       'new_csr', 'rec15', 'uf23', 'rec21', 'rec62', 'rec64', 'rec54',
       'rec53', 'cd']
    
    return pd.get_dummies(X, columns = list_cat)

X_train = recode_NA(X_train)
X_train = select_features(X_train)
X_train = impute_NA(X_train)
X_train = encode_OH(X_train)

X_test = recode_NA(X_test)
X_test = select_features(X_test)
X_test = impute_NA(X_test)
X_test = encode_OH(X_test)
