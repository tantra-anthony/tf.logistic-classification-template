# Classification template

# use tf.data-preprocessing-template to adjust parameters
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# get only age and EstimatedSalary
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# take 100 and 300 split

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =============================================================================

# Fit Logistic Regression to Training Set
# use linear model library, as they are separated into a linear model
from sklearn.linear_model import LogisticRegression

# create classifier for training set
classifier = LogisticRegression(random_state = 9)

# fit classifier into test set
# this is so that the classifier can learn the correlation between X and y train
classifier.fit(X_train, y_train)

# now we predict the test results after the training set has built the model
# use _pred for prediction notation
y_pred = classifier.predict(X_test)
