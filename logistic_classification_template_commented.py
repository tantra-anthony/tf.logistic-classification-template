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

# we need to create a Confusion Matrix
# then we need to evaluate the results, comparing the prediction and the actual test
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap

# first we set the x and y axis
X_set, y_set = X_train, y_train

# then we create the plot pixel by pixel for every possible point (or unit by unit)
# then the ListedColormap will put red and green for 0 and 1 results from the
# classifier respectively
# the limits are straight line
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# X_set[:, 0].min() - 1 takes the minimum value of the X_set - 1, so that it will not be cluttered
# same for the stop variable, and similar for the salary range
# this step decides the scale of the graph

# this following function is where we apply the classifier, where colours are also applied
# contourf takes care of the colouring
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

# the following code will plot the limits of the estimated salary
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# the loop here plots all the data points that are the real values from X_set, y_set
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
    
# the following code is just to indicate which is which
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

# display graph
plt.show()

# IMPLICATIONS FROM TRAINING SET
# points on the graph shows all the data points for the TRAINING SET
# each user is characterised by age and salary (x and y axis resp)
# red ones are training observations where purchased = 0
# green ones is 1
# people with higher salaries are predicted to buy the SUV
# older people are predicted to buy the SUV (probably because of savings?)
# goal is to classify the right users to the right category
# the two categories are yes or no to buying the SUV
# prediction regions (red and green) represent the classifications
# thus implication is that the company can target the ads more towards the 
# green region
# straight line is called the "prediction boundary"
# the fact that it is a straight line is because the logistic regression classifier
# is linear as we set it previously
# in 3D comparatively it's going to be a plane separating 2 3D spaces
# if we build a Non-Linear Classifier, then the prediction boundary will not
# be a straight line
# the randomness of data and the fact that we're using only a linear separator
# there are still quite a lot of errors


# now we visualise the test set using the same boilerplate code
# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# IMPLICATIONS FROM TEST SET
# the two matrixes we see is the Right or Wrong from 2 different prediction
# regions


