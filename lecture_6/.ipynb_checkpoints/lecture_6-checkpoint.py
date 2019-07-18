# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 22:08:54 2018

@author: kevin
"""
#lecture 6
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import math
import numpy as np
#For the part about kernel trick
#lets make some variables
x = [i**2 for i in range(1, 13)]#x^2
yes = [i-2 for i in range(1, 13)]
no = [i+2 for i in range(1, 13)]

plt.scatter(x, yes)
plt.scatter(x, no)
plt.xlabel('time')
plt.ylabel('response squared')
plt.title('response over time')


x1 = [math.sqrt(i) for i in x]#sqrt(x)

plt.scatter(x1, yes)
plt.scatter(x1, no)
plt.xlabel('time')
plt.ylabel('response square_rooted')
plt.title('response over time')

#lets try making circles
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=1000, noise=0.075, random_state=0)
plt.scatter(X[:,0], X[:,1], c=y, cmap = ListedColormap(('red', 'blue')))

#create train_test set
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#lets classify
#Logistic regression:
from sklearn.linear_model import LogisticRegression
circles_classifier = LogisticRegression()
circles_classifier.fit(X_train, y_train)
y_hat = circles_classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_hat)
print(classification_report(y_test, y_hat, target_names=['0', '1']))


#plot it:
# Visualising the decision boundary
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, circles_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Testing set) - Circles Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()


#Try SVM
from sklearn import svm
circles_classifier = svm.SVC(kernel='rbf')
circles_classifier.fit(X_train, y_train)
#do some predictions and get some statistics
y_hat = circles_classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_hat)
print(classification_report(y_test, y_hat, target_names=['0', '1']))

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, circles_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Testing set) - Circles Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

#try Decision tree

from sklearn.tree import DecisionTreeClassifier
circles_classifier = DecisionTreeClassifier()
circles_classifier.fit(X_train, y_train)
#do some predictions and get some statistics
y_hat = circles_classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_hat)
print(classification_report(y_test, y_hat, target_names=['0', '1']))


from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, circles_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision tree (Testing set) - Circles Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

#try random Forest

from sklearn.ensemble import RandomForestClassifier
circles_classifier = RandomForestClassifier(n_estimators=100)
circles_classifier.fit(X_train, y_train)
#do some predictions and get some statistics
y_hat = circles_classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_hat)
print(classification_report(y_test, y_hat, target_names=['0', '1']))

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, circles_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest (Testing set) - Circles Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

### SVM with RBF kernel###########################################
# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_csv(url, names=names)
#lets make it a binary problem, remove one of the classes
data = data[data['class'] != 'Iris-setosa']

X = data.iloc[:,2:4]#sepal length and width
y = data.iloc[:,-1]#class


#Encode the y labels
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
y_label_encoder = LabelEncoder()
y = y_label_encoder.fit_transform(y)
#lets quickly look at our data
plt.scatter(X['petal-length'], X['petal-width'], c=y)#zero is purple

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn import svm
classifier = svm.SVC()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_hat = classifier.predict(X_test)


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set) - Iris Dataset')
plt.xlabel('petal-length')
plt.ylabel('petal-width')
plt.legend()
plt.show()



# Visualising the Testing set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Testing set) - Iris Dataset')
plt.xlabel('petal-length')
plt.ylabel('petal-width')
plt.legend()
plt.show()

### Decision Tree on Iris Dataset
# Fitting Decision Tree to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_hat = classifier.predict(X_test)


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree (Training set) - Iris Dataset')
plt.xlabel('petal-length')
plt.ylabel('petal-width')
plt.legend()
plt.show()



# Visualising the Testing set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree (Testing set) - Iris Dataset')
plt.xlabel('petal-length')
plt.ylabel('petal-width')
plt.legend()
plt.show()
#definite overfitting!

### Random Forest on Iris Dataset
# Random Forest to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_hat = classifier.predict(X_test)


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest (Training set) - Iris Dataset')
plt.xlabel('petal-length')
plt.ylabel('petal-width')
plt.legend()
plt.show()

# Visualising the Testing set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest (Testing set) - Iris Dataset')
plt.xlabel('petal-length')
plt.ylabel('petal-width')
plt.legend()
plt.show()



# Visualising the Testing set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree (Testing set) - Iris Dataset')
plt.xlabel('petal-length')
plt.ylabel('petal-width')
plt.legend()
plt.show()



#K-means######################
#2 clusters
### Make clusters
from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=6)
#before assigning cluster
plt.scatter(X[:,0], X[:,1])
#after assigning clusters
plt.scatter(X[:,0], X[:,1], c = y, cmap = ListedColormap(('red', 'green')))

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
y_hat = kmeans.predict(X)
plt.scatter(X[:,0], X[:,1], c = y_hat, cmap = ListedColormap(('red', 'green')))

#3 clusters

### Make clusters
from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=1)
#before assigning cluster
plt.scatter(X[:,0], X[:,1])
#after assigning clusters
plt.scatter(X[:,0], X[:,1], c = y, cmap = ListedColormap(('red', 'green', 'blue')))
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
y_hat = kmeans.predict(X)
plt.scatter(X[:,0], X[:,1], c = y_hat, cmap = ListedColormap(('red', 'green', 'blue')))

# 3 clusters Harder
### Make clusters
from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=2)
#before assigning cluster
plt.scatter(X[:,0], X[:,1])
#after assigning clusters
plt.scatter(X[:,0], X[:,1], c = y, cmap = ListedColormap(('red', 'green', 'blue')))
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
y_hat = kmeans.predict(X)
plt.scatter(X[:,0], X[:,1], c = y_hat, cmap = ListedColormap(('red', 'green', 'blue')))
#we can see that its a little harder
#Random state 55 REAL BAD
from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=55)
#before assigning cluster
plt.scatter(X[:,0], X[:,1])
#after assigning clusters
plt.scatter(X[:,0], X[:,1], c = y, cmap = ListedColormap(('red', 'green', 'blue')))
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
y_hat = kmeans.predict(X)
plt.scatter(X[:,0], X[:,1], c = y_hat, cmap = ListedColormap(('red', 'green', 'blue')))

#okay Lets try with circles:
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=1000, noise=0.03, random_state=0)
plt.scatter(X[:,0], X[:,1], c=y, cmap = ListedColormap(('red', 'blue')))
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
y_hat = kmeans.predict(X)
plt.scatter(X[:,0], X[:,1], c = y_hat, cmap = ListedColormap(('red', 'blue')))
