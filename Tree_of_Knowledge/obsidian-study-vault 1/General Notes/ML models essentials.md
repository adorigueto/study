# Linear models
## Logistic Regression
```
from sklearn.linear_model import LogisticRegression

'''
X_train (features), y_train (outputs/labels),
X_test (features), y_test (outputs/labels)
'''

clf = LogisticRegression()  # Instantiate model
clf.fit(X_train, y_train)   # Fit the model to the dataset
clf.predict(X_test)         # Make predictions on the test set
clf.predict_proba(X_test)   # See probabilities on the test set
clf.score(X_test, y_test)   # Evaluate model using mean accuracy
```
## Linear Regression
```
from sklearn.linear_model import LinearRegression

'''
X_train (features), y_train (outputs/labels),
X_test (features), y_test (outputs/labels)
'''

reg = LinearRegression()    # Instantiate model
reg.fit(X_train, y_train)   # Fit the model to the dataset
reg.coef_                   # Estimated coefficients
reg.intercept_              # Estimated intercept
reg.predict(X_test)         # Make predictions on the test set
reg.score(X_test, y_test)   # Evaluate model using R²
```

## Lasso
```
from sklearn.linear_model import Lasso

'''
X_train (features), y_train (outputs/labels),
X_test (features), y_test (outputs/labels)
'''

clf = Lasso()               # Instantiate model
clf.fit(X_train, y_train)   # Fit the model to the dataset
clf.coef_                   # Estimated coefficients
clf.intercept_              # Estimated intercept
clf.predict(X_test)         # Make predictions on the test set
clf.score(X_test, y_test)   # Evaluate model using R²
```

## Ridge Classifier
```
from sklearn.linear_model import RidgeClassifier

'''
X_train (features), y_train (outputs/labels),
X_test (features), y_test (outputs/labels)
'''

clf = RidgeClassifier()     # Instantiate model
clf.fit(X_train, y_train)   # Fit the model to the dataset
clf.score(X_test, y_test)   # Evaluate using accuracy
```
# Tree models
## Concepts
Source: https://ml-playground.com/#
#### TL;DR - If computers played 21 questions
Design a tree that tries to put data into buckets, using certain thresholds on features (i.e. inputs)

#### Parameters
- Max depth (≤ 100): Maximum number of splits for the tree

#### Use Cases
- Binary Classification
- Multi-class Classification
- Regression

Basically flowcharts. You begin at the root node. Based on the value of one feature (or sometimes more), we go to the left or right child of the tree. Et cetera, until we arrive at a leaf node, and then we make a prediction based on that leaf.

It's easy to follow a tree once you've constructed one - it's a very simple chain of 'yes/no's and true/falses. The interesting part is obviously making the tree.

![](https://ml-playground.com/build/img/tree_flow.png)![](https://ml-playground.com/build/img/tree_entropy.png)

There are various algorithms for making trees out there, but in general all work towards minimizing entropy. Not the physical kind, but the Informational kind. In the context of a decision tree and nodes, entropy is high when points in a 'bucket' vary a lot in terms of their labels. If we have a tree that buckets equal numbers of orange and purple points together, this tree has high entropy and is bad. Vice versa - if we end up bucketing orange points together, and purple points separately, then there is low entropy, and this is a good tree.

Popular techniques for using trees involves Boosting and Bagging.

Boosting involves training a large number of low-depth (ie high bias) trees that predict just above random chance - then, you intelligently let each small tree contribute towards a final, weighted prediction. This approach primarily lowers the bias of your model.

Bagging involves resampling the dataset - with your training data, we want to generate a new training set that's just a bit different from the original. We do this by randomly picking one out of n points in the original dataset - and we continue this for as many times as we need to form our new 'bag' of training data. Keep in mind the same datapoint is allowed to be picked more than once. This approach tackles variance in your model, and can reduce overfitting.

The cool thing about bagging and boosting is that you usually don't have to worry about tradeoffs - boosting reduces bias without overfitting too much, and bagging reduces variance/overfitting without increasing bias too much. For a lot of machine learning, there's a tradeoff between bias and variance - so this ability to decrease one without significantly affecting the other makes bagging and boosting so powerful.

Note that bagging and boosting are general approaches that don't have to be specific to trees - it's just that trees are more commonly associated with them. You could reduce bias of any algorithm by boosting, and reduce its variance by bagging.

#### The Good
- Very easy to implement and interpret

#### The Bad
- Overfits if tree depth is too high
- Instable - if data differs by a little bit, the resulting tree can look drastically different, especially if trees have low depth.
- Decision boundaries are orthogonal - no drawing 'slanted' lines to separate classes
## Decision Tree Classifier
```
from sklearn.tree import DecisionTreeClassifier

'''
X_train (features), y_train (outputs/labels),
X_test (features), y_test (outputs/labels)
'''

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
```

# Neighbors models
## K Nearest Neighbors Classifier
```
from sklearn.neighbors import KNeighborsClassifier


```
### Concept
Source: https://ml-playground.com/#
#### TL;DR - Birds of a feather flock together
Picks the **k closest points from training data**, then decides prediction via popular vote.
#### Parameters
- k (≥ 1): number of closest neighbors to select

#### Use Cases:
- Binary Classification
- Multi-class Classification
- Regression

A simple and straightforward algorithm. The underlying assumption is that datapoints close to each other share the same label.

Analogy: if I hang out with CS majors, then I'm probably also a CS major (or that one Philosophy major who's minoring in everything.)

Note that distance can be defined different ways, such as Manhattan (sum of all features, or inputs), Euclidean (geometric distance), p-norm distance...typically Euclidean is used (like in this demo), but Manhattan can be faster and thus preferable.

#### The Good
- Simple to implement

#### The Bad
- Non-Parametric - size of model grows as training data grows. It could take a long time to compute distances for billions of datapoints.
- Curse of Dimensionality - as number of features increase (ie. more dimensions), the average distance between randomly distributed points converge to a fixed value. This means that most points end up equidistant to each other - so distance becomes less meaningful as a metric