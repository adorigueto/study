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
## K Neighbors Classifier
```
from sklearn.neighbors import KNeighborsClassifier


```