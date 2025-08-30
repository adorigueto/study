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

