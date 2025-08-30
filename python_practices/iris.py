
# %% Load necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# %% Load the Iris dataset from a CSV file
# Load the Iris dataset from the UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris = pd.read_csv(url, header=None, names=columns)

print(iris.head())

# %% Split the dataset into features and target variable
X = iris.drop("class", axis=1)
y = iris["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# %% Train Logistic Regression model

logreg = LogisticRegression()          # Instantiate model
logreg.fit(X_train, y_train)           # Fit the model to the dataset
logreg.predict(X_test)                 # Make predictions on the test set
logreg.predict_proba(X_test)           # See probabilities on the test set
print(logreg.score(X_test, y_test))    # Evaluate model using mean accuracy

# %% Train Decision Tree Classifier model

dct = DecisionTreeClassifier()          # Instantiate model
dct.fit(X_train, y_train)               # Fit the model to the dataset
print(dct.score(X_test, y_test))        # Evaluate model using mean accuracy

# %% 
