import numpy as np
from sklearn import linear_model

def desired_marketing_expenditure(marketing_expenditure, units_sold, desired_units_sold):

    X = np.array(marketing_expenditure).reshape(-1, 1)
    y = np.array(units_sold)
    
    model = linear_model.LinearRegression()
    model.fit(X, y)
    
    a = model.coef_[0]
    b = model.intercept_
    
    required_expenditure = (desired_units_sold - b) / a
    return required_expenditure


#For example, with the parameters below, the function should return 250000.0
print(desired_marketing_expenditure(
    [300000, 200000, 400000, 300000, 100000],
    [60000, 50000, 90000, 80000, 30000],
    60000))