"""
The sklearn.linear_model.LinearRegression class fits a classical
Ordinary Least Squares (OLS) linear regression model. Rather
than utilizing iterative optimization methods like Gradient
Descent, it relies on a direct, closed-form linear algebra
solution to compute coefficients.
"""
from sklearn.linear_model import LinearRegression

# 1. Prepare data (X must be a 2D array, y can be 1D)
hours_studied = [[1], [2], [3], [4], [5], [6]]
test_scores = [52, 58, 65, 70, 77, 83]

model = LinearRegression()
model.fit(hours_studied, test_scores)

estimated_hours = 7
prediction = model.predict([[estimated_hours]])
print(f"Predicted score for {estimated_hours} hours: {prediction[0]:.1f}")

linear_value = estimated_hours * 6.2 + 45.8
print(f"Calculated score from line chart: {linear_value:.1f}")
