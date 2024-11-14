# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Sample data: Student CGPA and their respective salary packages
# (In practice, you'd load a dataset with real values)
# CGPA (features) and Salary (target) - made-up data
CGPA = np.array([6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]).reshape(-1, 1)
salary = np.array([3.5, 4.0, 5.0, 5.5, 6.5, 7.0, 8.0, 8.5, 9.5])  # salary in LPA (example)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(CGPA, salary, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
lin_reg = LinearRegression()

# Train the Linear Regression model
lin_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lin_reg.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Linear Regression Model Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Example prediction for a student with a CGPA of 8.5
predicted_salary = lin_reg.predict(np.array([[8.5]]))
print(f"Predicted Salary for a student with CGPA 8.5: {predicted_salary[0]:.2f} LPA")
