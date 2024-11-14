# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# Sample data: Predictor variables (study hours, attendance percentage) and target variable (exam score)
# (In practice, load a real dataset instead)
data = {
    'study_hours': [2, 3, 5, 7, 8, 10, 12, 15, 18, 20],
    'attendance': [70, 75, 80, 85, 90, 95, 100, 100, 100, 100],
    'exam_score': [55, 60, 65, 70, 75, 80, 85, 90, 95, 98]
}
df = pd.DataFrame(data)

# Split the dataset into features (X) and target (y)
X = df[['study_hours', 'attendance']]
y = df['exam_score']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the Multivariate Regression model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Model evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Multivariate Regression Model Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Hypothesis Testing: Checking the significance of 'study_hours' and 'attendance' on 'exam_score'
# Null Hypothesis: The coefficient of the predictor variable is zero (no effect on exam scores)
# Alternative Hypothesis: The coefficient of the predictor variable is not zero

# Get the coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Calculate t-statistics and p-values for each predictor
n = len(y_train)  # sample size
p = X_train.shape[1]  # number of predictors

# Standard error of each coefficient
std_error = np.sqrt(np.diagonal(np.linalg.inv(X_train.T.dot(X_train)) * mse))

# T-scores and p-values for hypothesis testing
t_scores = coefficients / std_error
p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=n - p - 1)) for t in t_scores]

print("\nHypothesis Testing Results:")
for i, col in enumerate(X.columns):
    print(f"{col}: Coefficient = {coefficients[i]:.2f}, T-score = {t_scores[i]:.2f}, P-value = {p_values[i]:.4f}")

# Interpreting p-values: A small p-value (< 0.05) indicates strong evidence against the null hypothesis
