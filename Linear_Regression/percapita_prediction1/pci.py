import pandas as pd
​
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
​
df=pd.read_csv('C:/Users/ASUS/Desktop/projects/canada_per_capita_income.csv')
​
# Data preprocessing (You might need to handle missing values and other preprocessing steps)
# For simplicity, let's assume the dataset has a 'Year' column and a 'PerCapitaIncome' column
​
# Splitting data into features (X) and target (y)
X = df[['year']]  # Assuming 'Year' is the feature
y = df['per capita income (US$)']
​
​
# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
​
​
​
# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
​
​
y_pred = model.predict(X_test)
​
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
​
​
predicted_per_capita_2024 = model.predict([[2024]])
print(f"Predicted per capita income for 2024: {predicted_per_capita_2024[0]}")
​
Mean Squared Error: 15147815.5477862
Predicted per capita income for 2024: 44288.24753368879
