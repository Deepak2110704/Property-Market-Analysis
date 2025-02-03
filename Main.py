import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
df = pd.read_csv('dataset.csv')  # Replace 'dataset.csv' with the path to your dataset

# Define features and target variable
X = df.drop(columns=['Price'])  # Features (excluding 'Price')
y = df['Price']  # Target variable

categorical_features = ['District', 'Location', 'House_Type']  # Categorical columns to one-hot encode

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing for categorical data
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), categorical_features)]
)

# Create a RandomForest model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit the model
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")

# Function to predict future house price
def predict_future_price():
    print("\nProvide input for prediction:")
    user_data = {}
    
    # Take input for all features except the target 'Price'
    for column in X.columns:
        if column == 'Area_sqft':
            user_data[column] = float(input(f"Enter value for {column} (in square feet): "))
        elif column == 'Age_Years':
            user_data[column] = float(input(f"Enter value for {column} (in years): "))
        elif column in ['Bedrooms', 'Bathrooms']:
            user_data[column] = int(input(f"Enter value for {column}: "))
        elif column in ['District', 'Location', 'House_Type']:
            user_data[column] = input(f"Enter value for {column}: ")
    
    # Get the annual appreciation rate (user input or from the dataset, e.g., average rate)
    appreciation_rate = float(input("Enter the Annual Appreciation Rate (in percentage): ")) / 100  # Convert to decimal

    # Convert the user input to a DataFrame with the same columns as X
    user_df = pd.DataFrame([user_data])
    
    # Predict the current price using the model
    predicted_current_price = model.predict(user_df)[0]
    print(f"Predicted Current Price: {predicted_current_price}")

    # Input the number of years for prediction
    years = int(input("Enter the number of years to predict the future price: "))

    # Calculate the future price using the appreciation formula
    future_price = predicted_current_price * (1 + appreciation_rate) ** years
    print(f"Predicted Future Price in {years} years: {future_price:.2f}")

# Call the prediction function to allow user inputs for future price predictions
predict_future_price()
