from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('dataset.csv')  
X = df.drop(columns=['Price'])
y = df['Price']
categorical_features = ['District', 'Location', 'House_Type']

preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_features)])
model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
model.fit(X, y)

app = Flask(__name__)

@app.route('/')
def form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    user_data = {
        'Area_sqft': float(request.form['Area_sqft']),
        'Age_Years': float(request.form['Age_Years']),
        'Bedrooms': int(request.form['Bedrooms']), 
        'Bathrooms': int(request.form['Bathrooms']),
        'District': request.form['District'],
        'Location': request.form['Location'],
        'House_Type': request.form['House_Type']
    }

    appreciation_rate = float(request.form['appreciation_rate']) / 100
    years = int(request.form['years'])

    user_df = pd.DataFrame([user_data])

    predicted_current_price = model.predict(user_df)[0]

    future_price = predicted_current_price * (1 + appreciation_rate) ** years

    return render_template('result.html', current_price=predicted_current_price, future_price=future_price, years=years)

if __name__ == '__main__':
    app.run(debug=True)
