from flask import Flask, render_template, request, redirect, url_for, jsonify
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
app = Flask(__name__)

# Load the KNN model from the model.pkl file using joblib
model = joblib.load('model.pkl')

scaler = StandardScaler()
X_train=pd.read_csv("X_train.csv")

scaler.fit(X_train[['Age', 'AnnualSalary']])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        

        # Get user input from the form
        age = float(request.form['age'])
        salary = float(request.form['salary'])
        gender = int(request.form['gender'])


        # Create a dictionary with the user input
        user_input = {
            'Age': [age],
            'AnnualSalary': [salary],
            'Gender': [gender]
        }
        user_data = pd.DataFrame(user_input)

        # Extract Age, AnnualSalary, and Gender columns for scaling
        user_data_scaled = scaler.transform(user_data[['Age', 'AnnualSalary']])
        print(user_data_scaled)
        # Create a DataFrame
        

        # Scale user input from the form
        user_input_scaled = scaler.transform([[age, salary]])

        # Combine scaled data with Gender for final input to the model
        final_input = np.hstack((user_input_scaled, [[gender]]))

        # Make prediction using the loaded KNN model
        prediction = model.predict(final_input)

        # Redirect to the 'result' route with the prediction result and input values as query parameters
        return redirect(url_for('result', prediction=prediction[0], age=age, salary=salary, gender=gender))
    
    except Exception as e:
        # Handle exceptions (e.g., incorrect input format) gracefully
        return jsonify({'error': str(e)}), 400

@app.route('/result')
def result():
    # Get prediction result and input values from query parameters
    prediction = request.args.get('prediction')
    age = request.args.get('age')
    salary = request.args.get('salary')
    gender = request.args.get('gender')

    # Render the 'result.html' template with prediction result and input values
    return render_template('result.html', prediction=prediction, age=age, salary=salary, gender=gender)

if __name__ == '__main__':
    app.run(debug=False)
