from flask import Flask, render_template, request, redirect, url_for
import joblib

app = Flask(__name__)

# Load the KNN model from the model.pkl file using joblib
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    age = float(request.form['age'])
    salary = float(request.form['salary'])
    gender = int(request.form['gender'])

    # Make prediction using the loaded KNN model
    prediction = model.predict([[age, salary, gender]])

    # Redirect to the 'result' route with the prediction result and input values as query parameters
    return redirect(url_for('result', prediction=prediction[0], age=age, salary=salary, gender=gender))

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
