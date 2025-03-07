#Team Members
#1. Manu Sharan Kumar (ACE080BCT037)
#2. Ocean Neupane (ACE080BCT043)
#3. Niraj Kumar Jha (ACE080BCT040)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
file_path = "health_dataset.csv"  
df = pd.read_csv(file_path)

# Define independent (X) and dependent (y) variables
X = df[['Age', 'BMI', 'GlucoseLevel', 'PhysicalActivity']].values  
y = df['DiabetesRisk'].values 

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add intercept column (bias term)
intercept = np.ones((X_train.shape[0], 1))
X_train = np.concatenate((intercept, X_train), axis=1)
intercept = np.ones((X_test.shape[0], 1))
X_test = np.concatenate((intercept, X_test), axis=1)

# Mini-batch gradient descent for logistic regression
def mini_batch_GD(X, y, max_iter=1000):
    w = np.zeros(X.shape[1])
    l_rate = 0.01
    batch_size = int(0.1 * X.shape[0])

    for i in range(max_iter):
        ix = np.random.randint(0, X.shape[0]) 
        batch_X = X[ix:ix+batch_size]
        batch_y = y[ix:ix+batch_size]
        loss, grad = gradient(batch_X, batch_y, w)
        if i % 500 == 0:
            print(f"Loss at iteration {i}: {loss}")
        w = w - l_rate * grad
    return w

def gradient(X, y, w):
    m = X.shape[0]
    h = h_theta(X, w)
    error = h - y
    loss = - np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    grad = np.dot(X.T, error)
    return loss, grad

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def h_theta(X, w):
    return sigmoid(X @ w)

def output(pred):
    return np.round(pred)

# Train the model
w = mini_batch_GD(X_train, y_train, max_iter=5000)

# Evaluate the model
yhat = output(h_theta(X_test, w))
print("Accuracy:", accuracy_score(y_test, yhat))

# Function to predict diabetes risk based on user input
def predict_diabetes():
    print("\nEnter Patient Details:")
    age = float(input("Age (years): "))
    bmi = float(input("BMI (kg/m²): "))
    glucose = float(input("Glucose Level (mg/dL): "))
    activity = float(input("Physical Activity (hours per week): "))

    # Preprocess input data
    input_data = np.array([[age, bmi, glucose, activity]])
    input_scaled = scaler.transform(input_data)  # Apply same normalization
    input_scaled = np.concatenate((np.ones((1, 1)), input_scaled), axis=1)  # Add intercept

    # Predict using the trained model
    prediction = output(h_theta(input_scaled, w))

    # Display the result
    if prediction[0] == 1:
        print("\n⚠️ High Diabetes Risk!")
    else:
        print("\n✅ Low Diabetes Risk!")

# Call the function for user input prediction
predict_diabetes()
