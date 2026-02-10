'''
MS - Artificial Intelligence and Machine Learning
Course: CSC510: Foundations of Artificial Intelligence
Module 3: Critical Thinking Assignment
Professor: Dr. Bingdong Li
Created by Mukul Mondal
January 31, 2025

Problem statement: 
Implement a simple Artificial Neural Network (ANN) from scratch using Python and NumPy having one hidden layer.

'''

from os import system, name
import numpy as np  # pip install numpy


# Clears the terminal
def clearScreen():
    if name == 'nt':  # For windows
        _ = system('cls')
    else:             # For mac and linux(here, os.name is 'posix')
        _ = system('clear')
    return


def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def ANN_Algorithm_1HiddenLayer(input_size:int , hidden_size:int, output_size:int, lr:float, epochs:int, X:np.ndarray, y:np.ndarray):
    W1 = np.random.randn(input_size, hidden_size) 
    b1 = np.zeros((1, hidden_size))

    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))

    InsignificantLossCounter: int = 0
    for epoch in range(epochs):

        # ---- Forward pass ----
        z1 = X @ W1 + b1
        a1 = relu(z1)

        z2 = a1 @ W2 + b2
        y_pred = z2  # linear output

        # ---- Loss (MSE) ----
        loss = np.mean((y_pred - y) ** 2)

        # ---- Backpropagation ----
        dloss_dypred = 2 * (y_pred - y) / len(y)

        # Output layer gradients
        dW2 = a1.T @ dloss_dypred
        db2 = np.sum(dloss_dypred, axis=0, keepdims=True)

        # Hidden layer gradients
        da1 = dloss_dypred @ W2.T
        dz1 = da1 * relu_deriv(z1)

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # ---- Gradient descent update ----
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        if loss < 0.000001:
            InsignificantLossCounter += 1
            if InsignificantLossCounter >= 5:
                print(f"Early stopping at epoch {epoch} due to insignificant loss: {loss} (< 0.000001).")
                break
        # print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss = {loss:.6f}")
    return W1, b1, W2, b2
    
def Run_ANN_Algorithm(test_value:np.ndarray, actual_X:np.ndarray, actual_y:np.ndarray, epochs: int):
    if test_value is None or actual_X is None or actual_y is None:
        print("Error: actual input values or actual output values or test_value is None")
        return None    
    if len(test_value) == 0 or len(actual_X) == 0 or len(actual_y) == 0:
        print("Error: test_value or actual_X or actual_y is empty")
        return None
    
    lr: float = 0.01
    #epochs: int = 1000
    print("Learning rate:", lr, " Epochs:", epochs)
    W1, b1, W2, b2 = ANN_Algorithm_1HiddenLayer(1, 8, 1, lr, epochs, actual_X, actual_y)
    hidden = relu(test_value @ W1 + b1)
    prediction = hidden @ W2 + b2
    return prediction

if __name__ == "__main__":
    clearScreen()
    
    actual_X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    
    # Test 1: input-output linear relation y = 2*x + 1
    #print("\n === Test: input-output relation: y = 2*x + 1 ===\n")
    #actual_y = np.array([[3], [5], [7], [9], [11]], dtype=float)

    # Test 2: input-output linear relation y = 3*x
    #print("\n === Test: input-output relation: y = 3*x ===\n")
    #actual_y = np.array([[3], [6], [9], [12], [15]], dtype=float)

    # Test 3: input-output linear relation y = 4*x - 3
    print("\n === Test: input-output linear relation: y = 4*x - 3  ====\n")
    actual_y = np.array([[1], [5], [9], [13], [17]], dtype=float)

    # Test 4: input-output non-linear relation y = x**2
    #print("\n === Test: input-output relation: y = x**2 ===\n") # not good
    #actual_y = np.array([[1], [4], [9], [16], [25]], dtype=float)
    
    epochs: int = 1000
    test_value = np.array([[6]], dtype=float)
    prediction = Run_ANN_Algorithm(test_value, actual_X, actual_y, epochs)
    if prediction is None:
        print("Error: test1 returned None")
        exit(1)
        
    i: int = 0
    print("\nTraining data:")
    while i < len(actual_X):
        print("input: ",actual_X[i][0], " output: ",actual_y[i][0])
        i += 1
    
    print("\n=== ANN Experiment Result ===")
    print("input:",test_value[0][0], " output:",round(prediction[0][0], 2))
