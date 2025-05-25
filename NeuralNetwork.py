import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle 
import os

ITERATIONS = 6000
LEARNING_RATE = 0.07

#X is 784xm matrix where 784 is number of input pixels of a 28x28 image, m is the number of images/ training example

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def define_neurons(X, Y):
    Y = one_hot(Y)
    input_layer_nuerons = X.shape[0]
    hidden_layer_neurons = 10
    output_layer_nuerons = Y.shape[0]
    return (input_layer_nuerons, hidden_layer_neurons, output_layer_nuerons)

def initialize_parameters(input_layer_neurons, hidden_layer_nuerons, output_layer_neurons):
    weights_hidden = np.random.randn(hidden_layer_nuerons, input_layer_neurons) * 0.1
    biases_hidden = np.zeros((hidden_layer_nuerons, 1))
    weights_output = np.random.randn(output_layer_neurons, hidden_layer_nuerons) * 0.1
    biases_output = np.zeros((output_layer_neurons, 1))

    parameters = {
        "weights_hidden": weights_hidden,
        "biases_hidden": biases_hidden,
        "weights_output": weights_output,
        "biases_output": biases_output
    }
    return parameters

def forward_propogation(X, parameters):
    weights_hidden = parameters['weights_hidden']
    biases_hidden = parameters['biases_hidden']
    weights_output = parameters['weights_output']
    biases_output = parameters['biases_output']  

    unactivated_hidden = np.dot(weights_hidden, X) + biases_hidden
    activated_hidden = np.tanh(unactivated_hidden)
    unactivated_output = np.dot(weights_output, activated_hidden) + biases_output
    activated_output = sigmoid(unactivated_output) 

    outputs = {
        "unactivated_hidden": unactivated_hidden,
        "activated_hidden": activated_hidden,
        "unactivated_output": unactivated_output,
        "activated_output": activated_output
    }

    return activated_output, outputs

def log_cost(activated_output, Y):
    m = Y.shape[1]
    Y = one_hot(Y)
    logs = np.multiply(np.log(activated_output), Y) + np.multiply((1 - Y), np.log(1 - activated_output))
    cost = - np.sum(logs) / m
    return float(np.squeeze(cost))

def backward_propogation(parameters, outputs, X, Y):
    m = X.shape[1] #number of training examples
    Y = one_hot(Y)

    weights_output = parameters['weights_output']
    activated_hidden = outputs['activated_hidden']
    activated_output = outputs['activated_output']

    dunactivated_output = activated_output - Y #dC/dz^[2] C is cost function, z = wx + b
    dweights_output = 1/m * np.dot(dunactivated_output, activated_hidden.T)  #dC/dW = 1/m dC/dZ^[2] . A^[1]^T
    dbiases_output = 1/m * np.sum(dunactivated_output, axis=1, keepdims=True)

    dunactivated_hidden = np.multiply(np.dot(weights_output.T, dunactivated_output), 1 - np.power(activated_hidden, 2)) 
    dweights_hidden = 1/m * np.dot(dunactivated_hidden, X.T)
    dbiases_hidden = 1/m * np.sum(dunactivated_hidden, axis=1, keepdims=True)

    gradients = {
        "dweights_hidden": dweights_hidden,
        "dbiases_hidden": dbiases_hidden,
        "dweights_output": dweights_output,
        "dbiases_output": dbiases_output
    }

    return gradients

def gradient_descent(parameters, gradients):
    global LEARNING_RATE

    parameters['weights_hidden'] -= LEARNING_RATE * gradients['dweights_hidden']
    parameters['biases_hidden'] -= LEARNING_RATE * gradients['dbiases_hidden']
    parameters['weights_output'] -= LEARNING_RATE * gradients['dweights_output']
    parameters['biases_output'] -= LEARNING_RATE * gradients['dbiases_output']

    return parameters

def model(X, Y, hidden_layer_nuerons, num_iterations):
    input_layer_neurons, hidden_layer_neurons, output_layer_neurons = define_neurons(X, Y)
    parameters = initialize_parameters(input_layer_neurons, hidden_layer_neurons, output_layer_neurons)

    for i in range(num_iterations):
        activated_output, outputs = forward_propogation(X, parameters)
        cost = log_cost(activated_output, Y)
        gradients = backward_propogation(parameters, outputs, X, Y)
        parameters = gradient_descent(parameters, gradients)
        print(f"{i + 1}/{num_iterations} (cost = {cost})")

    return parameters

def prediction(parameters, X):
    activated_output, _ = forward_propogation(X, parameters)
    return np.argmax(activated_output, axis=0)

# Main program
print("Initializing...")
np.random.seed(3)

# Load dataset
print("Reading CSV Data...")
data = pd.read_csv('mnist_train.csv')
data = np.array(data)
np.random.shuffle(data)

# Split into train and test sets
m, n = data.shape
test_data = data[0:1000].T
Y_test = test_data[0]
X_test = test_data[1:n] / 255

train_data = data[1000:m].T
Y_train = train_data[0]
X_train = train_data[1:n] / 255

Y_train = np.array([Y_train])
Y_test = np.array([Y_test])

# Choose whether to train or load
if os.path.exists("trained_model.pkl"):
    choice = input("Type 'train' to train a new model or 'load' to use saved model: ").lower()
else:
    choice = 'train'

if choice == 'train':
    print("Training model...")
    parameters = model(X_train, Y_train, 10, ITERATIONS)
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(parameters, f)
    print("Model saved as 'trained_model.pkl'")
elif choice == 'load':
    print("Loading saved model...")
    with open('trained_model.pkl', 'rb') as f:
        parameters = pickle.load(f)
else:
    print("Invalid choice.")
    exit() 

# Visual test
tests = input("Enter the number of visual tests you want to run: ")
for i in range(int(tests)):
    current_image = X_train[:, i, None]
    label = Y_train[0][i]
    pred = prediction(parameters, current_image)

    print(f"Prediction: {pred[0]}")
    if pred[0] == label:
        print("Correct")
    else:
        print(f"Incorrect (Actual: {label})")

    image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(image, interpolation='nearest')
    plt.show()

# Accuracy on test set
predictions = prediction(parameters, X_test)
correct_predictions = np.sum(predictions == Y_test[0])
total_digits = Y_test[0].size

print()
print(f"Correct Predictions: {correct_predictions}")
print(f"Total Digits Tested: {total_digits}")
print(f"Accuracy: {np.round((correct_predictions / total_digits) * 100, 1)}%")
