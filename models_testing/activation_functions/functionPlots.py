import numpy as np
import matplotlib.pyplot as plt

# Define the ReLU function
'''
def relu(x):
    return np.maximum(0, x)

# Generate input values
x = np.linspace(-10, 10, 400)
y = relu(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='ReLU Function', color='blue')
plt.title('Rectified Linear Activation (ReLU) Function')
plt.xlabel('Input (x)')
plt.ylabel('Output (ReLU(x))')
plt.axhline(0, color='black', lw=0.5, ls='--')  # X-axis
plt.axvline(0, color='black', lw=0.5, ls='--')  # Y-axis
plt.grid(True)
plt.legend()
plt.xlim(-10, 10)
plt.ylim(-1, 11)
plt.show()
'''

# Define the Sigmoid function
'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate input values
x = np.linspace(-10, 10, 400)
y = sigmoid(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Sigmoid Function', color='orange')
plt.title('Logistic (Sigmoid) Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output (σ(x))')
plt.axhline(0, color='black', lw=0.5, ls='--')  # X-axis
plt.axvline(0, color='black', lw=0.5, ls='--')  # Y-axis
plt.grid(True)
plt.legend()
plt.xlim(-10, 10)
plt.ylim(-0.1, 1.1)
plt.show()
'''

'''
# Define the Tanh function
def tanh(x):
    return np.tanh(x)

# Generate input values
x = np.linspace(-10, 10, 400)
y = tanh(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Tanh Function', color='green')
plt.title('Hyperbolic Tangent (Tanh) Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output (tanh(x))')
plt.axhline(0, color='black', lw=0.5, ls='--')  # X-axis
plt.axvline(0, color='black', lw=0.5, ls='--')  # Y-axis
plt.grid(True)
plt.legend()
plt.xlim(-10, 10)
plt.ylim(-1.1, 1.1)
plt.show()
'''
'''
# Define the linear function
def linear(x):
    return x

# Generate input values
x = np.linspace(-10, 10, 400)
y = linear(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Linear Function', color='purple')
plt.title('Linear Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output (f(x))')
plt.axhline(0, color='black', lw=0.5, ls='--')  # X-axis
plt.axvline(0, color='black', lw=0.5, ls='--')  # Y-axis
plt.grid(True)
plt.legend()
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()
'''

# Define the softmax function
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # for numerical stability
    return exp_z / np.sum(exp_z)

# Generate input values (logits)
logits = np.linspace(-200, 200, 400)
softmax_values = softmax(logits)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(logits, softmax_values, label='Softmax Function', color='cyan')
plt.title('Softmax Activation Function')
plt.xlabel('Input (logits)')
plt.ylabel('Output (Softmax probabilities)')
plt.axhline(0, color='black', lw=0.5, ls='--')  # X-axis
plt.axvline(0, color='black', lw=0.5, ls='--')  # Y-axis
plt.grid(True)
plt.legend()
plt.xlim(-200, 200)
plt.ylim(-0.1, 1.1)
plt.show()
