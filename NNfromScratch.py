import numpy as np 
from keras.datasets import mnist  #keras imported only for data sets access reasons, the code doesnt use it
from keras.utils import np_utils
import matplotlib.pyplot as plt

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

n=10 #you can change this number
number=x_train[n]
print(type(number))

for k in range(0,28):
    for j in range(0,28):
        if number[k,j]>0:
            number[k,j]=1

correct=y_train[n]
print('the value in the picture is',correct)
print(number)

#%%
# Layers in pairs (Dense + Activation)
# symbols: m-matrix, X-input m, Y-output m, E-error/loss function, W-weights m, B-bias m, l_r- learning rate

# pair of (Dense + Activation) create one neural network layer
        
class Dense:
    
    def __init__(self, size_input: int, size_output: int):
        """
        Assigns initial random numbers to weights and biases.
        """
        # creates the matrix of [size of output] X [size of input] filled with random values in [0,1]
        self.weights = np.random.randn(size_output, size_input) 
        # creates the matrix of [size of output] X [1]  filled with random values in [0,1]
        self.bias = np.random.randn(size_output, 1) 
        
    # forward propagation
    def forward(self, _input: np.ndarray) -> np.ndarray:
        """
        Forward propagation method. Takes input and computes output using weights and biases.

        """
        self._input = _input # input of this layer (output of previous Activation layer)
        return np.dot(self.weights, self._input) + self.bias # Y = W*X + B
    
    # backward propagation
    def backward(self, loss_output_derivative: np.ndarray, learning_rate: float) -> np.ndarray: #(self, dE/dY, l_r)
        """
        Backward propagation method. Updates values of weights and biases basing on Loss derivative.
        """
        # dE/dW = dE/dY * X, .T-input matrix transposed for ease of operations
        loss_weights_derivative = np.dot(loss_output_derivative, self._input.T) 
        # Updates values of weights W = W - lr*dE/dW
        self.weights -= learning_rate * loss_weights_derivative
        # Updates values of bias B = B - lr*dE/dB (dE/dB = dE/dY)
        self.bias -= learning_rate * loss_output_derivative
        # dE/dX = W.T * dE/dY returns output of backward propagation (input for the next layer)
        return np.dot(self.weights.T, loss_output_derivative)
    # TODO: optimize learning_rate

class Activation: 
    def __init__(self):
        """
        Initializes the class, first creates activation function methods.
        """
        def tanh(x):
            return np.tanh(x) # Activation function we are using
        def tanh_derivative(x):
            return 1 - np.tanh(x) ** 2 # Derivative of the activation function
        self.tanh = tanh    # tanh(x)
        self.tanh_derivative = tanh_derivative   # d/dx[tanh(x)]
        
    def forward(self, _input): # input of this layer (output of previous Dense layer)
        self._input = _input
        return self.tanh(self._input) # Y = tanh(X) forward output of the layer

    def backward(self, loss_output_derivative, learning_rate):#output gradient is dE/dY
        return np.multiply(loss_output_derivative, self.tanh_derivative(self._input)) # dE/dX = dE/dY * d/dx[tanh(X)] backward output of this Layer(input of the next), element-wise matrix multiplication

        
# These functions dont belong to any of above classes. They are used to calculate our very last dE/dY value.
def loss_function(y_true, y_pred): 
    return np.mean((y_true - y_pred)** 2)

def loss_output_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true) # equals to last (first calculated) dE/dY

#%%

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data: 60000 samples total
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 28 * 28, 1)
x_train = x_train.astype('float32')
x_train /= 255 # normalisation as it needs to be in [0,1] range
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)
y_train = y_train.reshape(y_train.shape[0], 10, 1)

# same for test data: 10000 samples total
x_test = x_test.reshape(x_test.shape[0], 28 * 28, 1)
x_test = x_test.astype('float32')
x_test /= 255 # normalisation as it needs to be in [0,1] range
y_test = np_utils.to_categorical(y_test)
y_test = y_test.reshape(y_test.shape[0], 10, 1)

#%%

# neural network
network = [Dense(28 * 28, 40),
          Activation(),
          Dense(40, 10),
          Activation() ]
# Number of epochs (the higher the number the more effective is learining process)
epochs = 1000
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data: 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 28 * 28, 1)
x_train = x_train.astype('float32')
x_train /= 255 # data to be in <0,1> range
# encode output which is a number in range <0,9> into a vector of size 10
# e.g. number 3 becomes [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)
y_train = y_train.reshape(y_train.shape[0], 10, 1)

# same for test data: 10000 samples
x_test = x_test.reshape(x_test.shape[0], 28 * 28, 1)
x_test = x_test.astype('float32')
x_test /= 255 # we want our data to be in [0,1] range
y_test = np_utils.to_categorical(y_test)
y_test = y_test.reshape(y_test.shape[0], 10, 1)
learning_rate = 0.1

# set the number of samples to be trained on
n=20000

#lets train it
for ep in range(epochs):
    error = 0
    
    for x, y in zip(x_train[:n], y_train[:n]): # zip is 'pairing' the arrays
        # forward propagation
        output = x
        for layer in network:
            output = layer.forward(output)

        # error sum
        error += loss_function(y, output)

        # backward propagation - here is the real learning part (changing weights and biases)
        deriv = loss_output_derivative(y, output) #our first input in backward propagation
        # goes in reversed order (backwards)
        for layer in reversed(network):
            deriv = layer.backward(deriv, learning_rate)

    error /= n #10000 samples
    print('{}/{}, error={:2.6f}'.format(ep + 1, epochs, error))
    
#%%
# Number of sets to test the NN
num=2000
success_count=0
# test on 20 samples
for x, y in zip(x_test[:num], y_test[:num]):
    output = x
    for layer in network:
        output = layer.forward(output)
    #print('prediction:', np.argmax(output), '\t true value:', np.argmax(y))
    if np.argmax(output) == np.argmax(y):
        success_count+=1
        
print('Success {} out of {}'.format(success_count, num), 100*success_count/num, "%")
    
