import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x*(1-x)

class Perceptron:
    def __init__(self, weights, bias):
        self.weights = weights # Vector
        self.new_weights = []
        self.bias = bias
        self.new_bias = 0
        self.activation = 0
        self.delta = None
        self.input_size = len(weights)

    def calc_activity(self, input):
        # Initial activity value
        activity = 0
        for i in range(0, self.input_size):
            activity += input[i] * self.weights[i]
        activity += self.bias
        return activity

    def calc_activation(self, input_vector):
        activity = self.calc_activity(input_vector)
        self.activation = sigmoid(activity)

    def calc_delta(self, desired_output):
        # Compute error
        error = desired_output - self.activation
        # Compute delta - delta value computed using PDF
        self.delta = error * sigmoid_derivative(self.activation)

    def new_w(self, input_vector, eta):
        # Initialize a list for new weights
        self.new_weights = []

        for i in range(self.input_size):
            # Value of the change in weights based on computing PDF (Perceptron Delta Function)
            delta_weight = eta * self.delta * input_vector[i]
            self.new_weights.append(self.weights[i] + delta_weight)

    def new_b(self, eta):
        delta_bias = eta * self.delta
        self.new_bias = self.bias + delta_bias

    def compute(self, input_vector, eta, desired_output):
        self.calc_activation(input_vector)
        self.calc_delta(desired_output)
        self.new_w(input_vector, eta)
        self.new_b(eta)

    def update(self):
        self.weights = self.new_weights
        self.bias = self.new_bias


    def calc_activation2(self, input_vector):
        sum = self.calc_activity(input_vector)
        self.activation = np.sqrt(sum)
        return sum

    def new_w2(self, input_vector, eta):
        # Initialize a list for new weights
        self.new_weights = []

        for i in range(self.input_size):
            # Value of the change in weights based on computing PDF (Perceptron Delta Function)
            delta_weight = eta * self.delta * input_vector[i]
            self.new_weights.append(self.weights[i] - delta_weight)

    def calc_delta2(self, desired_output, sum):
        # Compute error
        error = desired_output - self.activation
        # Compute delta - delta value computed using PDF
        self.delta = -error * 0.5*(sum)**(-0.5)

    def compute2(self, input_vector, eta, desired_output):
        sum = self.calc_activation2(input_vector)
        self.calc_delta2(desired_output,sum)
        self.new_w2(input_vector, eta)
        self.new_b(eta)




