from typing import List
import numpy as np
def error(output, desired_output):
    return desired_output - output

class Model():
    def get_weights(self):
        raise NotImplementedError
    def train_step(self, inputs, eta, output):
        raise NotImplementedError
    def train(self, dataset, iterations, eta, debug=False):
        prev_weights = []
        for _ in range(iterations):
            for entry in dataset:
                inputs = entry["Inputs"]
                output = entry["Output"]
                self.train_step(inputs, eta, output)
            if debug: 
                prev_weights.append(self.get_weights())
        return prev_weights

class Network(Model):
    def __init__(self, hidden_layer,output_layer):
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer

    def feed_forward(self, input_vector):
        # Compute hidden layer activation values
        self.hidden_layer.layer_activation(input_vector)

        # Compute output layer activation values
        next_inputs = self.hidden_layer.outputs
        self.output_layer.layer_activation(next_inputs)
        output = self.output_layer.outputs
        return output[0]  # Output is a list

    def new_wb(self, input_vector, eta, desired_output):
        # Calculate new weights and bias of all network layers

        # First calculate output layer
        self.output_layer.calc_outputlayer(self.hidden_layer.outputs, eta, desired_output)

        # Prepare for computing the delta for hidden layers
        delta_output = self.output_layer.layer[0].delta
        output_weights = self.output_layer.weights
        # Second calculate hidden layer
        self.hidden_layer.calc_hiddenlayer(input_vector, eta, delta_output, output_weights)

    def update(self):
        self.hidden_layer.update()
        self.output_layer.update()

    def train_step(self, input_vector, eta, desired_output):
        # A whole FFBP train step
        self.feed_forward(input_vector)
        self.new_wb(input_vector, eta, desired_output)
        self.update()
    def get_weights(self):
        weights = []
        bias = []
        for layer in [self.hidden_layer, self.output_layer]:
            layer_weights = []
            for layer_ in layer.layer:
                layer_weights.append(layer_.weights)
                bias.append(layer_.bias)
            weights.append(layer_weights)
        return {"weights": weights, 
                "bias": bias}

# Formula of compute E:
# E = 0.5 * error ** 2

class Simple_Perceptron(Model):
    def __init__(self, weights: List, bias: float, eta: float):
        self.weights = weights
        self.bias = 0
        self.eta = eta

    def feed_forward(self, inputs: List) -> float:
        # for a simple perceptron the output is a dot product of the weights
        # and inputs plus the bias
        return sum([weight*input for weight,input in zip(self.weights, inputs)]) + self.bias

    def train_step(self, input_vector, eta, output):
        predicted_output = self.feed_forward(input_vector)
        for i in range(len(self.weights)):
            self.weights[i] += self.eta * (output[0] - predicted_output)*input_vector[i]
        self.bias += self.eta * (output[0] - predicted_output)
    def get_weights(self):
        return {"weights": self.weights, "bias": self.bias}

class Threshold:
    def __init__(self, model, dataset):
        self.model = model
        self.find_threshold(dataset)
        
    def find_threshold(self, dataset):
        max_thresh = 0 
        max_num_correct = 0
        for thresh in np.linspace(0, 1, 1000):
            curr_num_correct = 0
            for entry in dataset:
                inputs = entry["Inputs"]
                outputs = entry["Output"]
                if self.model.feed_forward(inputs) > thresh:
                    tmp_output = 1
                else:
                     tmp_output = 0
                if tmp_output == outputs[0]: 
                    curr_num_correct += 1
            if curr_num_correct > max_num_correct:
                max_thresh = thresh
                max_num_correct = curr_num_correct
        self.threshold = max_thresh
        self.correct = max_num_correct
        return max_thresh, max_num_correct
    
    def evaluate(self, inputs):
        output = self.model.feed_forward(inputs)
        return int(output >= self.threshold)
        



















