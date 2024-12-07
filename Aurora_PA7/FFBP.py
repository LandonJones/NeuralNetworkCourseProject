from typing import List

def error(output, desired_output):
    return desired_output - output


class Network:
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


# Formula of compute E:
# E = 0.5 * error ** 2

class Simple_Perceptron:
    def __init__(self, weights: List, eta: float):
        self.weights = weights
        self.bias = 0
        self.eta = eta

    def calculate(self, inputs: List) -> float:
        return self.weights[0] * inputs[0] + self.weights[1] * inputs[1] + self.bias
    
    def error_correct(self, inputs: List, predicted: float, expected: float):
        # Correct the weights and biases based on performance
        # Weight update formula = wi_original + eta * (y_true - y_pred) * x_i
        for i in range(len(self.weights)):
            self.weights[i] += self.eta * (expected - predicted) * inputs[i]
        # Update bias
        self.bias += self.eta * (expected - predicted)

    def round_predicted(self, predicted: float) -> int: 
        # Round the predicted value to either 0 or 1
        if predicted > 0.5: 
            return 1
        else: 
            return 0

    def train(self, inputs: List, expected: List, iterations: int):
        # Iterate over certain iterations
        for i in range(iterations):
            # Iterate over all inputs
            for data in range(len(inputs)):
                # Present i/o pair and calculate expected with rounding
                predicted = self.round_predicted(self.calculate(inputs[data]))
                # Update weights
                self.error_correct(inputs[data], predicted, expected[data])
        return 0
            
    def test(self, inputs: List) -> List: 
        # Iterate over all inputs
        predicted_list = []
        for data in range(len(inputs)):
            # Present input pair to calculate predicted value
            predicted_list.append(self.round_predicted(self.calculate(inputs[data])))
        # Return the predicted values
        return predicted_list



















