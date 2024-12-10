import perceptron as pt


def delta_hiddenlayer(curr_activation, delta_output, weight_jk):
    return (1-curr_activation) * curr_activation * delta_output * weight_jk


class NodeLayer:
    def __init__(self,perceptron_number, weights, bias):
        self.weights = weights # List of input weights to the layer
        self.layer = []  # A list of perceptron
        self.outputs = None

        # Initialize perceptrons
        for i in range(perceptron_number):
            self.layer.append(pt.Perceptron(weights[2*i: 2*i+2],bias))

    def layer_activation(self, input_vector):
        # Compute the activation value of each perceptron
        activations = []
        for perceptron in self.layer:
            perceptron.calc_activation(input_vector)
            activations.append(perceptron.activation)
        self.outputs = activations

    def calc_outputlayer(self, input_vector, eta, desired_output):
        # If this is an output layer, use this method to compute data of each perceptron 
        for perceptron in self.layer:
            perceptron.compute(input_vector, eta, desired_output)

    def calc_hiddenlayer(self, input_vector, eta, delta_output, output_weights):
         # If this is a hidden layer, use this method to compute data of each perceptron 
        for i in range(len(self.layer)):
            perceptron = self.layer[i]
            output_weight = output_weights[i]
            perceptron.delta = delta_hiddenlayer(perceptron.activation, delta_output, output_weight)
            perceptron.new_w(input_vector, eta)
            perceptron.new_b(eta)

    def update(self):
        weights = []
        # Let each perceptron update itself
        for perceptron in self.layer:
            perceptron.update()
            weights += perceptron.weights
        # Update new weights of this layer
        self.weights = weights








