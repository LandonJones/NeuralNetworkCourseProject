

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




















