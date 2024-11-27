import perceptron as per
import networkLayer as nl
import FFBP
import method as m
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Single set input
    weights = [0.24, 0.88]
    input = [1, 2]
    desired_output = 0.7

    # pair 1
    input1 = [1, 1]
    desired_output1 = 0.9

    # pair 2
    input2 = [-1, -1]
    desired_output2 = 0.05

    eta = 1.0
    bias = 0

    # output_layer_inputs = [0, 0] What's this line for?

    weights_hidden = [0.3, 0.3, 0.3, 0.3]
    weights_output = [0.8, 0.8]

    # Construct layers
    hidden_layer = nl.NodeLayer(2, weights_hidden, bias)
    output_layer = nl.NodeLayer(1, weights_output, bias)\

    # compute error
    def E(e):
        return 0.5 * e ** 2

    # Method 1
    NN1 = FFBP.Network(hidden_layer, output_layer)
    E1, E2 = m.method1(NN1, input1, input2, eta, desired_output1, desired_output2, 15)
    output1 = NN1.feed_forward(input1)
    output2 = NN1.feed_forward(input2)
    e1 = (desired_output1 - output1)
    e2 = (desired_output2 - output2)
    print("output1", output1, "output2", output2, "E1", E(e1), "E2", E(e2))

    # Construct layers
    hidden_layer = nl.NodeLayer(2, weights_hidden, bias)
    output_layer = nl.NodeLayer(1, weights_output, bias)

    # Method 2
    NN_2 = FFBP.Network(hidden_layer, output_layer)
    E3,E4 = m.method2(NN_2, input1, input2, eta, desired_output1, desired_output2, 15)
    output3 = NN_2.feed_forward(input1)
    output4 = NN_2.feed_forward(input2)
    e3 = (desired_output1 - output3)
    e4 = (desired_output2 - output4)
    print("output1", output3, "output2", output4, "e3", E(e3), "e4", E(e4))




























