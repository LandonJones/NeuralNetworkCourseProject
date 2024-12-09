import perceptron as per
import networkLayer as nl
import FFBP
import method as m
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   
    weights_hidden = []
    weights_output = []

    # Construct layers
    hidden_layer = nl.NodeLayer(2, weights_hidden, bias)
    output_layer = nl.NodeLayer(1, weights_output, bias)\

    # compute error
    def E(e):
        return 0.5 * e ** 2

    # Method 1
    NN1 = FFBP.Network(hidden_layer, output_layer)
    


























