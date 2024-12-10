Creating Objects (Perceptron/Network Layer/FFBP) in main.py to use them.

Perceptron Class
The fundamental building block is the Perceptron, which is defined in perceptron.py.
By default, this perceptron uses the sigmoid function as its activation function. If you wish to use a different activation or delta function, simply define a new calc_activation function and a new delta function.

Network Layer
A Network Layer is composed of multiple perceptrons. The NodeLayer class, defined in networkLayer.py, contains a list of perceptrons and records the inputs to the layer (i.e., inputs to all perceptrons) from top to bottom. By default, each perceptron is assumed to have 2 inputs and 1 bias.
There are two types of network layers. Use the corresponding calc_layer function for each type.

FFBP (Feed-Forward Backpropagation) Network
The FFBP network, defined in FFBP.py, is built from network layers. By default, the FFBP network consists of 1 hidden layer and 1 output layer.

Methods
The methods include two functions used in PA7.

