
# First Method
def single_cycle(NN, input1, input2, eta, desired_output1, desired_output2):
    # First input/output pair
    NN.train_step(input1, eta, desired_output1)
   # print("output_c1:", NN.output_layer.outputs, "input weights:", NN.hidden_layer.weights, "output weights:", NN.output_layer.weights)

    # Second input/output pair
    NN.train_step(input2, eta, desired_output2)
    #print("output_c2:", NN.output_layer.outputs, "input weights:", NN.hidden_layer.weights, "output weights:", NN.output_layer.weights)


def method1(NN, input1, input2, eta, desired_output1, desired_output2, times):
    for i in range(times):
        single_cycle(NN, input1, input2, eta, desired_output1, desired_output2)
    e1 = desired_output1 - NN.feed_forward(input1)
    e2 = desired_output2 - NN.feed_forward(input2)
    return e1, e2

# Second Method
def cycle_input1(NN, input1, eta, desired_output1, times):
    for i in range(times):
        NN.train_step(input1, eta, desired_output1)
        #print("c1: output:", NN.output_layer.outputs, "input weights:", NN.hidden_layer.weights, "output weights:", NN.output_layer.weights)

def cycle_input2(NN, input2, eta, desired_output2, times):
    for i in range(times):
        NN.train_step(input2, eta, desired_output2)
        #print("c2: output:", NN.output_layer.outputs, "input weights:", NN.hidden_layer.weights, "output weights:", NN.output_layer.weights)

def method2(NN, input1, input2, eta, desired_output1, desired_output2, times):
    cycle_input1(NN, input1, eta, desired_output1, times)
    cycle_input2(NN, input2, eta, desired_output2, times)

    e1 = desired_output1 - NN.feed_forward(input1)
    e2 = desired_output2 - NN.feed_forward(input2)
    return e1, e2

