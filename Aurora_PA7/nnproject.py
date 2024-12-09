import csv
import numpy as np
import matplotlib.pyplot as plt
import random
import perceptron as per
import FFBP
import method as m
import networkLayer as nl

def read_dataset():
    train = []
    test = []

    with open('nnfinaldataset.csv') as f:
        reader = csv.DictReader(f)
        for entry in reader:
            input_vector = [float(entry['LAC']), float(entry['SOW'])]
            output_vector = [float(entry['TACA'])]
            new_entry = {"Inputs": input_vector, "Output": output_vector}
            if int(entry["Data Item"]) % 2:
                train.append(new_entry)
            else:
                test.append(new_entry)
    return train, test

def create_network(random_weights=False):
    if not random_weights:
        weights_hidden = [0.3, 0.3, 0.3, 0.3]
        weights_output = [0.8, 0.8]
    else:
        weights_hidden = [random.uniform(-1, 1) for _ in range(4)]
        weights_output = [random.uniform(-1, 1) for _ in range(4)]
    hidden_layer = nl.NodeLayer(2, weights_hidden, 0)
    output_layer = nl.NodeLayer(1, weights_output, 0)

    nn1 = FFBP.Network(hidden_layer, output_layer)
    return nn1

def create_perceptron(eta: float, random_weights=False):
    # Inputs and Ouput layer
    if not random_weights: 
        weights = [0.3, 0.3]
    else:
        weights = [random.uniform(-1, 1) for _ in range(2)]
    # Create nn1
    return FFBP.Simple_Perceptron(weights, 0, eta)

def calc_rocs(nn1, dataset):
    fn = 0
    fp = 0
    tp = 0 
    tn = 0 
    for entry in dataset:
        inputs = entry["Inputs"]
        outputs = entry["Output"]
        if nn1.evaluate(inputs) == 0 and outputs[0] == 0:
            tn += 1
        if nn1.evaluate(inputs) == 1 and outputs[0] == 1:
            tp += 1
        if nn1.evaluate(inputs) == 1 and outputs[0] == 0:
            fp += 1
        if nn1.evaluate(inputs) == 0 and outputs[0] == 1:
            fn += 1
    rocs = {"fn": fn, "fp": fp, "tp": tp, "tn": tn}
    return rocs

def run_simple_perceptron(train, test, iterations, eta):
    # Train Simple Perceptron
    simple_nn = create_perceptron(eta, random_weights=True)
    # Train Perceptron 
    inputs = [entry["Inputs"] for entry in train]
    outputs = [entry["Output"][0] for entry in train]
    print(simple_nn.train(train, iterations, eta, debug=True))
    threshold_logic = FFBP.Threshold(simple_nn, train)
    print(threshold_logic.threshold)
    print(calc_rocs(threshold_logic, test))

def run_network(train, test,  iterations, eta):
    for i in range(20):
        print("="*20)
        nn1 = create_network(True)
        weights = nn1.train(train, iterations, eta, debug=True)
        threshold_logic = FFBP.Threshold(nn1, train)
        
        if threshold_logic.correct == 10:
            print(weights)
            print(threshold_logic.threshold)
            print(calc_rocs(threshold_logic, test))
        print("="*20) 

def main():
    train, test = read_dataset()
    run_network(train,test, 1000, 1)
    run_simple_perceptron(train, test, 30, 0.5)
if __name__ == "__main__":
    main()