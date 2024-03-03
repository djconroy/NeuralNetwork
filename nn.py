import math
import random
import os
from copy import copy
from dataclasses import dataclass


@dataclass
class Example:
    input: list[float]
    output: list[float]


def sigmoid(z):
    return 0.5 + (0.5 * math.tanh(z / 2))
    #return 1.0 / (1.0 + math.exp(-z)) # got OverflowError for math.exp


def sigmoid_derivative(z):
    sig = sigmoid(z)
    return sig * (1.0 - sig)


def tanh_derivative(z):
    return 1.0 - (math.tanh(z) ** 2)


def cross_entropy(target, output):
    # Add epsilon to the input to log2 to ensure the input is > 0
    epsilon = 0.0000000000000000000000000000000000001
    return -sum([target[i] * math.log2(output[i] + epsilon)
                 for i in range(len(target))])


def squared_error(target, output):
    return 0.5 * sum([(target[i] - output[i]) ** 2
                      for i in range(len(target))])


def softmax(zs):
    max_z = max(zs)
    # Subtract max_z to prevent overflow and underflow
    # https://stackoverflow.com/a/42606665
    # This is okay since softmax(x) = softmax(x + c)
    exponentials = [math.exp(z - max_z) for z in zs]
    sum_exponentials = sum(exponentials)
    return [exp / sum_exponentials for exp in exponentials]


class TwoLayerNN:

    # Layer 1: input nodes to hidden nodes
    # Layer 2: hidden nodes to output nodes

    def __init__(self, num_inputs, num_hidden, num_outputs,
                 use_softmax=False, tanh=False, sq_error=False):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.recip_num_inputs = 1 / num_inputs
        self.recip_num_hidden = 1 / num_hidden

        self.use_softmax = use_softmax
        self.activation_function = math.tanh if tanh else sigmoid
        self.activation_derivative = (
            tanh_derivative if tanh else sigmoid_derivative)

        # Some ugly logic below, but it works for any set of options chosen
        if use_softmax:
            self.error_function = cross_entropy
        elif tanh or sq_error:
            self.error_function = squared_error
        else:
            self.error_function = cross_entropy

        self.use_cross_entropy = self.error_function is cross_entropy

    def _init_weights(self):
        # Initialize the weights to small random values around 0
        self.weights = [[[random.uniform(-self.recip_num_inputs,
                                         self.recip_num_inputs)
                          for _ in range(self.num_hidden)]
                         for _ in range(self.num_inputs)],
                        [[random.uniform(-self.recip_num_hidden,
                                         self.recip_num_hidden)
                          for _ in range(self.num_outputs)]
                         for _ in range(self.num_hidden)]]
        # self.weights[L][i][j] is the weight in layer L from input i in layer
        # L to output j in layer L where L, i and j are zero-indexed

        # Add weights (initially all 0) for a bias for every non-input node
        self.weights[0].append([0.0] * self.num_hidden)
        self.weights[1].append([0.0] * self.num_outputs)

        # Initialize weight change array
        self.d_weights = [[[0.0] * self.num_hidden
                           for _ in range(self.num_inputs + 1)],
                          [[0.0] * self.num_outputs
                           for _ in range(self.num_hidden + 1)]]

    def train(self, examples, batch_size, num_epochs, learning_rate,
              log_file_name):
        num_examples = len(examples)
        self._init_weights()

        with open(log_file_name, 'w', encoding="utf-8") as log:
            for epoch in range(1, num_epochs + 1):
                error = 0.0
                for i, example in enumerate(examples, start=1):
                    error += self._one_pass(example)
                    if i % batch_size == 0 or i == num_examples:
                        self._update_weights(learning_rate)
                log.write(f"Error at epoch {epoch}: {error}\n")

    def _update_weights(self, learning_rate):
        for layer in [0, 1]:
            for i in range(len(self.weights[layer])):
                for j in range(len(self.weights[layer][0])):
                    self.weights[layer][i][j] += (
                        -learning_rate * self.d_weights[layer][i][j])

        self.d_weights = [[[0.0] * self.num_hidden
                           for _ in range(self.num_inputs + 1)],
                          [[0.0] * self.num_outputs
                           for _ in range(self.num_hidden + 1)]]

    def _one_pass(self, example):
        inputs, z1, hidden, z2, outputs, error = self._feedforward(example)
        self._backpropagation(example, inputs, z1, hidden, z2, outputs)
        return error

    def _feedforward(self, example):
        # Copy input
        inputs = copy(example.input)
        # Add 1 to the copy for the biases for the hidden nodes
        inputs.append(1.0)

        z1 = [sum([inputs[i] * self.weights[0][i][h]
                   for i in range(len(inputs))])
              for h in range(self.num_hidden)]

        hidden = [self.activation_function(z) for z in z1]
        # Add 1 to hidden for the biases for the output nodes
        hidden.append(1.0)

        z2 = [sum([hidden[h] * self.weights[1][h][o]
                   for h in range(len(hidden))])
              for o in range(self.num_outputs)]

        if self.use_softmax:
            outputs = softmax(z2)
        else:
            outputs = [self.activation_function(z) for z in z2]

        error = self.error_function(example.output, outputs)

        return inputs, z1, hidden, z2, outputs, error

    def _backpropagation(self, example, inputs, z1, hidden, z2, outputs):
        if self.use_cross_entropy:
            # No derivative
            delta2 = [outputs[o] - example.output[o]
                      for o in range(len(outputs))]
        else: # Use squared error
            delta2 = [self.activation_derivative(z2[o])
                      * (outputs[o] - example.output[o])
                      for o in range(len(outputs))]

        for h in range(len(hidden)):
            for o in range(len(outputs)):
                self.d_weights[1][h][o] += (hidden[h] * delta2[o])

        # Note: self.num_hidden == len(hidden) - 1
        # No delta1 or d_weights[0] for the "bias node" in the hidden layer
        delta1 = [sum([delta2[o] * self.weights[1][h][o]
                       for o in range(len(outputs))])
                  * self.activation_derivative(z1[h])
                  for h in range(self.num_hidden)]

        for i in range(len(inputs)):
            for h in range(self.num_hidden):
                self.d_weights[0][i][h] += (inputs[i] * delta1[h])

    def test(self, examples):
        return sum([self.predict(example) for example in examples])

    def predict(self, example):
        _, _, _, _, _, error = self._feedforward(example)
        return error


if __name__ == "__main__":
    # letter recognition
    data_file_name = "letter-recognition.data"
    os.mkdir("letters")

    data = []
    with open(data_file_name, 'r', encoding="utf-8") as data_file:
        for line in data_file:
            row = line.strip().split(sep=',')
            inputs = [float(num) for num in row[1:]]
            outputs = [0.0] * 26
            outputs[ord(row[0]) - ord('A')] = 1.0
            data.append(Example(inputs, outputs))
    examples = data[:16000]
    test_set = data[16000:]

    train_log_name = "letters/train_errors_h{}_bs{}_lr{}.txt"
    test_info = ("Hidden units: {} Batch size: {} "
                 "Learning rate: {:7.5f} Error: {:.12f}\n")

    # Parameters
    hidden_nums = [10, 15]
    batch_sizes = [800, 16000]
    learning_rates = [0.00001, 0.001, 0.1]

    with open("letters/test_errors.txt", 'w', encoding="utf-8") as log:
        for h in hidden_nums:
            nn = TwoLayerNN(16, h, 26, use_softmax=True)
            # uses sigmoid activations and cross-entropy loss
            for bs in batch_sizes:
                for lr_num, lr in enumerate(learning_rates, start=1):
                    nn.train(
                        examples,
                        batch_size=bs,
                        num_epochs=1000,
                        learning_rate=lr,
                        log_file_name=train_log_name.format(h, bs, lr_num))
                    log.write(test_info.format(h, bs, lr, nn.test(test_set)))
                log.write("\n")
            log.write("\n")
