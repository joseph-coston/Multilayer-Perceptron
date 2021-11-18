# Classes for building and running a neural network
# MLP Project - CSE489
# Joseph Coston & Douglas Newquist

import math
import random
import numpy as np
from typing import Callable, List


def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))


class Neuron:
    bias: float
    weights: List[float]
    sigmoid: Callable

    @property
    def output_size(self):
        return 1

    @property
    def input_size(self):
        return len(self.weights)

    def __init__(self, inputs, sigmoid=sigmoid, bias=None, weight_range=(-1, 1)) -> None:
        """Creates a new neuron

        Args:
            inputs (int or list of floats): How many input connections this neuron has, if an int each weight is randomized in the range (-1, 1).

            sigmoid (function, optional): Function to run on the sum of all inputs to this neuron.

            bias (float, optional): This neuron's bias defaults to a random value (-1, 1).

            weight_range (tuple, optional): The minimum and maximum random connection/bias weights
        """
        # assign neuron bias
        if bias is None:
            # pick a random bias
            self.bias = random.uniform(*weight_range)
        else:
            self.bias = bias

        # either choose n random weights or load input weights provided
        if isinstance(inputs, int):
            self.weights = [random.uniform(*weight_range)
                            for i in range(inputs)]
        else:
            self.weights = inputs

        self.sigmoid = sigmoid

    def evaluate(self, inputs: List[float]) -> float:
        total = self.bias

        for input, weight in zip(inputs, self.weights):
            total += input * weight

        if self.sigmoid is None:
            return total

        return self.sigmoid(total)

    def get_matrix(self, dim=500, scale=1) -> List[List[float]]:
        # generate an array to hold the output space of the perceptron
        op_space = [np.zeros(dim)]*dim

        # loop over inputs from -dim/2 to +dim/2 in two dimensions, storing the outputs of perceptron
        for i in range(dim):
            tmp = np.zeros(dim)
            for j in range(dim):
                tmp[j] = self.evaluate([(i-dim/2)*scale, (j-dim/2)*scale])
            op_space[i] = tmp.copy()

        return op_space


class Layer:
    neurons: List[Neuron]

    @property
    def input_size(self):
        if len(self.neurons) == 0:
            return 0

        return self.neurons[0].input_size

    @property
    def output_size(self):
        return len(self.neurons)

    def __init__(self, neurons, input_size=None, sigmoid=sigmoid, weight_range=(-1, 1)) -> None:
        """Creates a new neuron layer

        Args:
            neurons (int or list of Neurons): How many neurons are in this layer

            input_size (int): How many inputs every neuron has. Required if neurons is an int

            sigmoid (function): Function to run on the sum of all inputs to neurons. Required if neurons is an int

            weight_range (tuple, optional): The minimum and maximum random connection/bias weights
        """

        if isinstance(neurons, list):
            self.neurons = neurons
        else:
            self.neurons = [Neuron(input_size, sigmoid, weight_range=weight_range)
                            for n in range(neurons)]

    def evalute(self, inputs: List[float]) -> List[float]:
        return [neuron.evaluate(inputs) for neuron in self.neurons]

    def get_matrix(self, dim=500, scale=1, mode='add') -> List[List[float]]:
        '''
            Function to return a matrix representative of the layer's output space.
            Args:
                dim (int): the size of the output array (centered on 0)
                scale (double): the scale multiplier for changing range of array
                mode (string): can be one of:
                    'add'      - neuron matrices are simply summed together
                    'gradient' - neuron matrices are added to twice the previous matrix to create a gradient
                    'flatten'  - neuron matrices are added but sum is capped at 1
        '''
        msum = [np.zeros(dim)]*dim
        if mode=='flatten':
            msum = self.neurons[0].get_matrix(dim, scale)
        for neuron in self.neurons:
            if mode=='add':
                msum = np.add(neuron.get_matrix(dim, scale), msum)
            elif mode=='gradient':
                msum = np.add(neuron.get_matrix(dim, scale), np.add(msum, msum))
            elif mode=='flatten':
                msum = np.add(neuron.get_matrix(dim, scale), msum)
                msum = np.where(msum!=np.amax(msum), 0, msum)
                msum = np.where(msum==np.amax(msum), 1, msum)
        return msum


class NeuralNetwork:
    layers: List[Layer] = []

    @property
    def input_size(self):
        if len(self.layers) == 0:
            return 0

        return self.layers[0].input_size

    @property
    def output_size(self):
        if len(self.layers) == 0:
            return 0

        return self.layers[-1].output_size

    def __init__(self, layers: list, sigmoid=sigmoid, weight_range=(-1, 1)) -> None:
        """Creates a new neural network

        Args:
            layers (list): Either a list of Layers or a list of integers of form [input count, layer size, ..., layer size, output count].

            sigmoid (function): Function to run on the sum of all inputs to neurons.

            weight_range (tuple, optional): The minimum and maximum random connection/bias weights
        """

        if len(layers) > 0 and isinstance(layers[0], int):
            for input_size, size in zip(layers, layers[1:]):
                self.layers.append(
                    Layer(size, input_size, sigmoid, weight_range))
        else:
            self.layers = layers

    def evalute(self, inputs: List[float]) -> List[float]:
        for layer in self.layers:
            inputs = layer.evalute(inputs)

        return inputs


def main():
    # Example random neuron with 5 inputs
    a = Neuron(5, sigmoid=sigmoid)
    print(a.evaluate([1, 1, 1, 1, 1]))

    # Example neuron layer with 2 existing neurons with 5 inputs
    layer = Layer([a, a])
    print(layer.evalute([1, 1, 1, 1, 1]))

    # Example random neuron layer with 3 random neurons with 5 inputs
    random_layer = Layer(3, 5, sigmoid)
    print(random_layer.evalute([1, 1, 1, 1, 1]))

    # Example neural network with 2 layers
    net = NeuralNetwork([random_layer, Layer(2, 3, sigmoid)])
    print(net.evalute([1, 1, 1, 1, 1]))

    # Example linear perceptron describing the line (x + y > 0)
    s1 = Neuron([1, 1], lambda x: x > 0, 0)
    print(s1.evaluate((0, 0)))
    print(s1.evaluate((0, 1)))
    print(s1.evaluate((0, -1)))

    # Example linear perception describing the line (2x + y + 0.5 > 0)
    s2 = Neuron([2, 1], lambda x: x > 0, 0.5)
    print(s2.evaluate((0, 0)))
    print(s2.evaluate((0, 1)))
    print(s2.evaluate((0, -1)))

    # Example neural network with 3 inputs, 2 hidden layers of sizes 3 and 5, with 7 outputs
    net = NeuralNetwork([3, 3, 5, 7])
    print(net.evalute([1, 1, 1]))
    print(net.output_size)


if __name__ == "__main__":
    main()
