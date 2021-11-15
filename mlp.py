import math
import random
from typing import Callable, List


def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))


class Neuron:
    bias: float
    weights: List[float]
    sigmoid: Callable

    def __init__(self, inputs, sigmoid=None, bias=None) -> None:
        """Creates a new neuron

        Args:
            inputs (int or list of floats): How many input connections this neuron has, if an int each weight is randomized in the range (-1, 1).

            sigmoid (function): Function to run on the sum of all inputs to this neuron.

            bias (float, optional): This neuron's bias defaults to a random value (-1, 1).
        """
        # assign random biases
        if bias is None:
            bias = random.uniform(-1, 1)

        # either choose n random weights or load input weights provided
        if isinstance(inputs, int):
            self.weights = [random.uniform(-1.0, 1.0)
                            for i in range(inputs)]
        else:
            self.weights = inputs

        self.sigmoid = sigmoid
        self.bias = bias

    def evaluate(self, inputs: List[float]) -> float:
        total = self.bias

        for input, weight in zip(inputs, self.weights):
            total += input * weight

        if self.sigmoid is None:
            return total

        return self.sigmoid(total)


class Layer:
    neurons: List[Neuron]

    def __init__(self, neurons, input_size=None, sigmoid=None) -> None:
        """Creates a new neuron layer

        Args:
            neurons (int or list of Neurons): How many neurons are in this layer

            input_size (int): How many inputs every neuron has. Required if neurons is an int

            sigmoid (function): Function to run on the sum of all inputs to neurons. Required if neurons is an int
        """

        if isinstance(neurons, list):
            self.neurons = neurons
        else:
            self.neurons = [Neuron(input_size, sigmoid)
                            for n in range(neurons)]

    def evalute(self, inputs: List[float]) -> List[float]:
        return [neuron.evaluate(inputs) for neuron in self.neurons]


class NeuralNetwork:
    layers: List[Layer]

    def __init__(self, layers: List[Layer]) -> None:
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


if __name__ == "__main__":
    main()
