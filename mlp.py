import math
import random
from typing import Callable, List


def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))


class Neuron:
    bias: float = random.uniform(-1.0, 1.0)
    weights: List[float]
    sigmoid: Callable

    def __init__(self, input_size, sigmoid) -> None:
        self.weights = [random.uniform(-1.0, 1.0) for i in range(input_size)]
        self.sigmoid = sigmoid

    def evaluate(self, inputs):
        total = self.bias

        for input, weight in zip(inputs, self.weights):
            total += input * weight

        return self.sigmoid(total)


class Layer:
    neurons: List[Neuron]

    def __init__(self, neurons, input_size=None, sigmoid=None) -> None:
        if isinstance(neurons, list):
            self.neurons = neurons
        else:
            self.neurons = [Neuron(input_size, sigmoid)
                            for n in range(neurons)]

    def evalute(self, inputs):
        return [neuron.evaluate(inputs) for neuron in self.neurons]


class NeuralNetwork:
    layers: List[Layer]

    def __init__(self, layers: list) -> None:
        self.layers = layers

    def evalute(self, inputs):
        for layer in self.layers:
            inputs = layer.evalute(inputs)

        return inputs


def main():
    a = Neuron(5, sigmoid=sigmoid)
    print(a.evaluate([1, 1, 1, 1, 1]))

    layer = Layer([a])
    print(layer.evalute([1, 1, 1, 1, 1]))

    random_layer = Layer(3, 5, sigmoid)
    print(random_layer.evalute([1, 1, 1, 1, 1]))

    net = NeuralNetwork([random_layer, Layer(2, 3, sigmoid)])
    print(net.evalute([1, 1, 1, 1, 1]))


if __name__ == "__main__":
    main()
