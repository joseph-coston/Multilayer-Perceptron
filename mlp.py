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


def main():
    a = Neuron(5, sigmoid=sigmoid)
    print(a.evaluate([1, 1, 1, 1, 1]))


if __name__ == "__main__":
    main()
