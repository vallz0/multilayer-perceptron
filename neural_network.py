import random
from typing import List, Tuple

class NeuralNetwork:
    def __init__(self, layers: List[int]) -> None:
        self.layers = layers
        self.weights = []
        self.biases = []
        self._initialize_weights_and_biases()

    def _initialize_weights_and_biases(self) -> None:
        """Initialize the weights and biases of the network."""
        for i in range(1, len(self.layers)):
            weight = [[random.uniform(-1, 1) for _ in range(self.layers[i - 1])] for _ in range(self.layers[i])]
            bias = [random.uniform(-1, 1) for _ in range(self.layers[i])]
            self.weights.append(weight)
            self.biases.append(bias)

    @staticmethod
    def sigmoid(x: float) -> float:
        """Sigmoid activation function."""
        return 1 / (1 + (2.71828 ** -x))

    def feedforward(self, inputs: List[float]) -> List[float]:
        """Process an input through the network."""
        output = inputs
        for i in range(len(self.weights)):
            output = [self.sigmoid(sum(weight * x for weight, x in zip(self.weights[i][j], output)) + self.biases[i][j]) for j in range(len(self.weights[i]))]
        return output

    def backpropagate(self, inputs: List[float], expected: List[float], learning_rate: float) -> None:
        """Perform backpropagation and adjust weights and biases."""
        activations = [inputs]
        weighted_sums = []

        # Feedforward
        input_layer = inputs
        for i in range(len(self.weights)):
            weighted_sum = [sum(weight * x for weight, x in zip(self.weights[i][j], input_layer)) + self.biases[i][j] for j in range(len(self.weights[i]))]
            input_layer = [self.sigmoid(x) for x in weighted_sum]
            activations.append(input_layer)
            weighted_sums.append(weighted_sum)

        # Backpropagation
        deltas = []
        # Calculate delta for the output layer
        output_layer = activations[-1]
        delta_output = [output_layer[i] * (1 - output_layer[i]) * (expected[i] - output_layer[i]) for i in range(len(output_layer))]
        deltas.append(delta_output)

        # Calculate deltas for the previous layers
        for i in range(len(self.weights) - 2, -1, -1):
            layer_delta = [
                activations[i + 1][j] * (1 - activations[i + 1][j]) * sum(self.weights[i + 1][k][j] * deltas[0][k] for k in range(len(self.weights[i + 1]))) for j in range(len(activations[i + 1]))
            ]
            deltas.insert(0, layer_delta)

        # Update weights and biases
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] += learning_rate * deltas[i][j] * activations[i][k]
            for j in range(len(self.biases[i])):
                self.biases[i][j] += learning_rate * deltas[i][j]

    def train(self, training_data: List[Tuple[List[float], List[float]]], epochs: int, learning_rate: float) -> None:
        """Train the neural network."""
        for epoch in range(epochs):
            for inputs, expected in training_data:
                self.backpropagate(inputs, expected, learning_rate)

    def predict(self, inputs: List[float]) -> List[float]:
        """Make a prediction based on input data."""
        return self.feedforward(inputs)
