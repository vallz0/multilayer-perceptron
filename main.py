from neural_network import NeuralNetwork

def main() -> None:
    # Example neural network with 2 inputs, 3 neurons in the hidden layer, and 1 output
    nn = NeuralNetwork([2, 3, 1])

    # Training data for a simple AND problem
    training_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
    ]

    # Train the network
    nn.train(training_data, epochs=10000, learning_rate=0.1)

    # Test predictions
    for inputs, _ in training_data:
        print(f"Input: {inputs}, Prediction: {nn.predict(inputs)}")

if __name__ == "__main__":
    main()
