# single layer neural network sample
import numpy as np

class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator
        np.random.seed(1)

        # Single neuron - 3 input connections - 1 output connection
        # Random weights 3 x 1 matrix, in the range -1 to 1 and mean 0.
        self.weights = 2 * np.random.random((3, 1)) - 1

    # sigmoid function, S shaped curve.
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of the Sigmoid function (gradient)
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # train the neural network - adjusting the synaptic weights each time.
    def train(self, input_set, output_set, iterations):
        for iteration in range(iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(input_set)

            # Calculate the error (desired output vs. predicted output)
            error = output_set - output

            if (iteration%10000) == 0:
                print("Error:" + str(np.mean(np.abs(error))))

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = np.dot(input_set.T, error * self.sigmoid_derivative(output))

            # Adjust the weights.
            self.weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.sigmoid(np.dot(inputs, self.weights))

# Main code
if __name__ == "__main__":
    #Initialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.weights)

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = np.array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print("New synaptic weights after training: ")
    print(neural_network.weights)

    # Test the neural network with a new situation.
    print("Considering new situation [1, 0, 0] -> ?: ")
    print(neural_network.think(np.array([1, 0, 0])))
