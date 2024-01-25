import numpy
import scipy.special
import imageio

class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        self.lr = learningrate
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, input_list, target_list):
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), numpy.transpose(inputs))

    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

# Load training data
training_data_path = "/Users/belgee/Downloads/archive/mnist_test.csv"
try:
    training_data = numpy.genfromtxt(training_data_path, delimiter=',')
except FileNotFoundError:
    print(f"Error: File not found at path: {training_data_path}")
    exit()
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    exit()

# Initialize neural network
inputnodes = 784
hiddennodes = 200
outputnodes = 10
learningrate = 0.1
n = NeuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate)

# Training the neural network
epochs = 10
for e in range(epochs):
    for record in training_data:
        all_values = record.astype(int)
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(outputnodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
    print("Epoch:", e)

# Testing with an image
print("Loading image digit4.png")
img_array = imageio.imread("/Users/belgee/Desktop/MachineLearning/Machine_Learning_Beginning/trainDigit/digit4.png", pilmode="L")

img_data = 255.0 - img_array.reshape(784)
img_data = (img_data / 255.0 * 0.99) + 0.01
print("Min:", numpy.min(img_data))
outputs = n.query(img_data)
print(outputs)
label = numpy.argmax(outputs)
print("AI said:", label)
