import scipy.special, numpy

# Neural network class definition
class NeuralNetwork:
# Init the network, this gets run whenever we make a new instance of this class
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set the number of nodes in each input, hidden and output layer
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes

        # Weight matrices, wih (input -> hidden) and who (hidden -> output)
        self.wih = numpy.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = numpy.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))
        
        # Set the learning rate
        self.lr = learning_rate
        # Set the activation function, the logistic sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)
        # self.activation_function = lambda x: numpy.tanh(x)
        # self.activation_function = lambda x: numpy.maximum(0, x)
        # Train the network using back-propagation of errors
    
    @staticmethod
    def relu_function(x): 
        if x <= 0: 
            return 0. 
        return 0.01*x
    
    def train(self, inputs_list, targets_list):
       # Convert inputs into 2D arrays
        inputs_array = numpy.array(inputs_list, ndmin=2).T
        targets_array = numpy.array(targets_list, ndmin=2).T
        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs_array)
        # Calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # Calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # Current error is (target - actual)
        output_errors = targets_array - final_outputs
        # Hidden layer errors are the output errors, split by the weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
        numpy.transpose(hidden_outputs))
        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
        numpy.transpose(inputs_array))

    # Query the network
    def query(self, inputs_list):
        # Convert the inputs list into a 2D array
        inputs_array = numpy.array(inputs_list, ndmin=2).T
        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs_array)
        # Calculate output from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # Calculate signals into final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Calculate outputs from the final layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
    

class NeuralNetwork_3Layers:
# Init the network, this gets run whenever we make a new instance of this class
    def __init__(self, input_nodes, layer_1, layer_2, output_nodes, learning_rate):
        # Set the number of nodes in each input, hidden and output layer
        self.i_nodes = input_nodes
        self.h_1_nodes = layer_1
        self.h_2_nodes = layer_2
        self.o_nodes = output_nodes

        # Weight matrices, wih (input -> hidden) and who (hidden -> output)
        self.wih1 = numpy.random.normal(0.0, pow(self.h_1_nodes, -0.5), (self.h_1_nodes, self.i_nodes))
        self.wh1h2 = numpy.random.normal(0.0, pow(self.h_2_nodes, -0.5), (self.h_2_nodes, self.h_1_nodes))
        self.wh2o = numpy.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_2_nodes))

        # Set the learning rate
        self.lr = learning_rate
        # Set the activation function, the logistic sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)
        # Train the network using back-propagation of errors
    def train(self, inputs_list, targets_list):
       # Convert inputs into 2D arrays
        inputs_array = numpy.array(inputs_list, ndmin=2).T
        targets_array = numpy.array(targets_list, ndmin=2).T
        # Calculate signals into hidden layer
        layer_1_inputs = numpy.dot(self.wih, inputs_array)
        # Calculate the signals emerging from hidden layer
        layer_1_outputs = self.activation_function(layer_1_inputs)
        # Calculate signals into final output layer
        layer_2_inputs = numpy.dot(self.who, layer_1_outputs)
        

        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # Current error is (target - actual)
        output_errors = targets_array - final_outputs
        # Hidden layer errors are the output errors, split by the weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
        numpy.transpose(hidden_outputs))
        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
        numpy.transpose(inputs_array))

    # Query the network
    def query(self, inputs_list):
        # Convert the inputs list into a 2D array
        inputs_array = numpy.array(inputs_list, ndmin=2).T
        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs_array)
        # Calculate output from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # Calculate signals into final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Calculate outputs from the final layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs