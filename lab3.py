from ANN import NeuralNetwork
import numpy as np

def two_input_bool():
    ret = []
    for i in range(0, 2):
        for j in range(0, 2):
            ret.append((i, j))
    return ret 


def main():
    input_nodes = 2
    hidden_nodes = 2
    output_nodes = 1
    learning_rate = 0.3

    NN = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    inputs = two_input_bool()
    
    training_inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
                                ])
    
    training_outputs = np.array([[0], 
                                 [1], 
                                 [1], 
                                 [0]])  
    
    xor = lambda x: x[0] ^ x[1]
    _and = lambda x: x[0] & x[1]
    nand = lambda x: not (x[0] & x[1])
    _or = lambda x: x[0] | x[1]

    for i in range(1000):
        NN.train(training_inputs, training_outputs)
    
    for inp in inputs:
        print(f"Input: {inp}, Expected output: {xor(inp)}, Output: {NN.query(inp)}")

    print(f"""weights inputs->hidden: {NN.wih}, 
          
weights hidden->output: {NN.who}""")
    # To get a 4 input logical func to work, you may need 1 hidden neuron per state, so 16.
    # Might be reduced by multiple hidden layers. 

if __name__ == "__main__":
    main()