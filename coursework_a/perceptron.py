import numpy

# A single perceptron function
def perceptron(inputs_list, weights_list, bias):
# Convert the inputs list into a numpy array
    inputs = numpy.array(inputs_list)
    # Convert the weights list into a numpy array
    weights = numpy.array(weights_list)
    # Calculate the dot product
    summed = numpy.dot(inputs, weights)
    # Add in the bias
    summed = summed + bias
    # Calculate output
    # N.B this is a ternary operator, neat huh?
    # print(summed)
    output = 1 if summed > 0 else 0
    return output


def XOR_network(_input, weights_dict):
    *NAND_weights, NAND_bias = weights_dict["NAND"]
    *AND_weights, AND_bias = weights_dict["AND"]
    *OR_weights, OR_bias = weights_dict["OR"]

    NAND = perceptron(_input, NAND_weights, NAND_bias)
    OR =  perceptron(_input, OR_weights, OR_bias)

    AND = perceptron([NAND, OR], AND_weights, AND_bias)

    return AND

def main(): 
    # Our main code starts here
    # Test the perceptron
    # inputs = [1.0, 0.0]
    # weights = [1.0, 1.0]
    # bias = -1
    # print("Inputs: ", inputs)
    # print("Weights: ", weights)
    # print("Bias: ", bias)
    # print("Result: ", perceptron(inputs, weights, bias))
    bias = -1.5
    weights = [2, 2]

    weights_and_biases = {"AND": [1, 1, -1], 
                          "NAND": [-2, -2, 2.5], 
                          "OR": [2, 2, -1.5]}
    
    inputs = ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0))

    for _inp in inputs:
        #print(f"Input: {_inp}, Output: {perceptron(_inp, weights, bias)}")
        print(f"Input: {_inp}, Output: {XOR_network(_inp, weights_and_biases)}")

    # [1.0, 1.0] it's an AND gate.
    # Any other combination of two bools give 0 because the sum after bias requires the value to be >= 1 for a 
    # positive output. 

    # AND: w = 1, 1, b=-1
    # OR: w = 2, 2, b=-1.5
    # NAND: w = -2, -2, b=2.5
    # XOR: 

if __name__ == "__main__":
    main()