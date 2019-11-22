import numpy as np

def sigmond(x):
    return 1 / (1 + np.exp(-x))

def sigmond_derivative(x):
    return x * (1 - x)

training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T

#Setter seed til 1:
np.random.seed(1)


synaptic_weights = 2 * np.random.random((3,1)) - 1

print("Random Starting synaptic weights")
print(synaptic_weights)

for itaration in range(50000):

    input_layer = training_inputs

    outputs = sigmond(np.dot(input_layer, synaptic_weights))

    error = training_outputs - outputs

    adjustments = error * sigmond_derivative(outputs)

    synaptic_weights  += np.dot(input_layer.T, adjustments)


print('Synaptic weights after training')
print(synaptic_weights)

print('outputs after training: ' )
print(outputs)

