import numpy as np

np.random.seed(0)


# input to the neuron 
X = [[1,2,3,2.5],
    [2,5,-1,2],
    [-1.5,2.7,3.3,-0.8]]

# make the vales between [-1,1] so that the values dont explode
class Layer_Dense:
    def __init__(self ,n_input , n_neurons):
        self.weights = 0.1*np.random.randn(n_input , n_neurons )
        self.biases = np.zeros((1,n_neurons))
    def foward(self , inputs):

        self.output = np.dot(inputs , self.weights) + self.biases
         

layer1 = Layer_Dense(4 , 5)
layer2 = Layer_Dense(5 , 3)

layer1.foward(X)
print(layer1.output)
layer2.foward(layer1.output)
print(layer2.output)