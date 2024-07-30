import pickle
import os

# import numpy as np
# import numpy as np

# # Example numpy.float64
# x = np.float64(3.14)
# float_x = x.item()

# # Example numpy.float32
# y = np.float32(2.718)
# float_y = y.item()

# print(float_x)  # Output: 3.14
# print(float_y)  # Output: 2.718
# print("float_x is of type", type(float_x))
# print("float_y is of type", type(float_y))

class Datalogger:
    def __init__(self):
        self.data = []
        self.data2 = []

    def logging(self,datapoint):
        self.data.append(datapoint)
        self.data2.append('Hello World')

# Unpickle the hello_world.pickle file
# Print current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
with open(current_dir + '/hello_world.p', 'rb') as file:
    data = pickle.load(file)

print(data.data2)