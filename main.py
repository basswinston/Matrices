import numpy as np
import matplotlib.pyplot as plot
import seaborn as sea
import json

class Matrix:   
    def __init__(self, matrix):
        self._matrix = np.array(matrix)
       
    @staticmethod 
    def matrix_from_random(rows, cols, low=0, high=10):
        return Matrix(np.random.randint(low, high, (rows, cols)))
    
    @staticmethod
    def identity(size):
        return Matrix(np.eye(size))
    
    def add(self, other):
        if self._matrix.shape != other.matrix.shape:
            raise ValueError("Can't add matrices.")
        return Matrix(self._matrix + other.matrix)
    
    def subtract(self, other):
        if self._matrix.shape != other.matrix.shape:
            raise ValueError("Can't subtract matrices.")
        return Matrix(self._matrix - other.matrix)

    def multiply(self, other):
        if self._data.shape[1] != other.matrix.shape[0]:
            raise ValueError("Can't multiply matrices.")
        return Matrix(np.dot(self._matrix, other.matrix))

    def transpose(self):
        return Matrix(self._matrix.T)
    
    @staticmethod
    def zero(rows, cols):
        return Matrix(np.zeros((rows, cols)))

    @staticmethod
    def diagonal(diag_values):
        return Matrix(np.diag(diag_values))
    

test = Matrix.matrix_from_random(2, 3)

print(test.transpose()._matrix)

#print(test._matrix)