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
    def matrix_from_input():
        rows = int(input("Number of Rows: "))
        cols = int(input("Number of Columns: "))
        matrix = []
        for x in range(rows):
            row = list(map(int, input().split()))
            if len(row) != cols:
                raise ValueError("Elements in row don't match number of columns.")
            matrix.append(row)
        
        return Matrix(matrix)
    
    @staticmethod
    def identity(size):
        return Matrix(np.eye(size))
    
    def add(self, other):
        if self._matrix.shape != other._matrix.shape:
            raise ValueError("Can't add matrices.")
        return Matrix(self._matrix + other._matrix)
    
    def subtract(self, other):
        if self._matrix.shape != other._matrix.shape:
            raise ValueError("Can't subtract matrices.")
        return Matrix(self._matrix - other._matrix)

    def multiply(self, other):
        if self._matrix.shape[1] != other._matrix.shape[0]:
            raise ValueError("Can't multiply matrices.")
        return Matrix(np.dot(self._matrix, other._matrix))

    def transpose(self):
        return Matrix(self._matrix.T)
    
    @staticmethod
    def zero(rows, cols):
        return Matrix(np.zeros((rows, cols)))

    @staticmethod
    def diagonal(diag_values):
        return Matrix(np.diag(diag_values))
    
    def determinant(self):
        if self._matrix.shape[0] != self._matrix.shape[1]:
            raise ValueError("Need to have square matrix.")
        return np.linalg.det(self._matrix)

    def inverse(self):
        if self._matrix.shape[0] != self._matrix.shape[1]:
            raise ValueError("Need to have square matrix.")
        if np.linalg.det(self._matrix) == 0:
            raise ValueError("Matrix is singular, can't invert.")
        return Matrix(np.linalg.inv(self._matrix))

    def eigen_decomposition(self):
        if self._matrix.shape[0] != self._matrix.shape[1]:
            raise ValueError("Matrix must be square for eigen decomposition.")
        eigenvalues, eigenvectors = np.linalg.eig(self._matrix)
        return eigenvalues, eigenvectors
    
    def display(self):
        sea.heatmap(self._matrix, annot=True, fmt=".0f", cmap="inferno")
        plot.title("Matrix Visualization")
        plot.show()
        
    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self._matrix.tolist(), f)

    @staticmethod
    def load_from_file(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return Matrix(data)

    

#test1 = Matrix.matrix_from_random(5, 5)
#test2 = Matrix.matrix_from_random(5, 5)

#print(test1.transpose()._matrix)
#test1.display()
#print(test1._matrix)
#print("-----------------------")
#print(test2._matrix)
#print("-----------------------")
#print(test1.multiply(test2)._matrix)
#print("-----------------------")

#print(Matrix.matrix_from_input()._matrix)

