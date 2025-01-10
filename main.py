import numpy as np
import matplotlib.pyplot as plot
import seaborn as sea
import json

class Matrix:   
    def __init__(self, matrix):
        self._matrix = np.array(matrix)
        
    @property
    def matrix(self):
        return self._matrix
       
    @staticmethod 
    def matrix_from_random(rows=None, cols=None, low=0, high=9999):
        if rows is None:
            rows = np.random.randint(1, 20)
        if cols is None:
            cols = np.random.randint(1, 20)
            
        return Matrix(np.random.randint(low, high, (rows, cols)))
    
    @staticmethod
    def matrix_from_input():
        rows = int(input("Number of Rows: "))
        cols = int(input("Number of Columns: "))
        new_matrix = []
        for x in range(rows):
            row = list(map(int, input().split()))
            if len(row) != cols:
                raise ValueError("Elements in row don't match number of columns.")
            new_matrix.append(row)
        
        return Matrix(new_matrix)
    
    @staticmethod
    def identity(size):
        return Matrix(np.eye(size))
    
    def add(self, other):
        if self.matrix.shape != other.matrix.shape:
            raise ValueError("Can't add matrices.")
        return Matrix(self.matrix + other.matrix)
    
    def subtract(self, other):
        if self.matrix.shape != other.matrix.shape:
            raise ValueError("Can't subtract matrices.")
        return Matrix(self.matrix - other.matrix)

    def multiply(self, other):
        if self.matrix.shape[1] != other.matrix.shape[0]:
            raise ValueError("Can't multiply matrices.")
        return Matrix(np.dot(self.matrix, other.matrix))

    def transpose(self):
        return Matrix(self.matrix.T)
    
    @staticmethod
    def zero(rows, cols):
        return Matrix(np.zeros((rows, cols)))

    @staticmethod
    def diagonal(diag_values):
        return Matrix(np.diag(diag_values))
    
    def determinant(self):
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Need to have square matrix.")
        return np.linalg.det(self.matrix)

    def inverse(self):
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Need to have square matrix.")
        if np.linalg.det(self.matrix) == 0:
            raise ValueError("Matrix is singular, can't invert.")
        return Matrix(np.linalg.inv(self.matrix))

    def eigen_decomposition(self):
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Matrix must be square for eigen decomposition.")
        eigenvalues, eigenvectors = np.linalg.eig(self.matrix)
        return eigenvalues, eigenvectors
    
    def display(self):
        if len(self.matrix) == 0:
            raise ValueError("Matrix is empty. Can't display.")
        sea.heatmap(self.matrix, annot=True, fmt=".0f", cmap="inferno")
        plot.title("Matrix Visualization")
        plot.show()
        
    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.matrix.tolist(), f)

    @staticmethod
    def load_from_file(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return Matrix(data)
    

