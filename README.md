# Matrices Program

## Installations
1. Python 3.7+
2. Dependencies:
   pip install numpy matplotlib seaborn

## Functionalities
- Matrix Creation: Create matrices from user input (inputting arrays line by line), using random values with row and column
sizes 1-20 and values 0-9999, or special matrices such as zero, identity, or diagonal
- Arithmetic Operations: Add, subtract, multiply
- Advanced Operations: Transpose, determinants, inverses, and eigen decomposition
- Visualization: Display matrices using heat maps
- File Handling: Save and load matrices to and from files

## Usage
- Main Script: python main.py
- Unittest: python -m unittest unit_test.py

### Function Usage
#### Matrix Creation
- Random Matrix: Matrix.matrix_from_random(rows=<number>, cols=<number>, low=<min_value>, high=<max_value>)
- Matrix from Input: Matrix.matrix_from_input()

- Identity Matrix: Matrix.identity(size)
- Zero Matrix: Matrix.zero(rows, cols)
- Diagonal Matrix: Matrix.diagonal([values])

#### Operations
- Addition: mat1.add(mat2)
- Subtraction: mat1.subtract(mat2)
- Multiplication: mat1.multiply(mat2)
- Transpose: mat1.transpose()
- Determinant: mat1.determinant()
- Inverse: mat1.inverse()
- Eigenvalue Decomposition: mat1.eigen_decomposition()
- Visualization: mat.visualize()

#### Saving and Loading
- Save: mat.save_to_file("filename.json")
- Load: loaded_mat = Matrix.load_from_file("filename.json")

