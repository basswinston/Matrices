import unittest
import numpy as np
from main import Matrix

class TestMatrices(unittest.TestCase):
    def test_addition(self):
        mat1 = Matrix([[1, 2], [3, 4]])
        mat2 = Matrix([[5, 6], [7, 8]])
        result = mat1.add(mat2)
        expected = np.array([[6, 8], [10, 12]])
        np.testing.assert_array_equal(result._matrix, expected)

    def test_subtraction(self):
        mat1 = Matrix([[5, 6], [7, 8]])
        mat2 = Matrix([[1, 2], [3, 4]])
        result = mat1.subtract(mat2)
        expected = np.array([[4, 4], [4, 4]])
        np.testing.assert_array_equal(result._matrix, expected)

    def test_multiplication(self):
        mat1 = Matrix([[1, 2], [3, 4]])
        mat2 = Matrix([[2, 0], [1, 2]])
        result = mat1.multiply(mat2)
        expected = np.array([[4, 4], [10, 8]])
        np.testing.assert_array_equal(result._matrix, expected)

    def test_transpose(self):
        mat = Matrix([[1, 2, 3], [4, 5, 6]])
        result = mat.transpose()
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(result._matrix, expected)

    def test_determinant(self):
        mat = Matrix([[1, 2], [3, 4]])
        result = mat.determinant()
        expected = -2.0
        self.assertAlmostEqual(result, expected)

    def test_inverse(self):
        mat = Matrix([[1, 2], [3, 4]])
        result = mat.inverse()
        expected = np.array([[-2.0, 1.0], [1.5, -0.5]])
        np.testing.assert_array_almost_equal(result._matrix, expected)

    def test_eigen_decomposition(self):
        mat = Matrix([[5, 4], [1, 2]])
        eigenvalues, eigenvectors = mat.eigen_decomposition()
        expected_eigenvalues = np.array([6.0, 1.0])
        np.testing.assert_array_almost_equal(np.sort(eigenvalues), np.sort(expected_eigenvalues))

    def test_addition(self):
        mat1 = Matrix([[1, 2]])
        mat2 = Matrix([[3], [4]])
        with self.assertRaises(ValueError):
            mat1.add(mat2)

    def test_multiplication(self):
        mat1 = Matrix([[1, 2]])
        mat2 = Matrix([[3, 4]])
        with self.assertRaises(ValueError):
            mat1.multiply(mat2)

    def test_inverse_singular(self):
        mat = Matrix([[1, 2], [2, 4]])  # Singular matrix
        with self.assertRaises(ValueError):
            mat.inverse()

    def test_save_and_load(self):
        mat = Matrix([[1, 2], [3, 4]])
        mat.save_to_file("test_matrix.json")
        loaded_mat = Matrix.load_from_file("test_matrix.json")
        np.testing.assert_array_equal(mat._matrix, loaded_mat._matrix)

if __name__ == "__main__":
    unittest.main()