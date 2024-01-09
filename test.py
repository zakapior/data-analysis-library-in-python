'''
A test suite to check the data_transformations.py module.
'''

import unittest
from numpy import array, ones, array_equal
from src.data_transformations import transpose2d, window1d, convolution2d

class TestTransposition(unittest.TestCase):
    '''
    A TestCase class to test the transpose2d function against a number of matrices to determine, if
    it is working correctly. The sizes of matrixes are denoted in test name.
    '''

    def test_transposition_1x2(self):

        data = [[1], [2]]
        result = [[1, 2]]
        self.assertEqual(transpose2d(data), result)

    def test_transposition_2x1(self):

        data = [[1, 2]]
        result = [[1], [2]]
        self.assertEqual(transpose2d(data), result)

    def test_transposition_2x2(self):

        data = [[1, 2], [3, 4]]
        result = [[1, 3], [2, 4]]
        self.assertEqual(transpose2d(data), result)

    def test_transposition_2x3(self):

        data = [[1, 2], [3, 4], [5, 6]]
        result = [[1, 3, 5], [2, 4, 6]]
        self.assertEqual(transpose2d(data), result)

    def test_transposition_3x2(self):

        data = [[1, 3, 5], [2, 4, 6]]
        result = [[1, 2], [3, 4], [5, 6]]
        self.assertEqual(transpose2d(data), result)


class TestWindowsArray(unittest.TestCase):
    '''
    A TestCase class to test the window1d function against a simple range(5) list with different
    windowing parameters denoted in the test title
    '''

    def test_window1d_size_3(self):

        data = range(5)
        expected_result = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        self.assertEqual(window1d(data, size=3), expected_result)

    def test_window1d_size_2_stride_2(self):

        data = range(5)
        expected_result = [[0, 2], [1, 3], [2, 4]]
        self.assertEqual(expected_result, window1d(data, size=2, stride=2))

    def test_window1d_size_2_stride_3(self):

        data = range(5)
        expected_result = [[0, 3], [1, 4]]
        self.assertEqual(expected_result, window1d(data, size=2, stride=3))

    def test_window1d_size_2_stride_4(self):

        data = range(5)
        expected_result = [[0, 4]]
        self.assertEqual(expected_result, window1d(data, size=2, stride=4))

    def test_window1d_size_3_shift_2(self):

        data = range(5)
        expected_result = [[0, 1, 2], [2, 3, 4]]
        self.assertEqual(window1d(data, size=3, shift=2), expected_result)

    def test_window1d_size_2_stride_2_shift_2(self):

        data = range(5)
        expected_result = [[0, 2], [2, 4]]
        self.assertEqual(window1d(data, size=2, shift=2, stride=2), expected_result)


class TestWindowsNDArray(unittest.TestCase):
    '''
    A TestCase class to test the window1d function against a simple range(5) numpy ndarray with
    different windowing parameters denoted in the test title
    '''

    def test_window1d_ndarray_size_3(self):

        data = array(range(5))
        expected_result = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        self.assertEqual(window1d(data, size=3), expected_result)

    def test_window1d_size_2_stride_2(self):

        data = array(range(5))
        expected_result = [[0, 2], [1, 3], [2, 4]]
        self.assertEqual(expected_result, window1d(data, size=2, stride=2))

    def test_window1d_size_2_stride_3(self):

        data = array(range(5))
        expected_result = [[0, 3], [1, 4]]
        self.assertEqual(expected_result, window1d(data, size=2, stride=3))

    def test_window1d_size_2_stride_4(self):

        data = array(range(5))
        expected_result = [[0, 4]]
        self.assertEqual(expected_result, window1d(data, size=2, stride=4))

    def test_window1d_size_3_shift_2(self):

        data = array(range(5))
        expected_result = [[0, 1, 2], [2, 3, 4]]
        self.assertEqual(window1d(data, size=3, shift=2), expected_result)

    def test_window1d_size_2_stride_2_shift_2(self):

        data = array(range(5))
        expected_result = [[0, 2], [2, 4]]
        self.assertEqual(window1d(data, size=2, shift=2, stride=2), expected_result)


class TestConvolution2d(unittest.TestCase):
    '''
    A TestCase class to test the convolution2d function
    '''

    def test_convolution2d_stride1(self):
        '''Let's test a 3x3 matrix, 2x2 kernel and default stride (= 1)'''

        input_matrix = array(range(1, 10)).reshape(3, 3)
        kernel = ones((2, 2))
        expected_result = [[12, 16], [24, 28]]
        self.assertTrue(array_equal(convolution2d(input_matrix, kernel), expected_result))

    def test_convolution2d_stride2(self):
        '''Let's test a 3x3 matrix, 1x1 kernel and stride equals 2'''

        input_matrix = array(range(1, 10)).reshape(3, 3)
        kernel = ones((1, 1))
        expected_result = [[1, 3], [7, 9]]
        self.assertTrue(array_equal(convolution2d(input_matrix, kernel, stride=2), expected_result))

    def test_convolution2d_4x4(self):
        '''Let's test a 4x4 matrix, 2x2 kernel'''

        input_matrix = array(range(1, 17)).reshape(4, 4)
        kernel = array(range(1, 5)).reshape(2, 2)
        expected_result = [[26, 36, 46], [66, 76, 86], [106, 116, 126]] # calculated with SciPy
        self.assertTrue(array_equal(convolution2d(input_matrix, kernel), expected_result))


if __name__ == '__main__':
    unittest.main()
