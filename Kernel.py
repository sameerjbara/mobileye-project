import numpy as np
import scipy


class Kernel:
    """
    The kernel class represent kernel for convolution process on different images to detect objects in the image.
    """

    def __init__(self, threshold, kernel: np.array):
        """
        The constructor get a part of an image as a np.array that represent the requested kernel and
        normalizing it.
        :param threshold: The threshold for the normalization section.
        :param kernel: A NumPy ndarray that represent a kernel.
        """
        self.threshold = threshold
        self.kernel = kernel
        self.__normalize_kernel()

    def __normalize_kernel(self):
        """
        This function normalized the kernel. The normalization process change the sum of all
        the cells of the kernel to zero.
        :return: None
        """
        matrix_sum, to_divide = self.__calculate_matrix_sum()
        neg_value = -(matrix_sum / to_divide)
        self.__decrease_values(neg_value)

    def __calculate_matrix_sum(self):
        """
        This function calculates the sum of the cell of the kernel that meet the threshold and
        calculates the number of cells that not meet the threshold.
        :return: None
        """
        cells_in_threshold_sum = 0
        cells_not_in_threshold = 0

        for row in self.kernel:
            for number in row:
                if number > self.threshold:
                    cells_in_threshold_sum += number
                else:
                    cells_not_in_threshold += 1
        return cells_in_threshold_sum, cells_not_in_threshold

    def __decrease_values(self, neg_value):
        """
        This function put the negative value in the cells that not meet the threshold
        for changing the sum of the kernel to zero.
        :param neg_value: A negative value that consist of the sum of the cells of the kernel that meet
        the threshold devided by the number of cell that not meet the threshold.
        :return: None
        """
        for index_row, row in enumerate(self.kernel):
            for index_col, number in enumerate(row):
                if number <= self.threshold:
                    self.kernel[index_row][index_col] = neg_value

    def convolution(self, image: np.array):
        """
        The convolution function get an image and execute a convolution process with the kernel.
        :param image: An image to perform a convolution on it.
        :return: The image after the convolution process.
        """
        return scipy.signal.convolve(image.copy(), self.kernel, mode='same')
