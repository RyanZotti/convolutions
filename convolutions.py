"""Convolutions and transposed convolutions."""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def mkdir(path):
    """
    Make a directory if it doesn't exist.

    Parameters
    ----------
    path : str
        The directory to make

    Returns
    -------
    None
    """
    full_path = os.path.join(os.getcwd(), path)
    if not os.path.exists(full_path):
        os.mkdir(full_path)


def reshape_kernel(kernel, n_rows_input_matrix, n_cols_input_matrix):
    """Reshape a kernel to perform convolution.

    Parameters
    ----------
    kernel : :obj:`np.array`
        A numpy array representing a kernel

    input_matrix : :obj:`np.array`
        A numpy array for an image

    Returns
    -------
    :obj:`np.array`
        The reshaped kernel matrix
    """
    n_output_rows = 1 + (n_rows_input_matrix - kernel.shape[0])
    n_output_columns = 1 + (n_cols_input_matrix - kernel.shape[1])
    n_input_matrix = n_rows_input_matrix * n_cols_input_matrix

    kernel_reshaped = np.zeros(shape=(n_output_rows * n_output_columns, n_input_matrix))

    for row_offset in range(n_output_rows * n_output_columns):
        for row in range(kernel.shape[0]):
            for col in range(kernel.shape[1]):
                kernel_reshaped[
                    row_offset,
                    row * n_output_rows + row_offset + col
                ] = kernel[row, col]

    return kernel_reshaped, n_output_rows, n_output_columns


def perform_convolution(input_matrix, kernel):
    """Perform a convolution with the given kernel on the given image.

    Parameters
    ----------
    input_matrix : :obj:`np.array`
        A numpy array for an image

    kernel : :obj:`np.array`
        A numpy array representing a kernel

    Returns
    -------
    :obj:`np.array`
        The convoluted output
    """
    # Reshape kernel and input for matrix multiplication
    kernel_reshaped, n_output_rows, n_output_columns = reshape_kernel(
        kernel, input_matrix.shape[0],
        input_matrix.shape[1]
    )
    input_matrix_reshaped = input_matrix.reshape(
        [input_matrix.shape[0] * input_matrix.shape[1], 1]
    )

    return np.matmul(
        kernel_reshaped,
        input_matrix_reshaped
    ).reshape([n_output_rows, n_output_columns])


def perform_transposed_convolution(input_matrix, kernel, n_rows_output_matrix,
                                   n_cols_output_matrix):
    """Perform a convolution with the given kernel on the given image.

    Parameters
    ----------
    input_matrix : :obj:`np.array`
        A numpy array for an image

    kernel : :obj:`np.array`
        A numpy array representing a kernel

    n_rows_output_matrix : int
        The number of rows in the output matrix

    n_cols_output_matrix : int
        The number of columns in the output matrix

    Returns
    -------
    :obj:`np.array`
        The transposed convoluted output
    """
    # Reshape kernel and input for matrix multiplication
    kernel_reshaped, _, _ = reshape_kernel(kernel, n_rows_output_matrix, n_cols_output_matrix)
    input_matrix_reshaped = input_matrix.reshape([input_matrix.shape[0] * input_matrix.shape[1], 1])

    return np.matmul(
        np.transpose(kernel_reshaped),
        input_matrix_reshaped
    ).reshape([n_rows_output_matrix, n_cols_output_matrix])


def play(image_path, num_convolutions):
    """Load an image and perform the convolutions and transposed convolutions.

    Parameters
    ----------
    image_path : str
        The path to the image file

    num_convolutions : int
        The number of convolutions (and transposed convolutions) to perform

    Returns
    -------
    None
    """
    i = 0

    # Set up folder
    mkdir("examples")
    output_folder = "examples/{}_{:02d}_convolutions".format(
        image_path.split(os.path.sep)[-1],
        num_convolutions
    )
    mkdir(output_folder)

    # Plot a given image
    image = ndimage.imread(image_path).mean(axis=2)
    plt.pcolor(image)
    plt.savefig(os.path.join(output_folder, "{:02d}_original_image.png".format(i)))

    # Define kernel
    kernels = []
    for _ in range(num_convolutions):
        kernels.append(np.random.randint(low=0, high=3, size=9).reshape([3, 3]))

    # Perform convolutions
    convolutions = [image]
    for j in range(num_convolutions):
        i += 1
        convolutions.append(perform_convolution(convolutions[-1], kernels[j]))
        plt.pcolor(convolutions[-1])
        plt.savefig(os.path.join(output_folder, "{:02d}_convolution_{:02d}.png".format(i, j+1)))

    # Perform transposed convolutions
    transposed_convolutions = [convolutions[-1]]
    for j in range(num_convolutions):
        i += 1
        transposed_convolutions.append(
            perform_transposed_convolution(
                transposed_convolutions[-1],
                kernels[-1 - j],
                convolutions[-2 - j].shape[0],
                convolutions[-2 - j].shape[1]
            )
        )
        plt.pcolor(transposed_convolutions[-1])
        plt.savefig(os.path.join(output_folder, "{:02d}_transposed_convolution_{:02d}.png".format(i, j+1)))


def parse_args():
    """Parse the CLI arguments.

    Parameters
    ----------
    None

    Returns
    -------
    dict
        The parsed arguments dictionary
    """
    parser = argparse.ArgumentParser(
        description='Perform convolutions and transposed convolutions.'
    )
    parser.add_argument("--image", dest="image", type=str, default="images/python.png",
                        help="the image to be convoluted")
    parser.add_argument("--num-convolutions", dest="num_convolutions", type=int, default=3,
                        help="the number of convolutions (and transposed convolutions) to perform")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    play(args.image, args.num_convolutions)


if __name__ == "__main__":
    main()
