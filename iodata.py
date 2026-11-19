import numpy as np


def save_text(filename, array):
    """Save a NumPy array to a text file."""
    np.savetxt(filename, array)


def load_text(filename):
    """Load a NumPy array from a text file."""
    return np.loadtxt(filename)


def save_checkpoint(filename, array):
    """Save a NumPy array to a binary checkpoint (.npy) file."""
    np.save(filename, array)


def load_checkpoint(filename):
    """Load a NumPy array from a binary checkpoint (.npy) file."""
    return np.load(filename)
