import numpy as np
from pathlib import Path
from typing import Union


def save_array_text(array: np.ndarray, filepath: Union[str, Path], delimiter: str = ",") -> None:
    """Save a 1D or 2D NumPy array to a plain text file."""
    np.savetxt(filepath, array, delimiter=delimiter)


def load_array_text(filepath: Union[str, Path], delimiter: str = ",") -> np.ndarray:
    """Load a NumPy array from a plain text file."""
    return np.loadtxt(filepath, delimiter=delimiter)


def save_checkpoint(filepath: Union[str, Path], **arrays: np.ndarray) -> None:
    """Save one or more NumPy arrays to a binary checkpoint (.npz) file."""
    np.savez(filepath, **arrays)


def load_checkpoint(filepath: Union[str, Path]) -> dict[str, np.ndarray]:
    """Load NumPy arrays from a binary checkpoint (.npz) file as a dictionary."""
    with np.load(filepath) as data:
        return {key: data[key] for key in data.files}


if __name__ == "__main__":
    # Quick verification test
    sample_data = np.array([[1.0, 2.5, 3.0], [4.5, 5.0, 6.5]])

    # Test text I/O
    save_array_text(sample_data, "test_array.txt")
    loaded_txt = load_array_text("test_array.txt")
    assert np.allclose(sample_data, loaded_txt), "Text I/O mismatch!"

    # Test checkpoint I/O
    save_checkpoint("checkpoint.npz", weights=sample_data)
    loaded_ckpt = load_checkpoint("checkpoint.npz")
    assert np.allclose(sample_data, loaded_ckpt["weights"]), "Checkpoint I/O mismatch!"

    print("All NumPy I/O checks passed successfully.")
