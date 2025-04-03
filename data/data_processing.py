import numpy as np

def read_csv(file_path):
    """Read CSV data from file"""
    return np.genfromtxt(file_path, delimiter=',', dtype=str)

def split_data(data, shuffle=True):
    """Split data into training and test sets"""
    if shuffle:
        np.random.shuffle(data)
    split_idx = int(0.8 * len(data))
    return data[:split_idx], data[split_idx:]