import numpy as np


def gaussian_mean(kernel_points, center, bandwidth):
    weights = np.exp(-1 * np.linalg.norm((kernel_points - center) / bandwidth, axis=1))
    mean = np.array(np.sum(weights[:, None] * kernel_points, axis=0) / np.sum(weights), dtype=np.int64)
    return mean
