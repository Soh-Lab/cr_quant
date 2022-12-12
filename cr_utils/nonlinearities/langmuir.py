import numpy as np


def forward(x: np.ndarray) -> np.ndarray:
    ind_negative = x < 0
    x = x / (1 + x)
    x[ind_negative] = 0
    return x


def reverse(r: np.ndarray) -> np.ndarray:
    r = np.maximum(0.0, r)
    ind_over = r >= 1
    output = np.zeros(r.shape)
    output[ind_over] = np.inf
    output[~ind_over] = r[~ind_over] / (1 - r[~ind_over])
    return output


if __name__ == '__main__':
    print(reverse(np.array([-1, 2, 0.5, 1, 0.0001, 0.99999])))
