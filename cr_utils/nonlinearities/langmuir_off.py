import numpy as np


def forward(x: np.ndarray) -> np.ndarray:
    ind_negative = x < 0
    x = x / (1 + x)
    x[ind_negative] = 0
    return 1 - x


def reverse(r: np.ndarray) -> np.ndarray:
    # r = np.maximum(0.0, r)
    ind_over = r >= 1
    ind_under = r <= 0
    output = np.zeros(r.shape)
    output[ind_over] = 0
    output[ind_under] = np.infty
    ind_inrange = (~ind_over) & (~ind_under)
    output[ind_inrange] = 1.0 / r[ind_inrange] - 1.0
    return output


if __name__ == '__main__':
    print(reverse(np.array([-1, 2, 0.5, 1, 0.0001, 0.99999])))
