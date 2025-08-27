import pytest
from purification_state import Purification 
import numpy as np







@pytest.mark.parametrize(
    "rank, dimension",
    [
        (1,3),
        (2,3),
        (3,3)
    ]
)


def test_2n(rank, dimension):
    p = Purification(rank=rank, dimension=dimension)
    c = p.purification_state()
    m = 9
    matrix_x = np.array([
        [1/2**0.5, 0, 0],
        [0, 1/2, 0],
        [0, 0, 1/2**0.5],
        [0, 1/(2*2**0.5), -1j/2],
        [0, 1/(2*2**0.5), -1/2],
        [1/2, -1/((2*2**0.5)), 0],
        [1/2, -1j/((2*2**0.5)), 0],
        [1/((2*2**0.5)), 0, 1j/((2*2**0.5))],
        [1/((2*2**0.5)), 0, -1/((2*2**0.5))]
    ])
    H, lambda_j = p.matrix_information(matrix_x, c, rank, np.full(m*3, 1))
    c_real = p.real_state(c)
    assert np.round(np.conj(c_real.T) @ H @ c_real,10) == np.round(2*(np.sum(lambda_j)),10)


# @pytest.mark.parametrize("repeat", range(10))
# def test_rank_matrix(repeat):
#     dimension = np.random.randint(1,10)
#     if dimension == 1:
#         rank = 1
#     else:
#         rank = np.random.randint(1,dimension)
#     p = Purification(rank=rank, dimension=dimension)
#     rho = p.r_rank_r(rank, dimension, "complex")
#     N = len(rho)
#     w, v = np.linalg.eig(rho)
#     value_without_zeros = []
#     for i in range(N):
#         if np.round(w[i],10) != 0.0:
#             value_without_zeros.append(w[i]**0.5)
#     assert len(value_without_zeros) == rank 
