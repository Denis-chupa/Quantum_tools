import pytest
from purification_state import Purification 
import numpy as np







@pytest.mark.parametrize(
    "rank, dimension, start_matrix, teor_matrix",
    [
        # Пример 1: Ранг 3, размерность пространства 3
        (
            3, 3,
            np.array([
                [0.5, 0, 0],
                [0, 0.3, 0],
                [0, 0, 0.2]
            ]),
            np.array([
                [np.sqrt(0.5)], [0], [0],
                [0], [np.sqrt(0.3)], [0],
                [0], [0], [np.sqrt(0.2)]
            ])
        ),
        # Пример 2: Ранг 3, размерность пространства 4
        (
            3, 4,
            np.array([
                [0.4, 0, 0, 0],
                [0, 0.35, 0, 0],
                [0, 0, 0.25, 0],
                [0, 0, 0, 0]
            ]),
            np.array([
                [np.sqrt(0.4)], [0], [0], [0],
                [0], [np.sqrt(0.35)], [0], [0],
                [0], [0], [np.sqrt(0.25)], [0]])
        ),
        # Пример 3: Ранг 1, размерность пространства 2
        (
            1, 2,
            np.array([
                [1, 0],
                [0, 0]
            ]),
            np.array([[np.sqrt(1)], [0]])
        ),
        # Пример 4: Ранг 2, размерность пространства 2
        (
            2, 2,
            np.array([
                [0.6, 0],
                [0, 0.4]
            ]),
            np.array([
                [np.sqrt(0.6)], [0],
                [0], [np.sqrt(0.4)]
            ])
        ),
        # Новый пример 5: Ранг 1, размерность пространства 4
        (
            1, 4,
            np.array([
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]),
            np.array([
                [np.sqrt(1)], [0], [0], [0]
            ])
        )

    ]
)
def test_result(rank, dimension, start_matrix, teor_matrix):
    p = Purification(rank=rank, dimension=dimension)
    result_matrix = p.purification_state(start_matrix)
    assert np.array_equal(result_matrix, teor_matrix)


@pytest.mark.parametrize("repeat", range(10))
def test_rank_matrix(repeat):
    dimension = np.random.randint(1,10)
    if dimension == 1:
        rank = 1
    else:
        rank = np.random.randint(1,dimension)
    p = Purification(rank=rank, dimension=dimension)
    rho = p.r_rank_r(rank, dimension, "complex")
    N = len(rho)
    w, v = np.linalg.eig(rho)
    value_without_zeros = []
    for i in range(N):
        if np.round(w[i],10) != 0.0:
            value_without_zeros.append(w[i]**0.5)
    assert len(value_without_zeros) == rank 


@pytest.mark.parametrize("repeat", range(10))
def test_len_new_matrix(repeat):
    dimension = np.random.randint(1,10)
    if dimension == 1:
        rank = 1
    else:
        rank = np.random.randint(1,dimension)
    p = Purification(rank=rank, dimension=dimension)
    rho = p.r_rank_r(rank, dimension, "complex")
    result_matrix = p.purification_state(rho)
    rows = result_matrix.shape[0]
    assert rows == rank * dimension

@pytest.mark.parametrize("repeat", range(10))
def test_norm_new_matrix(repeat):
    dimension = np.random.randint(1,10)
    if dimension == 1:
        rank = 1
    else:
        rank = np.random.randint(1,dimension)
    p = Purification(rank=rank, dimension=dimension)
    rho = p.r_rank_r(rank, dimension, "complex")
    result_matrix = p.purification_state(rho)
    print( np.real(np.conj(result_matrix.T) @ result_matrix))
    assert np.round((np.conj(result_matrix.T) @ result_matrix),10) == 1.0