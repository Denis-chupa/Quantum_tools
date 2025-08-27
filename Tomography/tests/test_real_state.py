import pytest
from purification_state import Purification 
import numpy as np


@pytest.mark.parametrize(
        "comlex_matrix, teor_matrix",
        [
            # Простой случай: вектор из одного элемента
            (np.array([[1+1j]]), np.array([[1], [1]])),

            # Вектор из двух элементов
            (np.array([[1+1j], [2+2j]]), np.array([[1], [2], [1], [2]])),

            # Вектор с нулями и отрицательными числами
            (np.array([[0+0j], [-1+1j]]), np.array([[0], [-1], [0], [1]])),

            # Вектор из трех элементов с разными значениями
            (np.array([[1+2j], [2-3j], [-3-4j]]), np.array([
                [1], [2], [-3], [2], [-3], [-4]
            ])),

            # Вектор с реальными числами (без мнимых частей)
            (np.array([[1], [2], [3]]), np.array([
                [1], [2], [3], [0], [0], [0]
            ]))
        ]        
        )

def test_check_result(comlex_matrix, teor_matrix):
    r_s = Purification.real_state

    assert all(r_s(comlex_matrix) == teor_matrix)


@pytest.mark.parametrize("repeat", range(100))  # 10 повторений
def test_len(repeat):
    rows = np.random.randint(1, 20)
    cols = 1
    real_range = (-10, 10)
    imag_range = (-10, 10)

    real_part = np.random.uniform(*real_range, (rows, cols))
    imag_part = np.random.uniform(*imag_range, (rows, cols))
    complex_matrix = real_part + 1j * imag_part

    r_s = Purification.real_state
    len_rows_comp_matr = complex_matrix.shape[0]
    len_rows_real_matr = r_s(complex_matrix).shape[0]
    # print(f"len_rows_real_matr: {len_rows_real_matr, 2*len_rows_comp_matr}")

    assert len_rows_real_matr == 2 * len_rows_comp_matr





    
