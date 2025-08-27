import pytest
from purification_state import Purification 
import numpy as np


@pytest.mark.parametrize(
    "complex_matrix, expected_matrix",
    [
        (np.array([[1 + 1j]]), np.array([[1, -1], [1, 1]])),

        (
            np.array([[1 + 2j, 3 - 4j], [5 + 6j, 7 + 0j]]),
            np.array([[1, 3,  -2,  4],
                      [5,  7, -6,  0],
                      [2, -4,  1,  3],
                      [6,  0,  5,  7]])
        ),

        (np.array([[1 + 2j], 
                   [3 + 4j], 
                   [5 - 6j]]),
         np.array([[1, -2], 
                   [3, -4], 
                   [5,  6], 
                   [2,  1], 
                   [4,  3], 
                   [-6, 5]])),

        # Пример 4: Матрица 1x3
        (np.array([[1 + 0j, 0 - 2j, -3 + 4j]]),
        np.array([[ 1,  0, -3,  0,  2, -4], 
                [ 0, -2,  4,  1,  0, -3]])),

        # Пример 5: Матрица 2x3
        (np.array([[1 + 0j, 0 - 2j, -3 + 4j], 
                [0 + 5j, 6 + 0j, 7 - 8j]]),
        np.array([[ 1,  0, -3,  0,  2, -4], 
                [ 0,  6,  7,  -5,  0, 8], 
                [ 0, -2,  4,  1,  0, -3], 
                [ 5,  0, -8,  0,  6,  7]])),


        # Пример 6: Единичная матрица 2x2
        (np.array([[1 + 0j, 0 + 0j], 
                   [0 + 0j, 1 + 0j]]),
         np.array([[1,  0,  0,  0], 
                   [0,  1,  0,  0], 
                   [0,  0,  1,  0], 
                   [0,  0,  0,  1]])),

        # Пример 6: Единичная матрица 2x2
        (np.array([[5 + 0j, -2 + 0j, -2 + 0j], 
                   [6 + 0j, -4 + 0j, -2 + 0j]]),
         np.array([[5,  -2, -2, 0,  0,  0], 
                   [6,  -4, -2, 0,  0,  0], 
                   [0,  0,  0,  5,  -2, -2], 
                   [0,  0,  0,  6,  -4, -2]])),

        
        
    ]
)
def test_real_matrix(complex_matrix, expected_matrix):
    r_m = Purification.real_matrix
    result_matrix = r_m(complex_matrix)
    assert np.array_equal(result_matrix, expected_matrix)


@pytest.mark.parametrize("repeat", range(10))  # 10 повторений
def test_real_matrix_shape(repeat):
    # Генерация случайной комплексной матрицы
    rows = np.random.randint(1, 10)  # Случайное количество строк (1 до 10)
    cols = np.random.randint(1, 10)  # Случайное количество столбцов (1 до 10)
    
    real_range = (-10, 10)  # Диапазон для действительных чисел
    imag_range = (-10, 10)  # Диапазон для мнимых чисел
    
    # Создаём случайную комплексную матрицу
    real_part = np.random.uniform(real_range[0], real_range[1], (rows, cols))
    imag_part = np.random.uniform(imag_range[0], imag_range[1], (rows, cols))
    complex_matrix = real_part + 1j * imag_part

    # Вызываем функцию real_matrix
    result_matrix = Purification.real_matrix(complex_matrix)
    # print(f"len_rows_real_matr: {result_matrix.shape}")
    # Проверяем размерность
    assert result_matrix.shape == (2 * rows, 2 * cols)





    
