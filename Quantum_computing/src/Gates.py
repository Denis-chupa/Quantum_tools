from numpy import zeros, array, kron, tensordot, moveaxis, pi, exp, ones
from functools import reduce

class Gates:

    hadamar = array([[1, 1],[1, -1]], dtype=complex)/2**0.5
    paul_x = array([[0, 1], [1, 0]], dtype=complex)
    paul_y = array([[0, -1j], [1j, 0]], dtype=complex)
    paul_z = array([[1, 0], [0, -1]], dtype=complex)
    
    list_paul = [array([[1, 0], [0, 1]]), paul_x, paul_y, paul_z]

    T_gate = array([
        [1, 0],
        [0, exp(1j * pi/8)]
        ], dtype=complex)
    
    cnot = array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
        ], dtype=complex)

    swap = array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
        ])
    
    def unitary(self, T, R):
        return array([
            [T, R],
            [-R.conjugate(), T.conjugate()]
            ], dtype=complex )
    
    @ staticmethod
    def R_gate(k):
        return array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, exp(2j * pi / 2**k)]
        ])
    
    @classmethod
    def x_gate(cls, numb_qubit):
        list_x = [cls.paul_x] * numb_qubit
        return reduce(kron, list_x)

    @classmethod
    def y_gate(cls, numb_qubit):
        list_x = [cls.paul_y] * numb_qubit
        return reduce(kron, list_x)

    @classmethod
    def z_gate(cls, numb_qubit):
        list_x = [cls.paul_z] * numb_qubit
        result = reduce(kron, list_x)
        return result
    
    @staticmethod
    def QFT_matrix(qubits: int = 1):
        """
        Построение теоретической матрицы для квантового преобразования Фурье вида:
        1   1   1   1   1
        1   w   w^2 w^3 w^4
        1  w^2  w^4
        1  w^3
        где w = exp{2πi/2^n}, n - количество кубитов
        Args:
            qubits(int): количество кубитов
        Return:
            Теоретическая матрица
        """
        w = exp(2j * pi / (2<<(qubits-1)))
        m_qft = ones((2<<(qubits-1), 2<<(qubits-1)), dtype=complex)
        for i in range(1, 2<<(qubits-1)):
            for j in range(1, 2<<(qubits-1)):
                m_qft[i][j] = w**(i*j)
        return m_qft / (2<<(qubits-1))**0.5