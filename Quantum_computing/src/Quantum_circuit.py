from numpy import zeros, array, kron, tensordot, moveaxis, trace, log2, hstack  
from scipy.linalg import sqrtm
from Gates import*   

class My_quantum_circuit(Gates):
    def __init__(self, num_qubits: int = 1, num_clbits: int = 0):
        """
        Initialization of parameters for generating a quantum circuit.
        Args:
            num_qubits (int): the number of qubits.
            num_clbits (int): the number of classical bits.
        """

        self.num_qubits = num_qubits
        # self.num_clbits = num_clbits
        self.data_gate = []    
        self.start_state = zeros(2**self.num_qubits)
        self.start_state[0] = 1
        self.state = self.start_state.reshape([2] * self.num_qubits)

    def add_x(self, list_index_qubits):
        self.data_gate.append([list_index_qubits, Gates.x_gate])

    def add_y(self, list_index):
        self.data_gate.append([list_index, Gates.y_gate])

    def add_z(self, list_index):
        self.data_gate.append([list_index, Gates.z_gate])

    def add_custom_gate(self, list_index, matrix_gate):
        self.data_gate.append([list_index, matrix_gate])

    def apply_1q(self, list_index, gate, moveaxis_use: bool = True):
        """
        Применяет однокубитные операции (гейты) к чистому состоянию квантовой системы.
        Args:
            gate: матрица однокубитной операции.
            list_index: список индексов кубитов, к которым нужно применить операцию.
            moveaxis(bool): флаг, указывающий нужно ли переставлять оси после применения операции.
        Return:
            state: cостояние после действия однокубитных операций.
        """

        if not callable(gate):
            matrix_gate = gate
        for k in list_index:
            if callable(gate):
                matrix_gate = gate()
            self.state = tensordot(matrix_gate, self.state, axes=[[1],[k]])
            if moveaxis_use:
                self.state = moveaxis(self.state, 0, k)

    def apply_2q(self, list_index, gate, moveaxis_use: bool = True):
        """
        Применение двухкубитного гейта.
        Args:
            state: начальное чистое состояние(вектор).
            index: пара индексов, с номерами кубитов к которым применяется гейт.
            gate: матрица гейта.
        Return:
            Состояние после применения гейта.
        """
        
        if not callable(gate):
            matrix_gate = gate
        for k, j in list_index:
            self.state = tensordot(matrix_gate, self.state, axes=[[2,3],[k, j]])
            if moveaxis_use:
              self.state = moveaxis(self.state, [0, 1], [k, j])

    def apply_q_gate_for_matr_density(matrix_density, U, list_index, moveaxis=True):
        """
        Применяет супероператор для произвольного числа кубитов к матрице плотности.
        Args:
            matrix_density: входная матрица плотности 
            U: оператор который применяется к матрице плотности
            list_index: список индексов или кортежей индексов, для которых применяем оператор
            moveaxis: флаг для использования moveaxis
        Return: 
            Матрица плотности после пременения оператора
        """
        
        if not callable(U):
            matrix_U = U
        else:
            matrix_U = U()
        N = int(log2(matrix_density.shape[0]))
        n_qubits = int(log2(matrix_U.shape[0]) // 2)
        matrix_density = matrix_density.reshape([2] * (2 * N))
        N_q = 2 << (N-1)
        matrix_U = matrix_U.reshape([2] * (4 * n_qubits))
        for indices in list_index:
            if callable(U):
                matrix_U = U()
                matrix_U = matrix_U.reshape([2] * (4 * n_qubits))

            indices = array(indices)
            qubit_axes = hstack((indices, N + indices))
            # Применяем оператор
            matrix_density = tensordot(matrix_U, matrix_density, axes=[list(range(-2*n_qubits, 0)), qubit_axes])
            # Если флаг moveaxis установлен, применяем moveaxis
            if moveaxis:
                matrix_density = moveaxis(matrix_density, list(range(2*n_qubits)), qubit_axes)

        return matrix_density.reshape((N_q, N_q))

    def apply_QFT(self, list_index_qubits):
        '''
        Применение к гейтовой схеме квантового преобразования Фурье
        Args:
            list_index_qubits: список кубитов к которым применяется квантовое преобразование Фурье
        '''
        len_index = len(list_index_qubits)
        for i in list_index_qubits:
            self.apply_1q([i], Gates.hadamar)
            for j in range(i+1, len(list_index_qubits)):
                self.apply_2q([[list_index_qubits[j], list_index_qubits[i]]], Gates.R_gate(j - i + 1).reshape(2,2,2,2))

        for index, i in enumerate(list_index_qubits[: (len_index//2)]):
            self.apply_2q([[list_index_qubits[index], list_index_qubits[-(index+1)]]], Gates.swap.reshape(2,2,2,2))
    
    def apply_reverse_QFT(self, list_index_qubits):
        '''
        Применение гейтовой схемы для обратного квантового преобразования Фурье
        Args:
            qubits: количество кубитов в схеме
            state: состояние подающееся на вход гейтовой схемы
        Return:
            Состояние после применения обратного квантового преобразования Фурье
        '''

        len_index = len(list_index_qubits)
        for index, i in enumerate(list_index_qubits[: (len_index//2)]):
            self.apply_2q([[list_index_qubits[index], list_index_qubits[-(index+1)]]], Gates.swap.reshape(2,2,2,2))

        state = self.apply_1q([len_index-1], Gates.hadamar)
        for i in reversed(range(len_index-1)):
            for j in reversed(range(i+1, len_index)):
                state = self.apply_2q([[list_index_qubits[j], list_index_qubits[i]]], Gates.R_gate(j - i + 1).reshape(2,2,2,2))        
            state = self.apply_1q([list_index_qubits[i]], Gates.hadamar)

        return state
    
    @staticmethod
    def Fidelity(matrix_1, matrix_2):
        """
        Фиделити по Ульману с учётом частных случ
        """
        len_m_1 = len(matrix_1.shape)
        len_m_2 = len(matrix_1.shape)
        if len_m_1 == 1 and len_m_2 == 1:
            return abs(matrix_1 @ matrix_1.conjugate())
        elif len_m_1 == 1 and len_m_2 > 1:
            pass
        elif len_m_1 > 1 and len_m_2 == 1:
            pass
        elif len_m_1 > 1 and len_m_2 > 1:
            return (trace(sqrtm(sqrtm(matrix_1) @ matrix_2 @ sqrtm(matrix_1))))**2
        # if  
        
