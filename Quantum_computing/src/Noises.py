from numpy import  random, cos, sin, eye, zeros, array_equal, linalg, linspace, exp, trace
from Gates import Gates, kron, array, pi
from itertools import product
from functools import reduce
from scipy.linalg import expm
from scipy import signal
from scipy.optimize import minimize

class Noise:

    s_0 = Gates.list_paul[0]
    s_x = Gates.list_paul[1]
    s_y = Gates.list_paul[2]
    s_z = Gates.list_paul[3]

    def x_noise(sigma_squared):
        """
        Генерирует шумовой оператор, представляющий собой X(δ)
        Args:
            sigma_squared: параметр уровня шума
        """
        sigma = (sigma_squared) ** 0.5
        delta = random.normal(loc=0, scale=sigma) 
        U = array([[cos(delta / 2), -1j * sin(delta / 2)],
                        [-1j * sin(delta / 2), cos(delta / 2)]])
        return U

    def krays_noise(prob_X):
        """
        С вероятностью prob_X генерирует шумовой оператор s_x,
        иначе возвращает единичную матрицу(2x2).
        """
        random_value = random.rand()
        if random_value < prob_X:
            return array([[0, 1],
                            [1, 0]])
        else:
            return eye(2)

    def depol_error(rho, list_index, noise:float=0.0):
        """
        Модель шума, в которой кубит с некоторой вероятностью 
        noise заменяется на полностью смешанное состояние (полностью случайное), а с вероятностью 
        1 − noise остается нетронутым.
        Args:
            noise: уровень шума(от 0 до 1).
            n(int): количество кубитов к которым применятеся деполяризующий шум.
        """
        n = len(list_index)
        coeff = noise / (4 ** n - 1)
        combinations = product(Gates.list_paul, repeat=n)
        kraus_ops = [reduce(kron, comb) for comb in combinations]
        Sup_oper = zeros((4**n, 4**n), dtype=complex)  # Инициализация супероператора
        for op in kraus_ops:
            Sup_oper += coeff * kron(op, op.conj())
        Sup_oper += (1 - 4**n * coeff) * eye((4**n))
        
        return Sup_oper

    def dephas_error(cls, rho, list_index, noise: float=0.0):
        """
        Модель шума описывает простейший случай фазового шума, где с вероятностью 
        noise кубит меняет фазу (фазовый переворот).
        Args:
            noise: уровень шума(от 0 до 1).
            n(int): количество кубитов к которым применятеся деполяризующий шум.
        """
        n = len(list_index)
        combinations = product([cls.s_0, cls.s_z], repeat=n)
        kraus_ops =[reduce(kron, comb) for comb in combinations]
        Sup_oper = zeros((4**n, 4**n), dtype=complex)  # Инициализация супероператора
        for op in kraus_ops:
            if not array_equal(op, eye((2**n))):
                Sup_oper += noise * kron(op, op.conj()) / (2**n -1)
        Sup_oper += (1 - noise) * eye((4**n))
        return  Sup_oper

    def damp_error(cls, rho, list_index, noise: float=0.0):
        """
        Модель шума описывает тип квантовой ошибки, при котором кубит теряет энергию, 
        переходя из возбуждённого состояния |1>  в основное |0>.
        Args:
            noise: уровень шума(от 0 до 1).
            n(int): количество кубитов к которым применятеся деполяризующий шум.
        """
        n = len(list_index)
        kraus_ops_for_1qubit = [0.5 * (cls.s_0 + cls.s_z) + 0.5 * (1-noise/n)**0.5 * (cls.s_0 - cls.s_z),  0.5 * noise/n**0.5 * (cls.s_x + 1j * cls.s_y)]
        combinations = product(kraus_ops_for_1qubit, repeat=n)
        kraus_ops =[reduce(kron, comb) for comb in combinations if comb != [cls.s_0]*n]
        Sup_oper = zeros((4**n, 4**n), dtype=complex)  # Инициализация супероператора
        for op in kraus_ops:
            Sup_oper += kron(op, op.conj())
        return Sup_oper

    def overrotation_error(cls, rho, list_index, type_gate, noise: float=0.0):
        """
        Модель шума описывает ошибку, возникающую при выполнении квантовых гейтов из-за неточного угла поворота 
        на Блоховской сфере.
        """
        
        if array_equal(type_gate, cls.s_x):
            overrotation = eye((2)) * cos(noise * pi / 4) + 1j * cls.s_x * sin(noise * pi / 4)
        elif array_equal(type_gate, cls.s_y):
            overrotation = eye((2)) * cos(noise * pi / 4) + 1j * cls.s_y * sin(noise * pi / 4)
        elif array_equal(type_gate, cls.s_z):
            overrotation = eye((2)) * cos(noise * pi / 4) + 1j * cls.s_z * sin(noise * pi / 4)
        elif array_equal(type_gate, Gates.T_gate):
            overrotation = eye((2)) * cos(noise * pi / 8) + 1j * cls.s_z * sin(noise * pi / 8)
        elif array_equal(type_gate, Gates.cnot):
            overrotation = eye((4)) * cos(noise * pi / 4) + 1j * cls.c_z * sin(noise * pi / 4)
        else:
            print("Unknown gate")
            return rho
        
        Sup_oper = kron(overrotation, overrotation.conj())
        return Sup_oper

    def rand_error_field(rho, list_index, noise: float=0):
        """
        """
        n = len(list_index)
        N_q = 2 << (n-1)  
        A = random.randn(N_q, N_q) + 1j * random.randn(N_q, N_q)   
        Q, R = linalg.qr(A)  # QR-разложение

        # Нормализуем элементы R в единичный круг
        R_normalized = R / max(abs(R))  # если надо строго |z| ≤ 1
        h = Q @ R_normalized 
        error = expm(-0.5j * noise * pi * (h + (h.T).conjugate()))
        Sup_oper = kron(error, error.conj())
        return Sup_oper

    def time_corr_f(rho, list_index, t):

        a = [-1.84534754,  0.8470944 ]
        b = [ 0.45212988, -1.40288053,  0.92906238]
        time = linspace(0, 100, 10001)
        y = signal.lfilter(b, a, time)
        U = array([[exp(1j*y[t]), 0], [0, exp(-1j*y[t])]])
        return kron(U, U.conj())

    def project_to_trace_preserving(self, chi):
        """
        Проекция χ-матрицы на подпространство trace-preserving операций.
        """
        dim = chi.shape[0]
        
        # Условие trace-preserving: Tr(chi * P) = 0 для всех P, кроме единичного
        def constraint(chi_vec):
            chi_mat = chi_vec.reshape((dim, dim))
            constraints = []
            for P in Gates.list_paul[1:]:  # Все кроме единичного
                constraints.append(trace(chi_mat @ (P / 2**0.5)))
            return array(constraints)
        
        # Минимизация расстояния до chi при выполнении условий
        res = minimize(lambda x: linalg.norm(x - chi.ravel())**2,
                    chi.ravel(),
                    constraints={'type': 'eq', 'fun': constraint})
        
        return res.x.reshape((dim, dim))

    def generate_random_operation_error(self, chi_0, noise):
        """
        Генерация шумовой χ-матрицы для random-operation error.
        Все операции считаются trace-preserving (включая измерения).
        """
        dim = chi_0.shape[0]
        
        # Генерация случайной эрмитовой матрицы H
        H = random.randn(dim, dim) + 1j * random.randn(dim, dim)
        H = (H + H.conj().T) / 2  
        H = H / linalg.norm(H)  
        
        # Добавляем шум
        chi_prime = chi_0 + noise * H
        
        # Проекция на trace-preserving операции (для всех операций)
        chi_double_prime = self.project_to_trace_preserving(chi_prime)
        
        # Коррекция положительной определённости
        eigvals = linalg.eigvalsh(chi_double_prime)
        lambda_min = min(eigvals)
        
        if lambda_min < 0:
            chi_triple_prime = chi_double_prime - lambda_min * eye(dim)
        else:
            chi_triple_prime = chi_double_prime
        
        # Нормировка
        max_eigval = max(linalg.eigvalsh(chi_triple_prime))
        chi = chi_triple_prime / max(1, max_eigval)
        
        return chi