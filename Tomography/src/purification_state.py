from Qutrit import*
class Purification:
    def __init__(self,rank: int, dimension: int, protocol: list = None, state: np.ndarray = None):
        """
        Args:
            state: (np.ndarray) quantum state.
            rank: (int) rank state r.
        """
        
        self.state = state
        self.rank = rank
        self.protocol = protocol
        self.dimension = dimension
        if self.state is None:
            self.state = self.r_rank_r(self.rank, self.dimension, "complex")
        else:
            self.state = state

    def value_matrix_information(self):
        """
        Find the eigenstate matrix information.
        """
        
        if self.protocol is None:
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
        else:
            m = len(self.protocol)
            matrix_x = np.zeros((m*2,3), complex)
            k = 0
            for i in self.protocol:
                for A in ([A01,A03]):
                    row = (i @ A @  np.array([[1],[1],[1]])).T
                    # # print(row)
                    # row = row/(np.sum(abs(row)**2))**0.5
                    # print(np.sum(abs(row)**2))
                    matrix_x[k] = row
                    k+=1


        c = self.purification_state()
        H, lambda_j = self.matrix_information(matrix_x, c, self.rank, np.full(m*2, 1))
        
        val_H, vec_H = np.linalg.eig(H) 

        
        return val_H

    def purification_state(self, state: np.ndarray = None, rank: int = None):
        """
        Purification of the quantum state in the form of Schmidt decomposition : |c> = sum (p_i)^0.5 * |e_i> * |c_i> 
        |c_i> - eigen state |c>.
        |e_i> - orthonormal basic states of the environment (lengths of rank). |e_i> = (0 , 0, ..., 1, ...) - vector with e_i[i] = 1.
        Args:
            state: (np.ndarray) quantum state.
            rank: (int) rank state.
        Return:
            pure_state: (np.ndarray) pure state.
        """
        if state is None:
            state = self.state
        if rank is None:
            rank = self.rank

        N = len(state)
        w, v = np.linalg.eig(state)
        vector_without_zeros = []
        value_without_zeros = []
        for i in range(N):
            if np.round(w[i],10) != 0.0:
                vector_without_zeros.append(v[:, i])
                value_without_zeros.append(w[i]**0.5)
        pure_state = 0
        for i in range(rank):
            e_i = np.zeros((rank,1))
            e_i[i] = 1
            pure_state += value_without_zeros[i] * (np.kron(e_i, (vector_without_zeros[i])[:,np.newaxis]))
        return pure_state
        
    @staticmethod
    def real_state(state: np.ndarray):
        """
        Made vector with only real coefficient. 
        |c> -> [
                real(c),
                imag(c)
        ]
        Args:
            state: (np.ndarray) quantum state
        Return:
            real_state: (np.ndarray) vetctor with only real coefficient
        """
        real_path = np.real(state)
        im_path = np.imag(state)
        real_state = np.concatenate([real_path, im_path])
        return real_state
    
    @staticmethod
    def real_matrix(matrix: np.ndarray):
        """
        Made matrix with only real coefficient.
        M -> [
            real(M), -imaginary(M)
            imaginary(M), real(M)
        ]
        Args:
            matrix: (np.ndarray) matrix
        Return:
            real_matrix: (np.ndarray) matrix with only real coefficient
        """
        real_path = np.real(matrix)
        im_path = np.imag(matrix)
        real_m = np.block([[real_path, -im_path],[im_path, real_path]])
        return real_m

    def matrix_information(self, X: np.ndarray, c: np.ndarray, rank: int, t: list):
        """
        Create matrix information.
        Args:
            X (np.ndarray): the measurement matrix.
            с (np.ndarray): the tomographed quantum state with complex coefficient.
            rank (int): rank the tomographed quantum state.
            t (list): list of measurement expositions
            l (list):
        Return:
            H (np.ndarray): matrix information.
        """
        N = X.shape[0]

        X_l = np.zeros((rank * N, rank * self.dimension), complex)
        L = [0]*N 

        lambda_j = [0]*N 

        c_real = self.real_state(c)
        for n in range(N):
            L_j = np.zeros((2 * rank * self.dimension, 2 * rank * self.dimension))
            for i in range(rank):
                e_i = np.zeros((rank,1))
                e_i[i] = 1
                X_l[n+i] = np.kron(np.conj(e_i.T), X[n])
                X_l_real = self.real_matrix(X_l[n+i])
                L_j += np.conj((X_l_real).T) @ X_l_real
            
            L[n] = L_j
            lambda_j[n] = (np.conj(c_real.T) @ L[n] @ c_real)[0][0]

        H = np.zeros((2 * rank * self.dimension, 2 * rank * self.dimension))
        for j in range(N):
            H += 2 * t[j]/lambda_j[j] * (L[j] @ c_real) @ np.conj(L[j] @ c_real).T

        return H, lambda_j
        
    def r_rank_r(self, r: int, n: int, type: str = "default"):
        """
        Method that generates a matrix with rank r and dimension n^2
        Args:
            r: rank new matrix
            n: dimension new matrix n^2
            type: type matrix 
        Returns:
            New matrix with rank r and dimension n^2
        """
        if type == "default":
            # Генерация матрицы без комплекчных чисел
            Q = np.random.randn(r, n)
        elif type == "complex":
            # Генерация матрицы с комплекчными числами
            real_part = np.random.rand(r, n)       # Реальная часть
            imaginary_part = np.random.rand(r, n)  # Мнимая часть
            Q = real_part + 1j * imaginary_part

        return (np.conj(Q.T) @ Q) / np.trace(np.conj(Q.T) @ Q)





# pure_class = Purification(rank= 3, dimension=3, protocol=start_protocol)
# print(np.round(np.real(pure_class.value_matrix_information()),30))
# np.random.seed(52)
# min_zero = 20
# min_zero_list = []
# start_protocol = [Gl_4(0), Gl_4(np.pi/8), Gl_4(3*np.pi/8), Gl_4(5*np.pi/8), Gl_4(7*np.pi/8)]
# # protocol_fed = [np.diag((1,1,1)),Gl_2(np.pi/8), Gl_2(np.pi/8) @ Gl_8(0)]
# pure_class = Purification(rank = 2, dimension = 3)
# val = list((np.real(np.round(pure_class.value_matrix_information(), 14))))
# nummb_zero = val.count(0)
# # print("\n",nummb_zero, val)
# # for i  in range(1000):
# #     pure_class = Purification(rank = 2, dimension = 3, protocol = start_protocol)
# #     val = list((np.real(np.round(pure_class.value_matrix_information(), 14))))
# #     # print(val)
# #     nummb_zero = val.count(0)
    
# #     if nummb_zero < min_zero:
# #         min_zero = nummb_zero
# #         min_zero_list.append(val)

# print(nummb_zero, np.round(sorted(val),3))
# print(min_zero, sorted(min_zero_list[-1]))