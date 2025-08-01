from numpy import array, linalg, ones, conj, trace, zeros, column_stack, full, hstack, cos, sin, pi, diag
from scipy.linalg import sqrtm



def tl_4(Q):
  return (pow(cos(Q), 2) + 1j * pow(sin(Q), 2)) / (1j)**0.5

def rl_4(Q):
  return (((1 - 1j) * cos(Q) * sin(Q)) / (1j)**0.5)

def tl_2(Q):
  return (1j * cos(2 * Q))

def rl_2(Q):
  return (1j * sin(2 * Q))

def tl_8(Q):
    return cos(-pi / 8) + 1j * sin(-pi / 8) * cos(2 * Q)

def rl_8(Q):
    return 1j * sin(-pi / 8) * sin(2*Q)


#Матрица преобразование кутрита
def Gl_4(Q):
  M = array([[tl_4(Q)**2,                          2**0.5 * tl_4(Q) * rl_4(Q),                rl_4(Q)**2],
             [-2**0.5 * tl_4(Q) * conj(rl_4(Q)) ,  abs(tl_4(Q))**2 - abs(rl_4(Q))**2,         2**0.5 * conj(tl_4(Q)) * rl_4(Q)],
             [conj(rl_4(Q))**2,                    -2**0.5 * conj(tl_4(Q)) * conj(rl_4(Q)),   conj(tl_4(Q))**2]])
  return M

def Gl_2(Q):
  M = array([[tl_2(Q)**2,                          2**0.5 * (tl_2(Q)) * (rl_2(Q)),            pow(rl_2(Q),2)],
             [-2**0.5 * tl_2(Q) * conj(rl_2(Q)),   abs(tl_2(Q))**2 - abs(rl_2(Q))**2,         2**0.5 * conj(tl_2(Q)) * rl_2(Q)],
             [conj(rl_2(Q))**2,                    -2**0.5 * conj(tl_2(Q))* conj(rl_2(Q)),    conj(tl_2(Q))**2]])
  return(M)

#Матрица преобразование кубита
def U_4(Q):
  return array([[tl_4(Q), rl_4(Q)], [-conj(rl_4(Q)), conj(tl_4(Q))]])

def U_2(Q):
  return array([[tl_2(Q),rl_2(Q)], [-conj(rl_2(Q)), conj(tl_2(Q))]])

# пластинка l/8
def U_8(Q):
    return array([[tl_8(Q), rl_8(Q)], [-conj(rl_8(Q)), conj(tl_8(Q))]])

def Gl_8(Q):
  M = array([[tl_8(Q)**2,                            2**0.5 * tl_8(Q) * rl_8(Q),                  rl_8(Q)**2],
              [-2**0.5 * tl_8(Q) * conj(rl_8(Q)),     abs(tl_8(Q))**2 - abs(rl_8(Q))**2,          2**0.5 * conj(tl_8(Q)) * rl_8(Q)],
              [conj(rl_8(Q))**2,                      -2**0.5 * conj(tl_8(Q)) * conj(rl_8(Q)),    conj(tl_8(Q))**2]])
  return M

class tomography_pol_qutrit:
  def __init__(self, protokol_angles: list):

    self.angles = protokol_angles
    self.len_protocol = len(self.angles)
    self.A00 = [diag([1,0,0]), diag([0,1,0]), diag([0,0,1])]
    self.projectors = tomography_pol_qutrit.matrix_P(self)
    self.B = tomography_pol_qutrit.matrix_B(self)
    
    # отдельные состояние которые удобно использовать
    self.hh = array([[1], [0], [0]])
    self.hv = array([[0], [1], [0]])
    self.vv = array([[0], [0], [1]])
    self.rr = array([[1 / 2], [1j / (2)**0.5], [-1 / 2]])
    self.aa = array([[1 / 2], [-1 / (2)**0.5], [1 / 2]])
    self.dd = array([[1 / 2], [1 / (2)**0.5], [1 / 2]])
    self.ll = array([[1 / 2], [-1j / (2)**0.5], [-1 / 2]])

  
  def psi(self, r):
    """
    Превращение матрицы плотности в матрицу состояния кутрита.
    Args:
      r: матрица плотности.
    Return:
      Полученный вектор состояния.
    """
    S, V, D = linalg.svd(r, full_matrices = True, compute_uv = True)
    N = len(V)
    K = (ones((N, N)) + V - ones((N, N)))
    for i in range(0, N):
      for j in range(0, N):
        if i != j:
          K[i][j] = 0
    psi = S @ K**0.5  # убедиться что праильный корень
    return psi

  def density(self, psi):
    """
      Превращение матрицы состояния кутрита в матрицу плотности ρ = |ψ><ψ|.
    Args:
      psi: вектор состояния ψ.
    Return:
      Матрица плотности ρ.
    """
    return psi @ (conj(psi)).T
  
  def matrix_P(self):
    """
      Создание набора проекторов измерения для заданного протокола на базисные состояния HH, VV, HV.
    """
    A = ones((3 * self.len_protocol, 3, 3), dtype=complex)
    for i in range(3):
      for j in range(self.len_protocol):
        A[i * self.len_protocol + j] = array(conj(self.angles[j]).T @ self.A00[i] @ self.angles[j])
    return A
  
  def matrix_B(self):
      """
        Создание матрицы протокольных измерений. Где каждая строчка - это
      проекторы измерения для заданного протокола на базисные состояния HH, VV, HV.
      """
      B = ones((3 * self.len_protocol, 9), dtype=complex)
      for index, el in enumerate(self.projectors):
          B[index] = array(el.flatten())
      return B

  def Fidelity(self, r, r_t):
     """
     Точность двух состояний по Ульману.
     Args:
        r: матрица плотности первого состояния. 
        r_t: матрица плотности второго состояния.
     Return:
        Мера близости двух состояний.
     """
     return (trace(sqrtm(sqrtm(r) @ r_t @ sqrtm(r))))**2

  def psevdoin(self, p, rank: int=1):
    """
      Выполняет метод псевдоинверсии для правила Борна p = B @ R, где p, R - столбцы вероятностей и элементов
    матрицы плотности соответственно, B - матрица протокольных измерений. При этом B представляется через SVD разложение.
    Args:
      p: вектор вероятностей
      rank(int): ранг позвращаемой матрицы плотности.
    Return:
      Нормированная матрица плотности с рангом rank.
    """
    R = linalg.pinv(self.B) @ p
    R = array([[R[0][0], R[3][0], R[6][0]],[R[1][0], R[4][0], R[7][0]],[R[2][0], R[5][0], R[8][0]]])
    D1, V1 = linalg.eigh(R)
    N = len(D1)
    
    # зануляются отрицательные собственные значения
    for j in range(N):
          if D1[j] < 0:
            D1[j] = 0

    D1 = sorted(D1, reverse = True)
    D1 = D1 / linalg.norm(D1)
    matrix = zeros((N,N))

    # создаётся матрица на диагонали которой собственные значения по убыванию
    for j in range(N):
      for i in range(N):
        if (i == j):
          if D1[i] < 0:
            matrix[i][j] = 0
          else:
            matrix[i][j] = D1[i]

    D1 = matrix
    #  Расчёт V1
    V11 = V1[:, 0]
    V12 = V1[:, 1]
    V13 = V1[:, 2]
    #  V0 = np.array([[0],[0],[0]])
    V1 = column_stack([V13, V12, V11])
    R = V1.dot(D1)
    PSI = self.psi(R)
    PSI = PSI[:, :rank]
    R = self.density(PSI)
    R = R / trace(R)
    return R

  def result(self, r0, k, sigma, epsilon, max_iter=1000, alpha=0.5):
        """
          Решение уравнения A(psi) @ psi = Q @ psi  методом простой итерации, где
        psi_(i+1) = (1 - a) * Q^(-1) @ A @ psi_i + a * psi_i. В нашем случае 
        A = Σ(k_j / p_j) * P_j, Q = Σσ_j * P_j
        Args:
          p(list): экспериментальные вероятности измерений
          r0: начальное приближение psi_0
          epsilon: точность метода
          k: частоты получаемые в эксперименте
          sigma: экспериментальные отклонения для соотношений между HH, HV, VV
          max_iter: максимальное количество сделанных итераций 
          alpha: коэффициент схождения метода.
        """

        N = 3 * self.len_protocol

        #Создание матрицы Q
        Q=0
        for j in range(0,N):
          Q += (sigma[j//self.len_protocol]) * self.projectors[j]

        #Метод простых итераций
        Psi0 = self.psi(r0)
        for i in range(max_iter):
        #Создание матрицы А
          A = 0
          for j in range(0, N):
            if k[j] == 0:
              A = A
            else:
              A += (k[j] / trace(self.projectors[j] @ self.density(Psi0))) * self.projectors[j]

          Psi1 = (1 - alpha) * ((linalg.inv(Q)) @ A @ Psi0) + alpha * Psi0
          if abs(linalg.norm(Psi0) - linalg.norm(Psi1)) < epsilon:
            break
        
          Psi0 = Psi1

        Rx = self.density(Psi0)
        Rx = Rx / trace(Rx)
        return Rx

  def experiment(self, k, start_state, sigma1, sigma2, sigma3, rank: int=1,\
                 epsilon: float=1.0e-11, max_iter: int=1000, alpha: float=0.5, visible: bool=True):
      """
      Восстановление матрицы плотности по экспериментальным частотам.
      Args:
        k(float): экспериментальные частоты.
        start_state: идеальное состояние.
        sigma1(float): частоты для HH.
        sigma2(float): частоты для HV.
        sigma3(float): частоты для VV.
        rank(int): ранг восстанавлиемой матрицы плотности.
        epsilon: точность метода итераций.
        max_iter: максимальное число итераций в методе итераций.
        alpha: параметр метода итераций.
        visible(bool): параметр при котором показывается фиделити между идеальным и восстановленным состоянием.
      Return:
        Fidelity по ульману между идеальной матрицей и матрицей псевдоинверсии и между идеальной матрицей и восстановленной
      матрицей.
      """

      # предобработка
      self.start_density = self.density(start_state)
      ph = k[::3] / sigma1
      phv = k[1::3] / sigma2
      pv = k[2::3] / sigma3
      k = list(k[::3]) + list(k[1::3]) + list(k[2::3])
      p = array(list(ph) + list(phv) + list(pv))
      self.matrix_psevdoin = self.psevdoin(p.reshape(3 * self.len_protocol, 1), rank=rank)                               #Нахождение матрицы плотности с помощью псевдоинверсии
      self.matrix_finish = self.result(self.matrix_psevdoin, k, [sigma1,sigma2,sigma3], epsilon, max_iter, alpha)      #Полученная матрицы с помощью метода простых итераций

      if visible == True:
          print("Fidelity between start matrix density and matrix pseudoinversion =", abs(self.Fidelity(self.matrix_psevdoin, self.start_density)))
          print("Finish fidelity =", (abs(self.Fidelity(self.matrix_finish, self.start_density))))