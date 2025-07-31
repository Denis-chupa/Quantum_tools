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
    self.A = tomography_pol_qutrit.matrix_A(self)
    self.B = tomography_pol_qutrit.matrix_B(self)
    self.hh = array([[1], [0], [0]])
    self.hv = array([[0], [1], [0]])
    self.vv = array([[0], [0], [1]])
    self.rr = array([[1 / 2], [1j / (2)**0.5], [-1 / 2]])
    self.aa = array([[1 / 2], [-1 / (2)**0.5], [1 / 2]])
    self.dd = array([[1 / 2], [1 / (2)**0.5], [1 / 2]])
    self.ll = array([[1 / 2], [-1j / (2)**0.5], [-1 / 2]])

  # Превращение матрицы плотности в матрицу состояния кутрита
  def psi(self, r):
    S, V, D = linalg.svd(r, full_matrices = True, compute_uv = True)
    N = len(V)
    K = (ones((N, N)) + V - ones((N, N)))
    for i in range(0, N):
      for j in range(0, N):
        if i != j:
          K[i][j] = 0
    psi = S @ K**0.5  # убедиться что праильный корень
    return psi

  # Превращение матрицы состояния кутрита в матрицу плотности
  def density(self, psi):
    return psi @ (conj(psi)).T

  # Вспомогательные функции
  def kh(self, k):
      k0 = 1j * ones(self.len_protocol)
      for i in range(self.len_protocol):
        k0[i] = k[3 * i]
      return k0
  
  def khv(self, k):
      k0 = 1j * ones(self.len_protocol)
      for i in range(self.len_protocol):
        k0[i] = k[3 * i + 1]
      return k0
  
  def kv(self, k):
      k0 = 1j * ones(self.len_protocol)
      for i in range(self.len_protocol):
        k0[i] = k[3 * i + 2]
      return k0
  
  def SUMK(self, k1, k2, k3):
      return array(list(k1) + list(k2) + list(k3))
  
  def SUMK1(self, k1, k2, k3):
      k0 = 1j * ones(3 * self.len_protocol)
      for i in range(self.len_protocol):
        k0[3 * i : 3 * i + 1] = [k1[i], k2[i], k3[i]]
      return k0
  
  def matrix_A(self):
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
      for index, el in enumerate(self.A):
          B[index] = array(el.flatten())
      return B

  def Fidelity(self, r, r_t):
     return (trace(sqrtm(sqrtm(r) @ r_t @ sqrtm(r))))**2

  # Мeтод псевдоинверсии
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

  # Метод простых итерций
  def result(self, p, P, r0, k=[0,0], epsilon: float=1.0e-11, sigma=[0,0], max=1000, alpha=0.5):
        
        N = len(p)
        if (sigma == full(N,0)).all():
          sigma = full(N,7000)
        if (k == full(N,0)).all():
          k = sigma * p

        #Создание матрицы Q
        Q=0
        i=0
        for j in range(0,N):
          Q=Q+(sigma[j])*P[j]

        #Метод простых итераций
        Psi0 = self.psi(r0)
        for i in range(max):
        #Создание матрицы А
          A = 0
          for j in range(0, N):
            if k[j] == 0:
              A = A
            else:
              A += (k[j] / trace(P[j] @ self.density(Psi0))) * P[j]

          Psi1 = (1 - alpha) * ((linalg.inv(Q)) @ A @ Psi0) + alpha * Psi0
          if abs(linalg.norm(Psi0) - linalg.norm(Psi1)) < epsilon:
            break
          
          i += 1
          Psi0 = Psi1

        Rx = self.density(Psi0)

        Rx = Rx / trace(Rx)
        return Rx

  #Обработка эксперимента
  def experiment(self, k, start_state, sigma1, sigma2, sigma3, rank: int=1, epsilon: float=1.0e-11, visible = True):

      sigma = hstack([sigma1,sigma2,sigma3])

      # предобработка
      self.start_density = self.density(start_state)
      ph = self.kh(k) / sigma1[1]
      phv = self.khv(k) / sigma2[1]
      pv = self.kv(k) / sigma3[1]
      k1 = ph * sigma1[1]
      k2 = phv * sigma2[1]
      k3 = pv * sigma3[1]
      k = self.SUMK(k1, k2, k3)
      p = self.SUMK(ph, phv, pv)
      r0 = self.psevdoin(p.reshape(3 * self.len_protocol, 1), rank=rank)      #Нахождение матрицы плотности с помощью псевдоинверсии
      Rx = self.result(p, self.A, r0, k, epsilon, sigma)                     #Полученная матрицы с помощью метода простых итераций

      self.matrix_psevdoin = r0
      self.matrix_finish = Rx
      if visible == True:
          print("Fidelity between start matrix density and matrix pseudoinversion =", abs(self.Fidelity(r0, self.start_density)))
          print("Finish fidelity =", (abs(self.Fidelity(Rx, self.start_density))))