"""
ACT WITH OUR MAXIMUM LIKLEHOOD
"""
from Qutrit import*
import cvxpy as cp
import matplotlib.pyplot as plt 
from scipy.linalg import sqrtm
from scipy.optimize import fsolve
from scipy.optimize import least_squares, Bounds
from tqdm import tqdm
import json
from Tomography_qutrit import tomography_pol_qutrit

class ACT:
    def __init__(self, protocol : list, r : int, n : int):
      """
      Initializarion paramerts for adaptive compressive tomography.
      Args:
          protocol (list): in the protocol, operators are written in groups of pvm components,
          i.e. if you add up all the components protocol[0] = I
          r (int): the rank for which we want to tomography the quantum state.
          n (int): dimension (maximum rank).
          epsilon (np.float16): convergence accuracy in ICC for s_cvx. 
      """

      self.r = r                  
      self.n = n                  
      self.protocol = protocol    
      self.len_protocol = len(self.protocol)

      # self.A01 = np.array([[1,0,0],[0,0,0],[0,0,0]])
      # self.A02 = np.array([[0,0,0],[0,1,0],[0,0,0]])
      # self.A03 = np.array([[0,0,0],[0,0,0],[0,0,1]])
      # self.A00 = [A01, A02, A03]
      

    # Calculation of fidelity according to Ulman
    Fidelity = lambda self, r, r_t: abs(np.trace(sqrtm(sqrtm(r) @ r_t @ sqrtm(r))))**2  
    
    # The Von Neumann entropy
    neumann_entoropy = lambda self, r: -np.trace(r*np.log(r))

    # Normalization of a function by the sum of squares                      
    norm_sum_squares = lambda self, x: x/(np.sum(abs(x)**2))**0.5                    
    
    def main(self , rank_psevdoin: int = None, type_ml: str = "default", Z : np.ndarray = None, random_r : np.ndarray = None,\
              type_solve_semidefinite_program = None, epsilon : np.float16 = 10**-5, max_iters_in_semidefinite_program = 10**7):
        
        self.epsilon = epsilon
        self.max_iters_in_semidefinite_program = max_iters_in_semidefinite_program
        if rank_psevdoin is None:
          rank_psevdoin = self.r

        if Z is None:
          self.Z = self.r_rank_r(self.n, self.n, "complex")
        else:
          self.Z = Z

        if random_r is None:
          self.random_r = self.r_rank_r(self.r, self.n, "complex")
        else:
          self.random_r = random_r

        if type_solve_semidefinite_program is None:
          self.solve_semidef = cp.SCS
        else:
          self.solve_semidef = type_solve_semidefinite_program
          

        v = []
        k = 0
        svx = 1
        svx_list = [0] * self.len_protocol
        x_min_list = [0] * self.len_protocol
        x_max_list = [0] * self.len_protocol
        R_list = [0] * self.len_protocol
        fidelity_list = [0] * self.len_protocol
        fidelity_x_max_list = [0] * self.len_protocol
        fidelity_x_min_list = [0] * self.len_protocol
        projectors = []
        
        while k <  self.len_protocol:

          ml = tomography_pol_qutrit(self.protocol[ : (k + 1)])

          projectors = projectors + list([ml.projectors[k], ml.projectors[(ml.len_protocol) + k], ml.projectors[2 * (ml.len_protocol) + k]])
          for i in projectors[3 * k :]:
              v.append(np.real(np.trace(i @ self.random_r)))
          
          if type_ml == "default":
            r0 = ml.psevdoin(np.array(v)[:, np.newaxis], rank = self.n) #Нахождение матрицы плотности с помощью псевдоинверсии
            # R = self.ml(v,start_protocol,r0,v)                                    #Полученная матрицы с помощью метода простых итераций
            R = ml.result(r0, v, [100, 100, 100], epsilon=10**-11)       
            R_list[k] = [[str(item) for item in row] for row in R.tolist()]
            fidelity_list[k] = self.Fidelity(R, self.random_r)
            probability = []
            for i in range(len(projectors)):
                  probability.append(np.trace(projectors[i] @ R))
          elif type_ml == "without_ml":
            probability = v
          
          self.f_max_0, x_max =  self.semidefinite_program(list([ml.projectors[0], ml.projectors[ml.len_protocol], ml.projectors[2 * ml.len_protocol]]), probability[:3], "maximize") # задаю max(f) на нулевом шаге, f = tr{XZ}
          self.f_min_0, x_min =  self.semidefinite_program(list([ml.projectors[0], ml.projectors[ml.len_protocol], ml.projectors[2 * ml.len_protocol]]), probability[:3], "minimize") # задаю min(f) на нулевом шаге, f = tr{XZ}
          
          semi_max, x_max = self.semidefinite_program(projectors, probability, "maximize")
          semi_min, x_min = self.semidefinite_program(projectors, probability, "minimize")
          svx = (semi_max - semi_min)/(self.f_max_0-self.f_min_0)

          
          x_min_list[k] = [[str(item) for item in row] for row in x_min.tolist()]
          x_max_list[k] = [[str(item) for item in row] for row in x_max.tolist()]
          fidelity_x_max_list[k] = self.Fidelity(x_max, self.random_r)
          fidelity_x_min_list[k] = self.Fidelity(x_min, self.random_r)
          if abs(svx) == np.inf:
             svx_list[k] = svx
             return svx_list, fidelity_list
          svx_list[k] = svx
          k+=1

        return svx_list, fidelity_list, fidelity_x_min_list, fidelity_x_max_list, x_min_list, x_max_list, R_list
    
    def r_rank_r(self, r: int, n: int, type: str = "default"):
       """
       Method that generates a matrix with rank r and dimension n^2
       Args:
        r: rank new matrix
        n: dimension new matrix n^2
        type: type matrix 
       Returns:
        New matrix with rank r and dimension n^26
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
    
    def matrix_B(self, protocol, l):
      """
      Creating a measurement matrix (the measurement projectors are stretched out in the rows)
      """
      B = 1j*np.ones((3 * (l + 1), 9))
      k = 0
      for i in range(3):
        for j in range(l + 1):
          B[k] = np.array((np.conj(protocol[j]).T @ self.A00[i] @ protocol[j]).flatten())
          k+=1
      return B
    
    def psi(self, r: int):
      """  
      Calculates the state vector knowing the density matrix
      """
      S,V,D = np.linalg.svd(r,full_matrices = True, compute_uv = True)
      N=len(V)
      K=(np.ones((N,N))+V-np.ones((N,N)))
      for i in range(0,N):
        for j in range(0,N):
          if i!=j:
            K[i][j]=0
      psi=np.dot(S,np.sqrt(K))
      return psi
    
    def density(self, psi: np.ndarray):
      """ 
      Calculates the density matrix knowing the state vector
      """
      return np.dot(psi,np.transpose(np.conj(psi)))
   
    def semidefinite_program(self, A, b, mode: str):
        """
        Searches for the maximum or minimum value of s_cvx
        Args:
          A :
          b :
          mode : (str) 
        """        
        X = cp.Variable((self.n,self.n), complex=True)  
        # The operator >> denotes matrix inequality.
        constraints = [X >> 0]
        constraints += [X == cp.conj((X).T)]
        # constraints += [cp.trace(X) == 1]  # added 23.11 возможно избыточно.
        constraints += [cp.trace(A[i] @ X) == b[i] for i in range(len(b))]
        
        if mode == "maximize":
          prob = cp.Problem(cp.Maximize(cp.trace(cp.real(self.Z @ X))),
                          constraints)
        elif mode == "minimize":
          prob = cp.Problem(cp.Minimize(cp.trace(cp.real(self.Z @ X))),
                          constraints)
        else:
          print("unknow mode")

        # prob.solve(max_iters = self.max_iters_in_semidefinite_program)
        # prob.solve(solver=cp.MOSEK)
        # prob.solve(solver=cp.ECOS)
        if self.solve_semidef == cp.MOSEK:
          solver_opts = {
              "MSK_IPAR_INTPNT_MAX_ITERATIONS": self.max_iters_in_semidefinite_program  # Set max iterations for interior-point method
          }
          prob.solve(solver=self.solve_semidef, mosek_params=solver_opts , eps = self.epsilon)
        elif self.solve_semidef == cp.SCS:
          prob.solve(solver=self.solve_semidef, max_iters = self.max_iters_in_semidefinite_program, eps = self.epsilon)
        elif self.solve_semidef == cp.ECOS:
          prob.solve(solver=cp.ECOS, max_iters = self.max_iters_in_semidefinite_program, verbose=True)
        elif self.solve_semidef == cp.CVXOPT:
          prob.solve(solver=cp.CVXOPT, solver_opts={"maxiter": self.max_iters_in_semidefinite_program})
        elif self.solve_semidef == cp.GUROBI:
          prob.solve(solver=cp.GUROBI)
        else:
          print("Unknown solver")
          return 0
        
        return prob.value, X.value
    
    # def psevdoin(self, p, B, rank: int):
    #   """ Нахождение нулевого приближения методом псевдоинверсии """
    #   R = np.linalg.pinv(B) @ p
    #   R = np.array([[R[0][0], R[3][0], R[6][0]],[R[1][0], R[4][0], R[7][0]],[R[2][0], R[5][0], R[8][0]]])

    #   s, w, v = np.linalg.svd(B)
    #   D1, V1 = np.linalg.eigh(R)
    #   N = len(D1)
    #   for j in range(N):
    #         if D1[j]<0:
    #           D1[j]=0
    #   D1=sorted(D1, reverse = True)
    #   D1=D1/np.linalg.norm(D1)
    #   matrix = np.zeros((N,N))
    #   for j in range(N):
    #     for i in range(N):
    #       if (i==j):
    #         if D1[i]<0:
    #           matrix[i][j]=0
    #         else:
    #           matrix[i][j]=D1[i]
    #   D1 = matrix
    #   #  Расчёт V1
    #   V11=V1[:,0]
    #   V12=V1[:,1]
    #   V13=V1[:,2]
    #   #  V0 = np.array([[0],[0],[0]])
    #   V1 = np.column_stack([V13, V12, V11])
    #   R = V1.dot(D1)
    #   PSI = self.psi(R)
    #   PSI = PSI[:,: rank]
    #   R = self.density(PSI)
    #   R = R/np.trace(R)
    #   return(R)
    
    # def ml(self, p, P, r0, k: list = [0,0], epsilon = 1.0e-11, sigma: list = [], max: int = 10000, alpha=0.5):
    #     """
    #     Maximum-liklehood 
    #     """
    #     N=len(p)
    #     sigma = np.full(N,1)
    #     # if (sigma==np.full(N,0)):
    #     #   sigma=np.full(N,7000)
    #     # if (k==np.full(N,0)).all():
    #     #   k=sigma*p
    #     #Создание матрицы Q
    #     Q=0
    #     i=0
    #     # print(sigma)
    #     for j in range(N):
    #       # print(P[j])
    #       Q=Q+(sigma[j])*P[j]
    #     #Метод простых итераций
    #     Psi0 = self.psi(r0)
    #     # Psi0 = np.array([[Psi0[0]],[Psi0[1]],[Psi0[2]]])
    #     for i in range (0,max):
    #     #Создание матрицы А
    #       A=0

    #       for j in range(0,N):
    #         if k[j]==0:
    #           A=A
    #         else:
    #           if np.trace(P[j] @ self.density(Psi0)) != 0:
    #             A=A+(k[j]/np.trace(P[j] @ self.density(Psi0)))*P[j]

    #       Psi1=(1-alpha)*((np.linalg.inv(Q))@ A @ Psi0)+alpha*Psi0
    #       if abs(np.linalg.norm(Psi0)-np.linalg.norm(Psi1))<epsilon:
    #         break
    #       # i+=1
    #       # self.fidelity_midle.append(abs(self.Fidelity(self.density(Psi0)/np.trace(self.density(Psi0)),self.start_density)))
    #       # Psi1 = self.norm_sum_squares(Psi1)
    #       Psi0 = Psi1
         
    #     Rx = self.density(Psi0)

    #     Rx = Rx/np.trace(Rx)
    #     return (Rx)

def save_json_fix_z():
  for epoch in range(10):
    type_protocola = "fedorov"  # может быть "fedorov" или "one_plate" в зависимости от протокола

    if type_protocola == "fedorov":
      tomography_1 = ACT(oper_fedorov_basis, 1, 3)
      x = np.array([1,2,3])
    elif type_protocola == "one_plate":
      tomography_1 = ACT(oper_start_protocol_mix, 1, 3)
      x = np.array([1,2,3,4,5])
    else:
      print("Not found name this protocol")

    """
    Отрисовка усреднённых N графиков зависимости s_cvx от k
    """
    from tqdm import tqdm
    import time
    svx_list = []         # набор s_cvx для 
    fidelity_list = []
    N_list = [10,50,100,500]
    random_r = tomography_1.r_rank_r(1,3,"complex")
    for N in N_list:
      for i in tqdm(range(N)):
        svx_list_one_measurement, fidelity_list_one_measurement = tomography_1.main(random_r=random_r)
        if svx_list_one_measurement is not np.inf :
          svx_list.append(svx_list_one_measurement)
          fidelity_list.append(np.abs(fidelity_list_one_measurement))

      y = np.mean(np.array(svx_list),axis = 0)
      std = np.std(np.array(svx_list),axis = 0)
      fidelity_mean = np.mean((fidelity_list), axis = 0)
      fidelity_std = np.std(np.array(fidelity_list),axis = 0)


      print("Mean fidelity:", fidelity_mean , "\tStd fidelity:", fidelity_std)
      print("Mean svx for protocol:", y,"\tStd s_cvx for protocol:", std)
      print()
      random_r_str = [[str(item) for item in row] for row in random_r.tolist()]

      data_to_save = {
          "density_matrix": random_r_str,
          "parameters": {
              "Number of iterations": N,
              "fidelity": list(fidelity_mean),
              "Std fidelity":list(fidelity_std),
              "S_cvx": list(y),
              "Std s_cvx": list(std)
          }
      }

      # Сохраняем в JSON файл

      with open("fix_matrix_r_notfix_z.json", 'a') as file:
              file.write(json.dumps(data_to_save) + '\n')

  print("Данные успешно сохранены в 'matrix_and_parameters.json'.")

def pl_fid_s_cvx(x,y,std,fidelity_mean,fidelity_std):
  # Создаем общую фигуру с subfigures
  fig = plt.figure(constrained_layout=True, figsize=(10, 5))
  subfigs = fig.subfigures(1, 2)  # одна строка, две колонки


  # Первая subfigure для графика SVX
  ax1 = subfigs[0].subplots()
  ax1.errorbar(x, y, 
             yerr=std,  # вертикальные погрешности
             fmt='o',   # стиль маркера (кружки)
             color='blue', 
             markersize=3, 
             capsize=2,  # размер "шапочки" погрешности
             label='$S_{\mathrm{cvx}} \pm$ std')
  # ax1.set_yscale('log')
# Дополнительная тонкая линия, соединяющая точки (опционально)
  ax1.plot(x, y, color='blue', alpha=0.3, linestyle='--', linewidth=1)
  # ax1.plot(x, y, yerr=std, label='S_cvx Mean', color='blue')
  # ax1.fill_between(x, y - std, y + std, color='lightblue', alpha=0.5, label='S_cvx Std')
  ax1.set_xlabel('Количество измерений')
  ax1.set_ylabel('$S_{\mathrm{cvx}}$')
  # ax1.set_title('График зависимости $S_{\mathrm{cvx}}$ \n от количества измерений.')
  # ax1.set_ylim(bottom=y[-1]/10, top=1)  
  ax1.set_ylim(-0.1, 1.1)
  ax1.set_xlim(1, len(x)+0.2)
  ax1.grid(True)
  ax1.legend()

  # Вторая subfigure для графика Fidelity
  ax2 = subfigs[1].subplots()
  ax2.errorbar(x, fidelity_mean, 
             yerr=fidelity_std,  # вертикальные погрешности
             fmt='o',            # стиль маркера (кружки)
             color='green', 
             markersize=3, 
             capsize=4,          # размер "шапочки" погрешности
             label='Fidelity ± std')
  
  # ax2.set_yscale('log')
# Дополнительная тонкая линия, соединяющая точки (как в исходном коде)
  ax2.plot(x, fidelity_mean, color='green', alpha=0.3, linestyle='--', linewidth=1)
  # ax2.plot(x, fidelity_mean, label='Fidelity Mean', color='green', marker='o')
  # ax2.fill_between(x, fidelity_mean - fidelity_std, fidelity_mean + fidelity_std, color='lightgreen', alpha=0.5, label='Fidelity Std')
  ax2.set_xlabel('Количество измерений')
  ax2.set_ylabel('Fidelity')
  # ax2.set_title('График зафисимости fidelity \n от количества измерений.')
  ax2.set_ylim(-0.1, 1.1)
  ax2.set_xlim(1, len(x) + 0.2)
  ax2.grid(True)
  ax2.legend(loc='upper left')

  # Отображаем общий заголовок
  # fig.suptitle(name_title, fontsize=16)
  
  # fig.suptitle(f'Протокол одной пластинкой λ/8', fontsize=16)
  # fig.suptitle(f'Comparison of S_cvx and Fidelity with eps = {10**-4}', fontsize=16)
  plt.show()

# np.random.seed(52)
# # protocol = np.concatenate((oper_fedorov_basis, np.expand_dims(oper_start_protocol_mix[4], axis=0)), axis=0)
# tomography_1 = ACT(oper_start_protocol_mix, 2, 3)
# x = np.array([1, 2, 3, 4, 5])
# svx_list = []         
# fidelity_list = []
# N = 1


# for i in tqdm(range(N)):
#   svx_list_one_measurement, fidelity_list_one_measurement, fidelity_x_min, fidelity_x_max, x_min, x_max, state_ml = tomography_1.main()
#   if svx_list_one_measurement is not np.inf :
#     svx_list.append(svx_list_one_measurement)
#     fidelity_list.append(np.abs(fidelity_list_one_measurement))

# y = np.mean(np.array(svx_list),axis = 0)
# std = np.std(np.array(svx_list),axis = 0)
# fidelity_mean = np.mean((fidelity_list), axis = 0)
# fidelity_std = np.std(np.array(fidelity_list),axis = 0)


# print("Mean fidelity:", fidelity_mean , "\tStd fidelity:", fidelity_std)
# print("Mean svx for protocol:", y,"\tStd s_cvx for protocol:", std)
# print()
# pl_fid_s_cvx(x,y,std,fidelity_mean,fidelity_std)
convert_dictlist_to_matrix = lambda matrix_str: np.array([[complex(cell) for cell in row] for row in matrix_str])

# np.random.seed(52)
# import numpy as np
# from collections import defaultdict
# from plot_json import convert_dictlist_to_matrix
# from purification_state import*
# convert_dictlist_to_matrix = lambda matrix_str: np.array([[complex(cell) for cell in row] for row in matrix_str])
# # открывыю из файла который восстанавливался для матриц с данным рангом
# with open("dicts_matrix\\fix_matrix_r_notfix_z_with_x_min_max_r_3.json", "r") as json_file:
#     data = []
#     # Чтение файла построчно
#     for line in json_file:
#         if line.strip():  # Пропуск пустых строк
#             data.append(json.loads(line))  # Парсим каждый объект JSON
# # Группировка данных по строковому представлению матрицы
# grouped_data_r = defaultdict(list)
# for entry in data:
#     # Преобразуем матрицу в строку для сравнения
#     density_matrix_str = str(entry["density_matrix"])
#     grouped_data_r[density_matrix_str].append(entry)
# data_s = data

# tomography_1 = ACT(oper_fedorov_basis, 1, 3)
# m = [3,7,11,15,19,23,27,31,35,39]
# m = 35
# matrix_str = convert_dictlist_to_matrix(data[m]["density_matrix"])
# matrix_complex = np.array([[complex(cell) for cell in row] for row in matrix_str])


# tomography_1 = ACT(oper_fedorov_basis, 1, 3)
# x = np.array([1, 2, 3])
# svx_list = []         
# fidelity_list = []
# N = 10



# for i in tqdm(range(N)):
#   svx_list_one_measurement, fidelity_list_one_measurement, fidelity_x_min, fidelity_x_max, x_min, x_max, state_ml =\
#       tomography_1.main(random_r=matrix_complex, epsilon = 10**-9, type_solve_semidefinite_program=cp.MOSEK, max_iters_in_semidefinite_program= 10**9)
#   if svx_list_one_measurement is not np.inf :
#     svx_list.append(svx_list_one_measurement)
#     fidelity_list.append(np.abs(fidelity_list_one_measurement))


# print(np.round(svx_list,10))

# import os

# # Проверка переменной окружения
# license_file_env = os.getenv("MOSEKLM_LICENSE_FILE", "Переменная не установлена")
# print("Путь из переменной MOSEKLM_LICENSE_FILE:", license_file_env)