import numpy as np
hh = np.array([[1], [0], [0]])
hv = np.array([[0], [1], [0]])
vv = np.array([[0], [0], [1]])
rr = np.array([[1 / 2], [1j / np.sqrt(2)], [-1 / 2]])
aa = np.array([[1 / 2], [-1 / np.sqrt(2)], [1 / 2]])
dd = np.array([[1 / 2], [1 / np.sqrt(2)], [1 / 2]])
ll = np.array([[1 / 2], [-1j / np.sqrt(2)], [-1 / 2]])
A01 = np.array([[1,0,0],[0,0,0],[0,0,0]])
A02 = np.array([[0,0,0],[0,1,0],[0,0,0]])
A03 = np.array([[0,0,0],[0,0,0],[0,0,1]])
A00 = [A01, A02, A03]

def density(psi):
    return np.dot(psi,np.transpose(np.conj(psi)))
def tl_4(Q):
  return(pow(np.cos(Q),2)+1j*pow(np.sin(Q),2))/np.sqrt(1j)
def rl_4(Q):
  return(((1-1j)*np.cos(Q)*np.sin(Q))/np.sqrt(1j))
def tl_2(Q):
  return(1j*np.cos(2*Q))
def rl_2(Q):
  return(1j*np.sin(2*Q))
def tl_8(Q):
    return np.cos(-np.pi/8)+1j*np.sin(-np.pi/8)*np.cos(2*Q)
def rl_8(Q):
    return 1j*np.sin(-np.pi/8)*np.sin(2*Q)

#Матрица преобразование кутрита

def Gl_4(Q):
 M=np.array([[pow(tl_4(Q),2),np.sqrt(2)*(tl_4(Q))*(rl_4(Q)),pow(rl_4(Q),2)],
  [-np.sqrt(2)*tl_4(Q)*np.conj(rl_4(Q)),pow(abs(tl_4(Q)),2)-pow(abs(rl_4(Q)),2) ,np.sqrt(2)*np.conj(tl_4(Q))*rl_4(Q)],
   [pow(np.conj(rl_4(Q)),2),-np.sqrt(2)*np.conj(tl_4(Q))*np.conj(rl_4(Q)),pow(np.conj(tl_4(Q)),2)]])
 return(M)
def Gl_2(Q):
 M=np.array([[pow(tl_2(Q),2),np.sqrt(2)*(tl_2(Q))*(rl_2(Q)),pow(rl_2(Q),2)],
  [-np.sqrt(2)*tl_2(Q)*np.conj(rl_2(Q)),pow(abs(tl_2(Q)),2)-pow(abs(rl_2(Q)),2) ,np.sqrt(2)*np.conj(tl_2(Q))*rl_2(Q)],
   [pow(np.conj(rl_2(Q)),2),-np.sqrt(2)*np.conj(tl_2(Q))*np.conj(rl_2(Q)),pow(np.conj(tl_2(Q)),2)]])
 return(M)

#Матрица преобразование кубита

def U_4(Q):
 return (np.array([[tl_4(Q),rl_4(Q)],[-np.conj(rl_4(Q)),np.conj(tl_4(Q))]]))
def U_2(Q):
  return (np.array([[tl_2(Q),rl_2(Q)],[-np.conj(rl_2(Q)),np.conj(tl_2(Q))]]))

# пластинка l/8

def U_8(Q):
    return np.array([[tl_8(Q), rl_8(Q)], [-np.conj(rl_8(Q)), np.conj(tl_8(Q))]])
def Gl_8(Q):
    M = np.array([[tl_8(Q)**2, np.sqrt(2)*(tl_8(Q))*(rl_8(Q)), rl_8(Q)**2],
               [-np.sqrt(2)*tl_8(Q)*np.conj(rl_8(Q)),pow(abs(tl_8(Q)),2)-pow(abs(rl_8(Q)),2) ,np.sqrt(2)*np.conj(tl_8(Q))*rl_8(Q)],
               [(np.conj(rl_8(Q)))**2,-np.sqrt(2)*np.conj(tl_8(Q))*np.conj(rl_8(Q)),(np.conj(tl_8(Q)))**2]])
    return M

def format_element(element):
    if isinstance(element, complex):
        return f"({element.real:.2f}+{element.imag:.2f}j)" if element.imag >= 0 else f"({element.real:.2f}{element.imag:.2f}j)"
    else:
        return f"{element:.2f}"

# Функция для печати матриц в два столбца
def print_matrices_in_columns(matrix1, matrix2):
    # Проверяем, что матрицы имеют одинаковое количество строк
    if matrix1.shape[0] != matrix2.shape[0]:
        raise ValueError("Матрицы должны иметь одинаковое количество строк")
    
    # Форматируем строки матриц
    formatted_matrix1 = np.array([[format_element(el) for el in row] for row in matrix1])
    formatted_matrix2 = np.array([[format_element(el) for el in row] for row in matrix2])
    
    # Объединяем строки матриц
    combined = np.hstack((formatted_matrix1, np.full((matrix1.shape[0], 1), ' '), formatted_matrix2))
    
    # Печатаем строки объединенной матрицы
    for row in combined:
        print(' '.join(map(str, row)))
        
        
protocol = [np.diag((1,1,1)),Gl_2(np.pi/8), Gl_2(np.pi/8) @ Gl_8(0) ]
oper_fedorov = 1j*np.ones((3*len(protocol),3,3))
k = 0
for i in range(3):
  for j in range(len(protocol)):
    oper_fedorov[k] = np.array(np.conj(protocol[j]).T @ A00[i] @ protocol[j])
    k+=1
oper_fedorov_mix = np.array([oper_fedorov[0],oper_fedorov[3],oper_fedorov[6],oper_fedorov[1],
                             oper_fedorov[4],oper_fedorov[7],oper_fedorov[2],oper_fedorov[5],oper_fedorov[8]])
# for i in range(0,8,2):
#     print_matrices_in_columns(oper_fedorov_mix[i], oper_fedorov_mix[i+1])
#     print()


oper_fedorov_basis_1 = [oper_fedorov_mix[0], oper_fedorov_mix[1], oper_fedorov_mix[2]]
oper_fedorov_basis_2 = [oper_fedorov_mix[3], oper_fedorov_mix[4], oper_fedorov_mix[5]]
oper_fedorov_basis_3 = [oper_fedorov_mix[6], oper_fedorov_mix[7], oper_fedorov_mix[8]]
oper_fedorov_basis = np.array([oper_fedorov_basis_3, oper_fedorov_basis_2, oper_fedorov_basis_1])

# print((oper_fedorov_basis_2))
# for i in range(2):
#    print_matrices_in_columns(oper_fedorov_basis_2[i],oper_fedorov_basis_2[i+1])

""" для протоколя λ/4"""
start_protocol = [Gl_4(0),Gl_4(np.pi/8),Gl_4(3*np.pi/8),Gl_4(5*np.pi/8),Gl_4(7*np.pi/8)]
oper_start_protocol = 1j*np.ones((3*len(start_protocol),3,3))
k = 0
for i in range(3):
  for j in range(len(start_protocol)):
    oper_start_protocol[k] = np.array(np.conj(start_protocol[j]).T @ A00[i] @ start_protocol[j])
    k+=1
oper_start_protocol_mix = np.array([[oper_start_protocol[0],oper_start_protocol[5],oper_start_protocol[10]],
                                    [oper_start_protocol[1],oper_start_protocol[6],oper_start_protocol[11]],
                                    [oper_start_protocol[2],oper_start_protocol[7],oper_start_protocol[12]],
                                    [oper_start_protocol[3],oper_start_protocol[8],oper_start_protocol[13]],
                                    [oper_start_protocol[4],oper_start_protocol[9],oper_start_protocol[14]]])


def generalized_pauli_matrices(n: int):
    """
    Generate matrix pauli
    Args:
      n: (int) dimension matrix
    Return:
      pauli_matrices: (list) all matrix pauli with dimension n
    """
    pauli_matrices = []
    
    # Оператор сдвига X
    X = np.zeros((n, n), dtype=complex)
    for i in range(n):
        X[i, (i + 1) % n] = 1
    
    # Оператор фазы Z
    Z = np.zeros((n, n), dtype=complex)
    for i in range(n):
        Z[i, i] = np.exp(2j * np.pi * i / n)
    
    # Добавляем комбинации операторов X^a Z^b
    for a in range(n):
        for b in range(n):
            matrix = np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(Z, b)
            pauli_matrices.append(np.array(matrix))
    
    return pauli_matrices

def generate_exp_state():
  """
  Generate the pure state in our experiment
  Return:
    (np.ndarray) state with rank = 2 
  """
  tetha = np.random.uniform(0,360) * np.pi/180
  phi = np.random.uniform(0,360) * np.pi/180
  pure_state = np.array([
    [np.exp(1j*phi)*np.sin(tetha/2)],
    [0],
    [np.cos(tetha/2)]
  ])
  
  state = pure_state @ np.conj(pure_state.T)
  return  (state/np.trace(state))

def generate_exp_state_rank2():
  """
  Generate the state with rank = 2 with pure state in our experiment
  Return:
    (np.ndarray) state with rank = 2 
  """
  p_1 = np.random.uniform(0, 1)
  p_2 = 1 - p_1
  return (p_1 * generate_exp_state() + p_2 * generate_exp_state())

# Пример для n = 2
n = 3
pauli_matrices = generalized_pauli_matrices(n)
# print(pauli_matrices[1])
# for i, mat in enumerate(pauli_matrices):
#     print(f"Генерализованная матрица Паули {i}:")
#     print(np.round(mat, 2))  # Округлим для удобства
#     print()

# print(generate_exp_state_rank2())
