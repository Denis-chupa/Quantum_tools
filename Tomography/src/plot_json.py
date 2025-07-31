from Adaptive_compresive_cending_qutrit import*
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def plot_s_cvx_and_fid_json(grouped_data):
    
    # Построение графиков для каждой матрицы (например, для одной матрицы с 4 записями)
    numb_group = 1
    for density_matrix_str, group in grouped_data.items():
        if len(group) == 4:  # Например, если для этой матрицы есть 4 записи
            fig, subfigs = plt.subplots(2, 4, figsize=(12, 6))  # Размер фигуры
            x = [1,2,3]

            for i, ax in enumerate(subfigs[0].flat):
                # params = entry["parameters"]
                params = group[i]["parameters"]
                s_cvx = np.array(params["S_cvx"])  # Среднее значение S_cvx
                s_cvx_std = np.array(params["Std s_cvx"])  # Среднее значение Std s_cvx
                N = int(params["Number of iterations"])
                # Построение графиков (например, fidelity и S_cvx для каждой записи)
                
                ax.plot(x, s_cvx, label="S_cvx", color='blue')
                ax.fill_between(x, s_cvx - s_cvx_std, s_cvx + s_cvx_std, 
                                color='blue', alpha=0.5, label='Std_s_cvx')
                # ax.set_title(group[i]["title"])
                ax.set_xlabel('Measurement number')
                ax.set_ylabel(f"S_cvx with N = {N}")
                ax.set_ylim(0, 1)
                ax.set_xlim((1, 3))
                ax.grid(True)
                ax.legend()

            for i, ax in enumerate(subfigs[1].flat):
                params = group[i]["parameters"]
                fidelity = np.array(params["fidelity"])
                std_fidelity = np.array(params["Std fidelity"])
                N = int(params["Number of iterations"])
                # Построение графиков (например, fidelity и S_cvx для каждой записи)
                
                ax.plot(x, fidelity, label="Fidelity", color='green')
                ax.fill_between(x, fidelity - std_fidelity, fidelity + std_fidelity, 
                                color='green', alpha=0.5, label='Fidelity_s_cvx')
                # ax.set_title(group[i]["title"])
                ax.set_xlabel('Measurement number')
                ax.set_ylabel(f"Fidelity with N = {N}")
                ax.set_ylim(0, 1)
                ax.set_xlim((1, 3))
                ax.grid(True)
                ax.legend()

            # Общие настройки для всей фигуры
            plt.suptitle(f'Graphs for fixed the density matrix {numb_group}')
            plt.tight_layout()  # Для добавления пространства под заголовком
            plt.show()
            numb_group+=1
            # break  # Если нужно только для первой группы

def save_2_json_fix_z(data):
    tomography_1 = ACT(oper_fedorov_basis, 1, 3)

    with open("fix_matrix_r_notfix_z_with_x_min_max_r_3.json", 'w') as file:
        pass
    number_state = 0
    for data_element in data:
        number_state += 1
        matrix_str = data_element["density_matrix"]
        matrix_complex = np.array([[complex(cell) for cell in row] for row in matrix_str])
        N = data_element["parameters"]["Number of iterations"]
        svx_list = [] 
        fidelity_list = []
        fidelity_x_min_list = []
        fidelity_x_max_list = []
        x_min_list = []
        x_max_list = []
        state_ml_list = []
        for i in tqdm(range(N)):
            svx_list_one_measurement, fidelity_list_one_measurement, fidelity_x_min, fidelity_x_max, x_min, x_max, state_ml = tomography_1.main(random_r=matrix_complex)
            if svx_list_one_measurement is not np.inf :
                svx_list.append(svx_list_one_measurement)
                fidelity_list.append(np.abs(fidelity_list_one_measurement))
                fidelity_x_min_list.append(np.abs(fidelity_x_min))
                fidelity_x_max_list.append(np.abs(fidelity_x_max))
                x_min_list.append(x_min)
                x_max_list.append(x_max)
                state_ml_list.append(state_ml)
        
        
        data_to_save = {
                "density_matrix": matrix_str,
                "parameters": {
                    "Number of iterations": N,
                    "Svx": svx_list,
                    "Fidelity": fidelity_list,
                    "Fidelity_x_min": fidelity_x_min_list,
                    "Fidelity_x_max": fidelity_x_max_list,
                    "X_min": x_min_list,
                    "X_max": x_max_list,
                    "State_ml": state_ml_list
                }
            }

        data_to_save = convert_ndarray_to_list(data_to_save)

        with open("fix_matrix_r_notfix_z_with_x_min_max_r_3.json", 'a') as file:
                file.write(json.dumps(data_to_save) + '\n')
        print(number_state)
    print("Данные успешно сохранены в 'fix_matrix_r_notfix_z_with_x_min_max_r_3.json'.")

def convert_ndarray_to_list(obj):
    """
    Recursively convert all NumPy ndarrays in the input object to lists.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_ndarray_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_ndarray_to_list(item) for item in obj)
    else:
        return obj

def plot_with_x_maxmin(data,m):
    
    numb_group = 0
    for j in m:
        numb_group += 1
        fig, subfigs = plt.subplots(1, 4, figsize=(12, 6))  # Размер фигуры
        x = [1,2,3]

        parametrs = data[j]["parameters"]
        N = int(parametrs["Number of iterations"])
        s_cvx_list = np.array(parametrs["Svx"]) 
        s_cvx = np.mean(np.array(s_cvx_list),axis = 0)
        s_cvx_std = np.std(np.array(s_cvx_list),axis = 0)
        fidelity_list = np.array(parametrs["Fidelity"])
        fidelity = np.mean(np.array(fidelity_list),axis = 0)
        fidelity_std = np.std(np.array(fidelity_list),axis = 0)
        fidelity_x_min_list = np.array(parametrs["Fidelity_x_min"])
        fidelity_x_min = np.mean(np.array(fidelity_x_min_list),axis = 0)
        fidelity_x_min_std = np.std(np.array(fidelity_x_min_list),axis = 0)
        fidelity_x_max_list = np.array(parametrs["Fidelity_x_max"])
        fidelity_x_max = np.mean(np.array(fidelity_x_max_list),axis = 0)
        fidelity_x_max_std = np.std(np.array(fidelity_x_max_list),axis = 0)

        for i, ax in enumerate(subfigs.flat):
            match i:
                case 0:       
                    ax.plot(x, s_cvx, color='blue')
                    ax.fill_between(x, s_cvx - s_cvx_std, s_cvx + s_cvx_std, 
                                    color='blue', alpha=0.5, label='Std_s_cvx')
                    ax.set_ylabel('S_cvx')
                case 1:
                    ax.plot(x, fidelity, color='green')
                    ax.fill_between(x, fidelity - fidelity_std, fidelity + fidelity_std, 
                                    color='green', alpha=0.5, label='Std_s_cvx')
                    ax.set_ylabel('Fidelity')
                case 2:
                    ax.plot(x, fidelity_x_min, color='green')
                    ax.fill_between(x, fidelity_x_min - fidelity_x_min_std, fidelity_x_min + fidelity_x_min_std, 
                                    color='green', alpha=0.5, label='Std_s_cvx')
                    ax.set_ylabel('Fidelity_x_min')
                case 3:
                    ax.plot(x, fidelity_x_max, color='green')
                    ax.fill_between(x, fidelity_x_max - fidelity_x_max_std, fidelity_x_max + fidelity_x_max_std, 
                                    color='green', alpha=0.5, label='Std_s_cvx')
                    ax.set_ylabel('Fidelity_x_max')

            ax.set_xlabel('Measurement number')
            ax.set_ylim(0, 1)
            ax.set_xlim((1, 3))
            ax.grid(True)
            # ax.legend()
        plt.suptitle(f'Graphs for fixed the density matrix {np.floor(j/4)+1} with N = {N}')
        plt.tight_layout()  # Для добавления пространства под заголовком
        plt.show()


convert_dictlist_to_matrix = lambda matrix_str: np.array([[complex(cell) for cell in row] for row in matrix_str])

# # Открытие и чтение данных из файла
# with open("dicts_matrix\\fix_matrix_r_notfix_z_with_x_min_max_r_3.json", "r") as json_file:
#     data = []
#     # Чтение файла построчно
#     for line in json_file:
#         if line.strip():  # Пропуск пустых строк
#             data.append(json.loads(line))  # Парсим каждый объект JSON
# # Группировка данных по строковому представлению матрицы
# grouped_data = defaultdict(list)
# for entry in data:
#     # Преобразуем матрицу в строку для сравнения
#     density_matrix_str = str(entry["density_matrix"])
#     grouped_data[density_matrix_str].append(entry)


# j = 23
# m_6 = convert_dictlist_to_matrix(data[j]["density_matrix"])
# print(f"density matrix {np.floor(j/4)+1}")
# parametrs = data[j]["parameters"]
# # print("Параметры x_min:", convert_dictlist_to_matrix(np.array(parametrs["X_min"])[0][0]))
# # print([[0]*3]*10)
# x_min = convert_dictlist_to_matrix(parametrs["X_min"][25][0])

# print(parametrs["Fidelity_x_min"][33])
# x_min = np.array([[0.5,1j],[-1j,0.5]])
# w, v = np.linalg.eig(x_min) 

# print("Printing the Eigen values of the given square array:\n", w) 
# print("Printing the Eigen vectors of the given square array:\n", v) 


# tomography_1 = ACT(oper_fedorov_basis, 1, 3)
# # m = [3,7,11,15,19,23,27,31,35,39]
# m = [34]
# for j in m:
#     matrix_str = convert_dictlist_to_matrix(data[j]["density_matrix"])
#     parametrs = data[j]["parameters"]
#     s_cvx_list = np.array(parametrs["Svx"]) 
#     N = np.array(parametrs["Number of iterations"]) 
#     # fidelity_list = np.array(parametrs["Fidelity"]) 
#     # x_min_list = [[convert_dictlist_to_matrix(element[0]), convert_dictlist_to_matrix(element[1]), convert_dictlist_to_matrix(element[2])] for element in np.array(parametrs["X_min"])]
#     # x_max_list = [[convert_dictlist_to_matrix(element[0]), convert_dictlist_to_matrix(element[1]), convert_dictlist_to_matrix(element[2])]  for element in np.array(parametrs["X_max"])] 
#     # stateml_list = [[convert_dictlist_to_matrix(element[0]), convert_dictlist_to_matrix(element[1]), convert_dictlist_to_matrix(element[2])]  for element in np.array(parametrs["State_ml"])]
#     # fidelity_list_xmax_xmin = [[tomography_1.Fidelity(x_min_list[ind][0], x_max_list[ind][0]), tomography_1.Fidelity(x_min_list[ind][1], x_max_list[ind][1]), tomography_1.Fidelity(x_min_list[ind][2], x_max_list[ind][2])] for ind in range(len(x_max_list))]
#     # fidelity_list_xmax_stateml = [[tomography_1.Fidelity(stateml_list[ind][0], x_max_list[ind][0]), tomography_1.Fidelity(stateml_list[ind][1], x_max_list[ind][1]), tomography_1.Fidelity(stateml_list[ind][2], x_max_list[ind][2])]  for ind in range(len(x_max_list))]
#     # fidelity_list_xmin_stateml = [[tomography_1.Fidelity(stateml_list[ind][0], x_min_list[ind][0]), tomography_1.Fidelity(stateml_list[ind][1], x_min_list[ind][1]), tomography_1.Fidelity(stateml_list[ind][2], x_min_list[ind][2])]  for ind in range(len(x_max_list))]
    
#     # print("S_cvx with esp = 10^-5 :", np.mean(np.array(s_cvx_list),axis = 0) )
#     # print("std_S_cvx with esp = 10^-5 :", np.std(np.array(s_cvx_list),axis = 0) )

#     matrix_complex = np.array([[complex(cell) for cell in row] for row in matrix_str])
#     # N = 200
#     svx_list = [] 
#     fidelity_list = []
#     fidelity_x_min_list = []
#     fidelity_x_max_list = []
#     x_min_list = []
#     x_max_list = []
#     state_ml_list = []
#     for i in tqdm(range(N)):
#         svx_list_one_measurement, fidelity_list_one_measurement, fidelity_x_min, fidelity_x_max, x_min, x_max, state_ml = tomography_1.main(type_ml = "without_ml", random_r=matrix_complex)
#         if svx_list_one_measurement is not np.inf :
#             svx_list.append(svx_list_one_measurement)
#             fidelity_list.append(np.abs(fidelity_list_one_measurement))
#             fidelity_x_max_list.append(np.abs(fidelity_x_max))
#             x_min_list.append(x_min)
#             x_max_list.append(x_max)
#             state_ml_list.append(state_ml)
#             x_min_list = [[convert_dictlist_to_matrix(element[0]), convert_dictlist_to_matrix(element[1]), convert_dictlist_to_matrix(element[2])] for element in np.array(parametrs["X_min"])]
#             x_max_list = [[convert_dictlist_to_matrix(element[0]), convert_dictlist_to_matrix(element[1]), convert_dictlist_to_matrix(element[2])]  for element in np.array(parametrs["X_max"])] 
#             stateml_list = [[convert_dictlist_to_matrix(element[0]), convert_dictlist_to_matrix(element[1]), convert_dictlist_to_matrix(element[2])]  for element in np.array(parametrs["State_ml"])]
#             fidelity_list_xmax_xmin =  [[tomography_1.Fidelity(x_min_list[ind][0], x_max_list[ind][0]), tomography_1.Fidelity(x_min_list[ind][1], x_max_list[ind][1]), tomography_1.Fidelity(x_min_list[ind][2], x_max_list[ind][2])] for ind in range(len(x_max_list))]
#             fidelity_list_xmax_stateml = [[tomography_1.Fidelity(stateml_list[ind][0], x_max_list[ind][0]), tomography_1.Fidelity(stateml_list[ind][1], x_max_list[ind][1]), tomography_1.Fidelity(stateml_list[ind][2], x_max_list[ind][2])]  for ind in range(len(x_max_list))]
#             fidelity_list_xmin_stateml = [[tomography_1.Fidelity(stateml_list[ind][0], x_min_list[ind][0]), tomography_1.Fidelity(stateml_list[ind][1], x_min_list[ind][1]), tomography_1.Fidelity(stateml_list[ind][2], x_min_list[ind][2])]  for ind in range(len(x_max_list))]
#     s_cvx = np.mean(np.array(s_cvx_list),axis = 0)
#     s_cvx_std = np.std(np.array(s_cvx_list),axis = 0)
#     fidelity = np.mean(np.array(fidelity_list),axis = 0)
#     fidelity_std = np.std(np.array(fidelity_list),axis = 0)
#     fidelity_xmax_xmin = np.mean(np.array(fidelity_list_xmax_xmin),axis = 0)
#     fidelity_xmax_xmin_std = np.std(np.array(fidelity_list_xmax_xmin),axis = 0)
#     fidelity_xmax_stateml = np.mean(np.array(fidelity_list_xmax_stateml),axis = 0)
#     fidelity_xmax_stateml_std = np.std(np.array(fidelity_list_xmax_stateml),axis = 0)
#     fidelity_xmin_stateml = np.mean(np.array(fidelity_list_xmin_stateml),axis = 0)
#     fidelity_xmin_stateml_std = np.std(np.array(fidelity_list_xmin_stateml),axis = 0)
#     # fidelity_x_min = np.mean(np.array(fidelity_x_min_list),axis = 0)
#     # fidelity_x_min_std = np.std(np.array(fidelity_x_min_list),axis = 0)
#     # fidelity_x_max = np.mean(np.array(fidelity_x_max_list),axis = 0)
#     # fidelity_x_max_std = np.std(np.array(fidelity_x_max_list),axis = 0)
#     fig, subfigs = plt.subplots(1, 5, figsize=(15, 4))  # Размер фигуры
#     x = [1,2,3]
#     for i, ax in enumerate(subfigs.flat):
#         match i:
#             case 0:       
#                 ax.plot(x, s_cvx, color='blue')
#                 ax.fill_between(x, s_cvx - s_cvx_std, s_cvx + s_cvx_std, 
#                                 color='blue', alpha=0.5, label='Std_s_cvx')
#                 ax.set_ylabel('S_cvx')
#             case 1:
#                 ax.plot(x, fidelity, color='green')
#                 ax.fill_between(x, fidelity - fidelity_std, fidelity + fidelity_std, 
#                                 color='green', alpha=0.5, label='Std_s_cvx')
#                 ax.set_ylabel('Fidelity')
#             case 2:
#                 ax.plot(x, fidelity_xmax_xmin, color='green')
#                 ax.fill_between(x, fidelity_xmax_xmin - fidelity_xmax_xmin_std, fidelity_xmax_xmin + fidelity_xmax_xmin_std, 
#                                 color='green', alpha=0.5, label='Std_s_cvx')
#                 ax.set_ylabel('Fidelity_x_min')
#             # case 3:
#             #     ax.plot(x, fidelity_xmax_stateml, color='green')
#             #     ax.fill_between(x, fidelity_xmax_stateml - fidelity_xmax_stateml_std, fidelity_xmax_stateml + fidelity_xmax_stateml_std, 
#             #                     color='green', alpha=0.5, label='Std_s_cvx')
#             #     ax.set_ylabel('Fidelity_x_max')
#             # case 4:
#             #     ax.plot(x, fidelity_xmin_stateml, color='green')
#             #     ax.fill_between(x, fidelity_xmin_stateml - fidelity_xmin_stateml_std, fidelity_xmin_stateml + fidelity_xmin_stateml_std, 
#             #                     color='green', alpha=0.5, label='Std_s_cvx')
#             #     ax.set_ylabel('Fidelity_x_max')
#         ax.set_xlabel('Measurement number')
#         ax.set_ylim(0, 1)
#         ax.set_xlim((1, 3.2))
#         ax.grid(True)
#         # ax.legend()
#     plt.suptitle(f'Graphs for fixed the density matrix {np.floor(j/4)+1} with N = {N} ')
#     plt.tight_layout()  # Для добавления пространства под заголовком
#     plt.show()

#     # print("S_cvx with esp = 10^-10 :", s_cvx) 
#     # print("std_S_cvx with esp = 10^-10 :", s_cvx_std)



# m = [11,15,19,23,27,31,35]
# m = [3,7,11,15,19,23,27,31,35,39]
# plot_with_x_maxmin(data,m)





# tomography_1 = ACT(oper_fedorov_basis, 1, 3)
# m = [3,7,11,15,19,23,27,31,35,39]
# m = [22]
# for j in m:
#     matrix_str = convert_dictlist_to_matrix(data[j]["density_matrix"])
#     parametrs = data[j]["parameters"]
#     # s_cvx_list = np.array(parametrs["Svx"]) 
#     N = np.array(parametrs["Number of iterations"]) 

#     matrix_complex = np.array([[complex(cell) for cell in row] for row in matrix_str])
#     svx_list = [] 
#     fidelity_list = []
#     fidelity_x_min_list = []
#     fidelity_x_max_list = []
#     x_min_list = []
#     x_max_list = []
#     state_ml_list = []
#     for i in tqdm(range(N)):
#         svx_list_one_measurement, fidelity_list_one_measurement, fidelity_x_min, fidelity_x_max, x_min, x_max, state_ml = tomography_1.main(type_ml = "without_ml", random_r=matrix_complex)
#         if svx_list_one_measurement is not np.inf :
#             svx_list.append(svx_list_one_measurement)
#             fidelity_list.append(np.abs(fidelity_list_one_measurement))
#             fidelity_x_max_list.append(np.abs(fidelity_x_max))
#             x_min_list.append(x_min)
#             x_max_list.append(x_max)
#             x_min_list = [[convert_dictlist_to_matrix(element[0]), convert_dictlist_to_matrix(element[1]), convert_dictlist_to_matrix(element[2])] for element in np.array(parametrs["X_min"])]
#             x_max_list = [[convert_dictlist_to_matrix(element[0]), convert_dictlist_to_matrix(element[1]), convert_dictlist_to_matrix(element[2])]  for element in np.array(parametrs["X_max"])] 
#             fidelity_list_xmax_xmin =  [[tomography_1.Fidelity(x_min_list[ind][0], x_max_list[ind][0]), tomography_1.Fidelity(x_min_list[ind][1], x_max_list[ind][1]), tomography_1.Fidelity(x_min_list[ind][2], x_max_list[ind][2])] for ind in range(len(x_max_list))]
            
#     s_cvx1 = np.mean(np.array(svx_list),axis = 0)
#     s_cvx_std = np.std(np.array(svx_list),axis = 0)
#     fidelity_xmax_xmin = np.mean(np.array(fidelity_list_xmax_xmin),axis = 0)
#     fidelity_xmax_xmin_std = np.std(np.array(fidelity_list_xmax_xmin),axis = 0)
#     fig, subfigs = plt.subplots(1, 2, figsize=(15, 4))  # Размер фигуры
#     x = [1,2,3]
#     for i, ax in enumerate(subfigs.flat):
#         match i:
#             case 0:       
#                 ax.plot(x, s_cvx1, color='blue')
#                 ax.fill_between(x, s_cvx1 - s_cvx_std, s_cvx1 + s_cvx_std, 
#                                 color='blue', alpha=0.5, label='Std_s_cvx')
#                 ax.set_ylabel('S_cvx')
#             case 1:
#                 ax.plot(x, fidelity_xmax_xmin, color='green')
#                 ax.fill_between(x, fidelity_xmax_xmin - fidelity_xmax_xmin_std, fidelity_xmax_xmin + fidelity_xmax_xmin_std, 
#                                 color='green', alpha=0.5, label='Std_s_cvx')
#                 ax.set_ylabel('Fidelity_x_min')

#         ax.set_xlabel('Measurement number')
#         ax.set_ylim(0, 1)
#         ax.set_xlim((1, 3.2))
#         ax.grid(True)
#         # ax.legend()
#     print(s_cvx1)
#     plt.suptitle(f'Graphs for fixed the density matrix {np.floor(j/4)+1} with N = {N} ')
#     plt.tight_layout()  # Для добавления пространства под заголовком
#     plt.show()