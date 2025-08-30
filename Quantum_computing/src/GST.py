import numpy as np
from scipy.integrate import quad
from scipy.linalg import sqrtm



def S_omega(omega, c:float, tau_c:float):
    """
    Вычисляет спектральную плотность Лоренца S(ω).
    Args:
        omega(float/np.ndarray): угловая частота (ω).
        c(float): диффузионная константа.
        tau_c(float): время корреляции.
    Returns:
        (float): значение S(ω).
    """
    return (c * tau_c**2) / (1 + (omega * tau_c)**2)

def nascent_Dirac_delta(x, eps=1e-2):
    """
    дельта-функция Дирака вида η_ε(x):
        η_ε(x) = (ε / (π x^2)) * sin^2(x/ε)

    Аргументы:
        x   : float или массив, аргумент функции
        eps : малый параметр, имитирующий предел δ(x)

    Возвращает:
        float или массив значений η_ε(x).
    """
    return (eps / (np.pi * x**2)) * (np.sin(x / eps))**2

def filter_gamma_1(omega, Omega, t):
    """
    Функция фильтрации F_{Γ1}(ω, Ω, t).
    Args:
        omega : частота ω
        Omega : частота Ω
        t     : время
    Возвращает:
        значение фильтра F_{Γ1}.
    """
    return (t/4.0) * (nascent_Dirac_delta(Omega - omega, 2.0 / t) + nascent_Dirac_delta(Omega + omega, 2.0 / t))

def delta_filter_delta_1(omega, Omega, t):
    """
    Дополнительный член δF_{Δ1}(ω, Ω, t).
    Args:
        omega : частота ω
        Omega : частота Ω
        t     : время
    Return:
        Дополнительный член δF_{Δ1}
    """
    term1 = np.sin((omega - Omega)*t) / ((omega - Omega)**2)
    term2 = np.sin((omega + Omega)*t) / ((omega + Omega)**2)
    return (term1 - term2) / (4 * np.pi)

def filter_delta_1(omega, Omega, t):
    """
    Фильтр F_{Δ1}(ω, Ω, t). Определяется как:
        F_{Δ1} = (Ω t) / (2π(Ω^2 - ω^2)) + δF_{Δ1}.
    Args:
        omega : частота ω
        Omega : частота Ω
        t     : время
    Return:
        Значение фильтра F_{Δ1}
    """
    return (Omega * t) / (2 * np.pi * (Omega**2 - omega**2)) + delta_filter_delta_1(omega, Omega, t)

def filter_gamma_2(omega, Omega, t):
    """
    Функция фильтрации F_{Γ2}(ω, Ω, t).
    Args:
        omega : частота ω
        Omega : частота Ω
        t     : время
    Возвращает:
        значение фильтра F_{Γ2}.
    """
    return (2 * np.cos(Omega * t) / (np.pi * (omega**2 - Omega**2))) * \
           np.sin(0.5 * (omega - Omega) * t) * np.sin(0.5 * (omega + Omega) * t)

def filter_delta_2(omega, Omega, t):
    """
    Фильтр F_{Δ2}(ω, Ω, t). 
    Args:
        omega : частота ω
        Omega : частота Ω
        t     : время
    Return:
        Значение фильтра F_{Δ2}
    """
    return (2 * np.sin(Omega * t)/(np.pi * (omega**2 - Omega**2))) * \
           np.sin(0.5 * (omega - Omega) * t) * np.sin(0.5 * (omega + Omega) * t)

def Gamma_n(n, t, Omega, c, tau_c, wmax=500, limit=1000):
    """
    Вычисляет коэффициент Γ_n(t).
    
    Γ_n(t) = ∫ S(ω) F_{Γn}(ω, Ω, t) dω
    
    Аргументы:
        n     : номер (1 или 2)
        t     : время
        Omega : частота Ω
        wmax  : предел интегрирования (численно аппроксимирует ∞)
    
    Возвращает:
        значение Γ_n(t).
    """
    if n == 1:
        integrand = lambda w: S_omega(w, c, tau_c) * filter_gamma_1(w, Omega, t)
    elif n == 2:
        integrand = lambda w: S_omega(w, c, tau_c) * filter_gamma_2(w, Omega, t)
    else:
        raise ValueError("n must be 1 or 2")
    return quad(integrand, -wmax, wmax, limit=limit)[0]

def Delta_n(n, t, c, tau_c, Omega, wmax=50, limit=500):
    """
    Вычисляет коэффициент Δ_n(t).
    
    Δ_n(t) = ∫ S(ω) F_{Δn}(ω, Ω, t) dω
    
    Аргументы:
        n     : номер (1 или 2)
        t     : время
        Omega : частота Ω
        wmax  : предел интегрирования (численно аппроксимирует ∞)
    
    Возвращает:
        значение Δ_n(t).
    """
    if n == 1:
        integrand = lambda w: S_omega(w, c, tau_c) * filter_delta_1(w, Omega, t)
    elif n == 2:
        integrand = lambda w: S_omega(w, c, tau_c) * filter_delta_2(w, Omega, t)
    else:
        raise ValueError("n must be 1 or 2")
    return quad(integrand, -wmax, wmax, limit=limit)[0]

def trace_dist_analitick(Gamma_hat, Gamma_bar, Delta_hat, Delta_bar):
    """
    Следовое расстояние для двух гейтов через функции фильтрации

    Аргументы:
        Gamma_hat : float
            Значение Γ̂1
        Gamma_bar : float
            Значение Γ̄1
        Delta_hat : float
            Значение Δ̂1
        Delta_bar : float
            Значение Δ̄1

    Возвращает:
        float : значение T
    """
    term1 = (np.exp(-Gamma_hat) - np.exp(-Gamma_bar)) / 4.0
    term2 = 0.5 * (np.exp(-Gamma_hat) + np.exp(-Gamma_bar) - \
                   2 * np.exp(-(Gamma_hat + Gamma_bar) / 2.0) * \
                   np.cos((Delta_hat - Delta_bar) / 2.0))
    return np.abs(term1) + 0.5 * (term2)**0.5

def simulate_delta(N_steps, dt, t_start, tau_c, c, delta0=0.0, random_seed=None):
    """
    Симуляция стохастического процесса δ(t), описанного уравнением Орнштейна–Уленбека:

        dδ(t)/dt = -(1/τ_c) δ(t) + sqrt(c) ξ(t),       (SDE)

    где ξ(t) — белый гауссовский шум с корреляцией:
        <ξ(t) ξ(t')> = δ(t - t').

    Эквивалентная дискретная рекуррентная форма (см. формулу (16)):

    δ(t_{n+1}) = δ(t_n) * exp(-dt/tau_c)
               + sqrt(c*tau_c/2 * (1 - exp(-2*dt/tau_c))) * u_n
    
    где u_n ~ N(0,1).    
    Args:
        N_steps(int): количество шагов моделирования
        dt(float): шаг по времени Δt
        tau_c(float): корреляционное время τ_c
        c(float): параметр интенсивности шума
        delta0(float): начальное значение δ(t=0)
        random_seed(int/None): для воспроизводимости
    Returns:
        t    : массив времени
        delta: массив значений δ(t)
    """

    if random_seed is not None:
        np.random.seed(random_seed)
    
    # массивы
    t = np.arange(t_start, t_start + N_steps * dt , dt)
    delta = np.zeros_like(t, dtype=float)
    delta[0] = delta0
    
    # коэффициенты
    exp_factor = np.exp(-dt/tau_c)
    noise_prefactor = (c * tau_c / 2 * (1 - exp_factor**2))**0.5
    
    # генерация шума
    for n in range(N_steps-1):
        u_n = np.random.normal(0, 1)
        delta[n+1] = delta[n] * exp_factor + noise_prefactor * u_n
    
    return t, delta

def haar_random_state(d=2):
    # Случайный комплексный вектор
    z = np.random.normal(size=(d,)) + 1j*np.random.normal(size=(d,))
    # Нормализация
    z /= np.linalg.norm(z)
    return z

def fidelity(rho, sigma):
    """Uhlmann fidelity between two density matrices"""
    sqrt_rho = sqrtm(rho)
    product = sqrt_rho @ sigma @ sqrt_rho
    return (np.trace(sqrtm(product))**2).real

def infidelity(rho, sigma):
    """Uhlmann infidelity"""
    return 1 - fidelity(rho, sigma)


def algorithm_Heun(dt, t_start, N_steps, q, g, tau_c, c, X_0):
    """
    Один шаг схемы для X(t).

    Parameters
    ----------
    X_ti : float or np.ndarray
        Значение X в момент времени ti.
    ti : float
        Время текущего шага.
    dt : float
        Шаг по времени (Δt).
    q : callable
        Функция q(t, X).
    g : callable
        Функция g(t, X).
    kappa, ell : float or np.ndarray
        Константы (смещение).
    eta_dt : callable
        Функция генерации шума η_Δt(t).

    Returns
    -------
    X_next : float or np.ndarray
        Значение X в момент времени ti+1.
    """

    # шум
    t, eta_val = simulate_delta(N_steps, dt, t_start, tau_c, c, delta0=0.0, random_seed=None)
    eta_val *= dt #/2
    # eta_val= 0.5 * dt * (eta_val[:-1] + eta_val[1:])
    
    X_stoch = [X_0]
    for i, ti in enumerate(t):
        
        q_i = q(ti, X_stoch[i])
        g_i = g(ti, X_stoch[i])

        kappa = dt * q_i
        ell = eta_val[i] * g_i 
        q_part = q_i + q(ti + dt, X_stoch[i] + kappa + ell)
        g_part = g_i + g(ti + dt, X_stoch[i] + kappa + ell)
    
        X_stoch_new = X_stoch[i] + 0.5 * dt * q_part + 0.5 * eta_val[i] * g_part
        # print(np.sum(np.abs(X_stoch_new)**2))
        X_stoch_new /= (np.sum(np.abs(X_stoch_new)**2))**0.5
        X_stoch.append(X_stoch_new)

    
    return np.array(X_stoch)

def evol_Heun(M_mc, dt, t_start, N_steps, q, g, tau_c, c, r_start=None):

    if r_start is None:
        r_start = haar_random_state(2)
    
    r_00 = 0
    r_01 = 0
    r_10 = 0
    r_11 = 0

    for i in range(M_mc):
        amplitude_state = algorithm_Heun(dt, t_start, N_steps, q, g, tau_c, c, X_0=r_start)
        c_1 = amplitude_state[:, 0]
        c_2 = amplitude_state[:, 1]

        r_00 += np.abs(c_1)**2
        r_01 += c_1 * c_2.conj()
        r_10 += c_2 * c_1.conj()
        r_11 += np.abs(c_2)**2
    

    return np.array([r_00, r_01, r_10, r_11], dtype=complex) / M_mc

def non_mark_noise(t_j, rabi, c, tau_c, r_start):

    state_plus = np.array([[1],[1]])/2**0.5
    state_minus = np.array([[1],[-1]])/2**0.5

    delt_1 = Delta_n(1, t_j, c, tau_c, rabi)
    prob = (1 - np.exp(-Gamma_n(1, t_j, rabi, c, tau_c)))
    paul_x = np.array([[0, 1], [1, 0]], dtype=complex)

    K_1 = prob**0.5 * (np.cos(rabi * t_j) * state_plus @ state_minus.T.conj() +\
                       1j *np.sin(rabi * t_j) * state_minus @ state_plus.T.conj()) / 2**0.5
    K_2 = prob**0.5 * (np.cos(rabi * t_j) * state_minus @ state_plus.T.conj() +\
                       1j *np.sin(rabi * t_j) * state_plus @ state_minus.T.conj()) / 2**0.5
    K_3 = np.exp(-1j / 4 * delt_1 * paul_x) * ((1 - prob)**0.5 * state_minus @ state_minus.T.conj() +\
                       state_plus @ state_plus.T.conj()) / 2**0.5
    K_4 = np.exp(-1j / 4 * delt_1 * paul_x) * ((1 - prob)**0.5 * state_plus @ state_plus.T.conj() +\
                       state_minus @ state_minus.T.conj()) / 2**0.5
    
    K_list = [K_1, K_2, K_3, K_4]

    r_finish = np.zeros((2, 2), dtype=complex)
    for K in K_list:
        r_finish += K @ r_start @ K.T.conj()
    
    return r_finish