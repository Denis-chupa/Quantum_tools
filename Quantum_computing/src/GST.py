import numpy as np
from scipy.integrate import quad



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

def Gamma_n(n, t, Omega, wmax=50, limit=500):
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
        integrand = lambda w: S_omega(w) * filter_gamma_1(w, Omega, t)
    elif n == 2:
        integrand = lambda w: S_omega(w) * filter_gamma_2(w, Omega, t)
    else:
        raise ValueError("n must be 1 or 2")
    return quad(integrand, -wmax, wmax, limit=limit)[0]

def Delta_n(n, t, Omega, wmax=50, limit=500):
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
        integrand = lambda w: S_omega(w) * filter_delta_1(w, Omega, t)
    elif n == 2:
        integrand = lambda w: S_omega(w) * filter_delta_2(w, Omega, t)
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

def simulate_delta(N_steps, dt, tau_c, c, delta0=0.0, random_seed=None):
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
    t = np.arange(0, N_steps * dt, dt)
    delta = np.zeros_like(t)
    delta[0] = delta0
    
    # коэффициенты
    exp_factor = np.exp(-dt/tau_c)
    noise_prefactor = (c * tau_c / 2 * (1 - exp_factor**2))**0.5
    
    # генерация шума
    for n in range(N_steps-1):
        u_n = np.random.normal(0, 1)
        delta[n+1] = delta[n] * exp_factor + noise_prefactor * u_n
    
    return t, delta
