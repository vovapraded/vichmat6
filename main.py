import numpy as np
import matplotlib.pyplot as plt

# --- Определения уравнений и их точных решений ---

def ode1(x, y):
    return -y + x

def ode1_exact(x, x0, y0):
    return x - 1 + (y0 - (x0 - 1)) * np.exp(-(x - x0))

def ode2(x, y):
    return y * np.sin(x)

def ode2_exact(x, x0, y0):
    return y0 * np.exp(-np.cos(x0) + np.cos(x))

def ode3(x, y):
    return x**2 - y

def ode3_exact(x, x0, y0):
    C = y0 - (x0**2 - 2*x0 + 2)
    return (x**2 - 2*x + 2) + C * np.exp(-(x - x0))

ODES = [
    ("y' = -y + x", ode1, ode1_exact),
    ("y' = y * sin(x)", ode2, ode2_exact),
    ("y' = x^2 - y", ode3, ode3_exact)
]

# --- Численные методы ---

def euler(f, x0, y0, xn, h):
    n = int(np.round((xn - x0) / h))
    x = np.linspace(x0, xn, n+1)
    y = np.zeros(n+1)
    y[0] = y0
    for i in range(n):
        y[i+1] = y[i] + h * f(x[i], y[i])
    return x, y

def runge_kutta4(f, x0, y0, xn, h):
    n = int(np.round((xn - x0) / h))
    x = np.linspace(x0, xn, n+1)
    y = np.zeros(n+1)
    y[0] = y0
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h/2, y[i] + h*k1/2)
        k3 = f(x[i] + h/2, y[i] + h*k2/2)
        k4 = f(x[i] + h, y[i] + h*k3)
        y[i+1] = y[i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return x, y

def milne(f, x0, y0, xn, h):
    n = int(np.round((xn - x0) / h))
    x = np.linspace(x0, xn, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    # "Разгон" методом Рунге-Кутты 4-го порядка
    for i in range(min(4, n)):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h/2, y[i] + h*k1/2)
        k3 = f(x[i] + h/2, y[i] + h*k2/2)
        k4 = f(x[i] + h, y[i] + h*k3)
        y[i+1] = y[i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    f_vals = [f(x[j], y[j]) for j in range(4)]  # f(x0, y0), ..., f(x3, y3)

    for i in range(4, n+1):
        # Шаг прогноза
        y_pred = y[i-4] + (4*h/3)*(2*f_vals[i-3] - f_vals[i-2] + 2*f_vals[i-1])
        f_pred = f(x[i], y_pred)
        # Шаг коррекции
        y_corr = y[i-2] + (h/3)*(f_vals[i-2] + 4*f_vals[i-1] + f_pred)
        y[i] = y_corr
        # Добавляем новое значение f для следующей итерации
        f_vals.append(f(x[i], y[i]))
    return x, y

# --- Оценка точности ---

def runge_rule(f_method, f, x0, y0, xn, h, order):
    _, y1 = f_method(f, x0, y0, xn, h)
    _, y2 = f_method(f, x0, y0, xn, h/2)
    y2 = y2[::2]
    runge_est = np.max(np.abs((y2 - y1) / (2**order - 1)))
    return runge_est

def exact_error(x, y_num, y_exact_func, x0, y0):
    y_exact = y_exact_func(x, x0, y0)
    return np.max(np.abs(y_num - y_exact)), y_exact

# --- Автоматический подбор шага ---

def auto_method(method, f, x0, y0, xn, eps, f_exact=None, is_runge=False, order=1, h_start=0.25):
    h = h_start
    h_min = 1e-6
    max_iter = 20
    for _ in range(max_iter):
        x, y = method(f, x0, y0, xn, h)
        if is_runge:
            err = runge_rule(method, f, x0, y0, xn, h, order)
        else:
            err, _ = exact_error(x, y, f_exact, x0, y0)
        if err <= eps:
            return x, y, h, err
        h /= 2
        if h < h_min:
            print("Внимание: минимальный шаг достигнут!")
            break
    return x, y, h, err

def input_float(prompt, min_value=None):
    while True:
        try:
            val = float(input(prompt))
            if min_value is not None and val < min_value:
                print(f"Значение должно быть не меньше {min_value}")
                continue
            return val
        except ValueError:
            print("Некорректный ввод, попробуйте еще раз.")

def main():
    print("Численное решение ОДУ. Выберите уравнение:")
    for idx, (desc, *_ ) in enumerate(ODES):
        print(f"{idx+1}) {desc}")
    while True:
        try:
            ode_idx = int(input("Номер уравнения: ")) - 1
            if 0 <= ode_idx < len(ODES):
                break
            else:
                print("Некорректный номер.")
        except ValueError:
            print("Введите число.")

    desc, f, f_exact = ODES[ode_idx]
    x0 = input_float("Введите x0: ")
    y0 = input_float("Введите y0: ")
    xn = input_float("Введите xn (> x0): ", x0 + 1e-8)
    h0 = input_float("Введите стартовый шаг h (> 0): ", 1e-8)
    eps = input_float("Введите точность eps (> 0): ", 1e-12)

    # Автоматический подбор шага для каждого метода
    print("\nАвтоматический подбор шага для метода Эйлера...")
    x_euler_auto, y_euler_auto, h_euler, runge_err_euler = auto_method(euler, f, x0, y0, xn, eps, is_runge=True, order=1, h_start=h0)
    err_euler_auto, _ = exact_error(x_euler_auto, y_euler_auto, f_exact, x0, y0)

    print("Автоматический подбор шага для метода Рунге-Кутта 4 порядка...")
    x_rk_auto, y_rk_auto, h_rk, runge_err_rk = auto_method(runge_kutta4, f, x0, y0, xn, eps, is_runge=True, order=4, h_start=h0)
    err_rk_auto, _ = exact_error(x_rk_auto, y_rk_auto, f_exact, x0, y0)

    print("Автоматический подбор шага для метода Милна...")
    x_milne_auto, y_milne_auto, h_milne, err_milne = auto_method(milne, f, x0, y0, xn, eps, f_exact=f_exact, is_runge=False, h_start=h0)
    err_milne_auto, _ = exact_error(x_milne_auto, y_milne_auto, f_exact, x0, y0)

    # Создание сетки для таблицы с шагом h0
    n_table = int(np.round((xn - x0) / h0)) + 1
    x_table = np.linspace(x0, xn, n_table)

    # Интерполяция значений для таблицы
    y_euler_table = np.interp(x_table, x_euler_auto, y_euler_auto)
    y_rk_table = np.interp(x_table, x_rk_auto, y_rk_auto)
    y_milne_table = np.interp(x_table, x_milne_auto, y_milne_auto)

    # Вычисление ошибок для таблицы
    err_euler_table, _ = exact_error(x_table, y_euler_table, f_exact, x0, y0)
    err_rk_table, _ = exact_error(x_table, y_rk_table, f_exact, x0, y0)
    err_milne_table, _ = exact_error(x_table, y_milne_table, f_exact, x0, y0)

    # Таблица значений на сетке с шагом h0
    print("\nТаблица значений (шаг h = {:.5g}):".format(h0))
    print("   x      Точное      Эйлер      РК-4     Милн")
    for i in range(len(x_table)):
        x = x_table[i]
        print(f"{x:7.4f}  {f_exact(x, x0, y0):9.5f}  {y_euler_table[i]:9.5f}  {y_rk_table[i]:9.5f}  {y_milne_table[i]:9.5f}")

    # Вывод информации о шагах и точности
    print("\nИспользованный шаг (h) для автоматического подбора:")
    print(f"  Эйлер:       {h_euler:.5g}")
    print(f"  РК-4:        {h_rk:.5g}")
    print(f"  Милна:       {h_milne:.5g}")

    print("\nОценка точности (автоматический подбор шага, контроль через eps):")
    print("Одношаговые методы (Эйлер, Рунге-Кутта 4): по правилу Рунге")
    print(f"  Эйлер:   оценка ошибки по правилу Рунге = {runge_err_euler:.2e} (<= eps: {eps})")
    print(f"  РК-4:    оценка ошибки по правилу Рунге = {runge_err_rk:.2e} (<= eps: {eps})")
    print("Многошаговый метод (Милна): по сравнению с точным решением")
    print(f"  Милна:   max|точн - числ| = {err_milne:.2e} (<= eps: {eps})")
    print("\nГлобальные ошибки для таблицы (шаг h = {:.5g}):".format(h0))
    print(f"  Эйлер:   max|точн - числ| = {err_euler_table:.2e}")
    print(f"  РК-4:    max|точн - числ| = {err_rk_table:.2e}")
    print(f"  Милна:   max|точн - числ| = {err_milne_table:.2e}")

    # График (используем автоматические шаги для большей точности)
    plt.plot(x_euler_auto, f_exact(x_euler_auto, x0, y0), label='Точное решение', color='black', linewidth=2)
    plt.plot(x_euler_auto, y_euler_auto, 'o-', label=f'Эйлер (h={h_euler:.2g})', alpha=0.7)
    plt.plot(x_rk_auto, y_rk_auto, 's-', label=f'РК-4 (h={h_rk:.2g})', alpha=0.7)
    plt.plot(x_milne_auto, y_milne_auto, 'x-', label=f'Милн (h={h_milne:.2g})', alpha=0.7)
    plt.legend()
    plt.title(f"Решение ОДУ: {desc}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()