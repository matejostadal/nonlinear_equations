import pandas as pd
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


newton_data = {"x": [], "f_x": [], "g_x": []}
secant_data = {"x": [], "f_x": []}
regula_falsi_data = {"x": [], "f_x": []}

acc_deviation = 10 ** (-6)


def f(x):
    return x**3 + 4 * (x**2) - 10


def f_derivative(x):
    return 3 * (x**2) + 8 * x


def g(x):
    """The form of g function with the usage of f."""
    return x - f(x) / f_derivative(x)


def add_data(data, values):
    """Adds values into the global variable."""
    for key, val in zip(data.keys(), values):
        data[key].append(val)


def newton_method(x, max_iterations=100):
    """Performs newton method method for an equation and finds the approximate solution."""
    add_data(newton_data, [x, f(x), g(x)])

    # next iteration
    iteration = 1
    next_x = g(x)

    # deviation check
    while abs(x - next_x) > acc_deviation:

        add_data(newton_data, [next_x, f(next_x), g(next_x)])

        # max iterations check
        if iteration >= max_iterations:
            print(f"\nCalculation was stopped. The max number of iterations was reached ({max_iterations} iterations):")
            # raise RuntimeError(f"Calculation was stopped. The max number of iterations was reached ({max_iterations} iterations).")
            return  

        # next iteration
        iteration += 1
        x = next_x
        next_x = g(x)

    add_data(newton_data, [x, f(x), g(x)])

    return next_x


def secant_function(x_1, x_2):
    """Calculates the new x value using the suitable form."""
    return x_2 - ((x_2 - x_1) / (f(x_2) - f(x_1))) * f(x_2)


def secant_method(x_1, x_2, max_iterations=100):
    """Performs secant method method for an equation and finds the approximate solution."""
    add_data(secant_data, [x_1, f(x_1)])
    add_data(secant_data, [x_2, f(x_2)])

    # next iteration
    iteration = 2
    next_x = secant_function(x_1, x_2)

    # deviation check
    while abs(x_2 - next_x) > acc_deviation:
        
        add_data(secant_data, [next_x, f(next_x)])

        # max iterations check
        if iteration >= max_iterations:
            print(f"\nCalculation was stopped. The max number of iterations was reached ({max_iterations} iterations):")
            # raise RuntimeError(f"Calculation was stopped. The max number of iterations was reached ({max_iterations} iterations).")
            return 
        
        # next iteration
        iteration += 1
        x_1 = x_2
        x_2 = next_x
        next_x = secant_function(x_1, x_2)

    add_data(secant_data, [next_x, f(next_x)])

    return next_x


def regula_falsi_method(x_k, x_s):
    """Performs regula falsi method for an equation and finds the approximate solution."""
    # checking initial interval
    if f(x_k) * f(x_s) > 0:
        raise ValueError(
            f"Invalid starting interval. f(x_k) and f(x_s) are of the same sign: f({x_k}) = {f(x_k)}, f({x_s}) = {f(x_k)}")

    add_data(regula_falsi_data, [x_k, f(x_k)])
    add_data(regula_falsi_data, [x_s, f(x_s)])

    next_x = secant_function(x_k, x_s)

    # new x_s change if possible
    if f(next_x) * f(x_s) > 0:
        x_s = x_k

    # deviation check
    while abs(x_k - next_x) > acc_deviation:
        
        add_data(regula_falsi_data, [next_x, f(next_x)])

        # next iteration
        x_k = next_x
        next_x = secant_function(x_k, x_s)
        
        # new x_s change if possible
        if f(next_x) * f(x_s) > 0:
            x_s = x_k

    add_data(regula_falsi_data, [next_x, f(next_x)])

    return next_x


# Newton method test
newton_method(1)
# newton_method(3)

df = pd.DataFrame(newton_data)
print(f"\n{df}\n")


# Secant method test
secant_method(1, 2)
# secant_method(-2, 2)
# secant_method(1, 2, max_iterations=3)
 
df = pd.DataFrame(secant_data)
print(f"\n{df}\n")


# Regula falsi method test
# regula_falsi_method(1, 2)
regula_falsi_method(0, 2)

df = pd.DataFrame(regula_falsi_data)
print(f"\n{df}\n")
