import pandas as pd
import numpy as np


pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


iterative_data = {"x_1": [], "x_2": [], "f_1": [], "f_2": []}
newton_data = {"x_1": [], "x_2": [], "f_1": [], "f_2": []}


acc_deviation = 10 ** (-10)


def F(x_1, x_2):
    """Represents the system of equations."""
    return [x_1**2 - 2 * x_1 - x_2 + 0.5,
            x_1**2 + 4 * (x_2**2) - 4]


def G(x_1, x_2):
    """Represents the g functions."""
    return [(x_1**2 - x_2 + 0.5) / 2,
            (- x_1**2 - 4 * (x_2**2) + 8 * x_2 + 4) / 8]


def J(x_1, x_2):
    """Represents the Jacobi matrix corresponding with a system."""
    return [[2 * x_1 - 2, -1],
            [2 * x_1, 8 * x_2]]
    

def add_data(data, values):
    """Adds values into the global variable."""
    for key, val in zip(data.keys(), values):
        data[key].append(val)


def acceptable_deviation(x_vals, next_x_vals):
    """Checks if all values comply the max acceptable deviation."""
    for x_val, next_x_val in zip(x_vals, next_x_vals):

        if abs(x_val - next_x_val) > acc_deviation:
            return False

    return True


def iterative_method(F, G, x_vals, max_iterations=100):
    """Performs iterative method for a system of nonlinear equations. We assume that arguments are of suitable length.

        Parameters:
            F :
                function that represents the equations of a system
            G :
                function that is "equivalent" (in our understanding) to F
            x_vals :
                initial values of variables in a system
    """
    add_data(iterative_data, [*x_vals, F(*x_vals)[0], F(*x_vals)[1]])

    # next iteration
    iteration = 1
    next_x_vals = G(*x_vals)

    # deviation check
    while not acceptable_deviation(x_vals, next_x_vals):

        add_data(iterative_data, [*next_x_vals, F(*next_x_vals)[0], F(*next_x_vals)[1]])

        # max iterations check
        if iteration >= max_iterations:
            print(f"\nCalculation was stopped. The max number of iterations was reached ({max_iterations} iterations):")
            # raise RuntimeError(f"Calculation was stopped. The max number of iterations was reached ({max_iterations} iterations).")
            return 

    	# next iteration
        iteration += 1
        x_vals = next_x_vals
        next_x_vals = G(*x_vals)

    add_data(iterative_data, [*next_x_vals, F(*next_x_vals)[0], F(*next_x_vals)[1]])

    return next_x_vals


def newton_method(F, J, x_vals, max_iterations=100):
    """Performs newton method on a system of nonlinear equations. We assume that arguments are of suitable length.
        Parameters:
            F :
                function that represents the equations of a system
            J :
                jacobi matrix corresponding with the system represented by F
            x_vals :
                initial values of variables in a system
    """
    add_data(newton_data, [*x_vals, F(*x_vals)[0], F(*x_vals)[1]])

    # next iteration
    iteration = 1
    # linear equation system solving with numpy    
    deltas = np.linalg.solve(np.array(J(*x_vals)), np.array(np.negative(F(*x_vals))))
    next_x_vals = [x + delta for x, delta in zip(x_vals, deltas)]

    # deviation check
    while not acceptable_deviation(x_vals, next_x_vals):

        add_data(newton_data, [*next_x_vals, F(*next_x_vals)[0], F(*next_x_vals)[1]])

        # max iterations check
        if iteration >= max_iterations:
            print(f"\nCalculation was stopped. The max number of iterations was reached ({max_iterations} iterations):")
            # raise RuntimeError(f"Calculation was stopped. The max number of iterations was reached ({max_iterations} iterations).")
            return 

    	# next iteration
        iteration += 1
        x_vals = next_x_vals

        # linear equation system solving with numpy
        deltas = np.linalg.solve(np.array(J(*x_vals)), np.array(np.negative(F(*x_vals))))
        next_x_vals = [x + delta for x, delta in zip(x_vals, deltas)]

    add_data(newton_data, [*next_x_vals, F(*next_x_vals)[0], F(*next_x_vals)[1]])

    return next_x_vals



# Iterative method test
iterative_method(F, G, [0, 1])

# iterative_method(F, G, [0, 1], max_iterations=10)

df = pd.DataFrame(iterative_data)
print(f"\n{df}\n")


# Newton method test
newton_method(F, J, [0, 1])

df = pd.DataFrame(newton_data)
print(f"\n{df}\n")