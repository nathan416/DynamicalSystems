"""dynamical systems and chaos functions
    By: Nathan Flack
    Version: 1.6
"""
from __future__ import division

import itertools
import logging
import multiprocessing
import pprint as pp
import time
from functools import partial
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from numba import vectorize, jit, njit, cuda

import matplotlib.pyplot as plt
import mpl_scatter_density
import numpy as np
import psutil
import cupy as cp
from click import progressbar
from sympy import *
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

SEED = 100
LOGGER = logging.getLogger(__name__)
LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -43s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
DT = .004

x, y, z, t = symbols('x y z t')
k, m, n = symbols('k m n', integer=True)
f, g, h = symbols('f g h', cls=Function)


def find_fixed_points(expression):
    try:
        fixed_points = solve(Eq(expression, x), x)
    except NotImplementedError as exc:
        print(exc)
        return None
    return fixed_points


def find_attractiveness_of_fixed_points(expression: Basic, list_of_fixed_points: list) -> list:
    derivative = diff(expression, x)
    lambda_derivative = lambdify(x, derivative)
    return list(map(lambda_derivative, list_of_fixed_points))


def iterate_lambda_expression(iterating_function: callable, value: float, precision: int, max_iterations=10000, min_iterations=220) -> list:
    """iterates over a function with the initial value and returns
        the list of iterated values

    Args:
        expression (callable)): function of x to be iterated
        value (float): initial value
        precision (int): decimal precision
        max_iterations (int): amount of iterations

    Returns:
        list: list of iterated values
    """
    iterating_value = iterating_function(value)
    last_iterating_value = iterating_value + 1.0
    iterate_list = []
    iterate_list.append(value)
    iterate_list.append(iterating_value)
    count = 0

    while ((abs(last_iterating_value - iterating_value) > 10**(-1 * precision) or count < min_iterations) and count < max_iterations):
        last_iterating_value = iterating_value
        try:
            iterating_value = iterating_function(iterating_value)
        except OverflowError:
            iterating_value = np.NAN
        if iterating_value < 1e200 and iterating_value > -1e200:
            iterate_list.append(iterating_value)
        else:
            break
        count += 1
    return iterate_list


def iterate_array_expression(iterating_function: callable, initial_values: np.ndarray, iteration_count=10000) -> list:
    """iterates over a function with the initial value and returns
        the list of iterated values

    Args:
        expression (numpy.NDArray): function of x to be iterated
        initial_values (numpy.NDArray): array of initial values
        max_iterations (int): amount of iterations

    Returns:
        list: list of iterated values
    """
    iterating_values = initial_values

    for iteration in range(iteration_count):
        iterating_values = iterating_function(iterating_values)
    return iterating_values


def iterations_till_divergence(expression: callable, initial_values: np.ndarray, iteration_count=1000, comp_method='cpu') -> list:
    """ iterates over a function with the initial value and returns
        a list of when the modulus of each complex value goes above 4

    Args:
        expression (ufunc): function of x to be iterated
        initial_values (numpy.NDArray): array of initial values
        max_iterations (int): amount of iterations

    Returns:
        list: list of when each value diverges
    """
    iterating_values_h = initial_values
    divergence_h = np.zeros(len(iterating_values_h), dtype=np.int32)

    # @cp.fuse(kernel_name='helper_func')
    def helper_func(complex_value, divergence):
        try:
            if complex_value.real**2 + complex_value.imag**2 < 4:
                return divergence + 1
            else:
                return divergence
        except OverflowError:
            return divergence
    
    @cuda.jit('void(complex128[:], int32[:])')
    def helper_func5(complex_value, divergence):
        # if complex_value[i].real**2 + complex_value[i].imag**2 < 4:
        #     divergence[i] = divergence[i] + 1
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        for i in range(start, complex_value.shape[0], stride): 
            if complex_value[i].real**2 + complex_value[i].imag**2 < 2:
                divergence[i] = divergence[i] + 1
    
    helper_func3 = cp.ElementwiseKernel(
    'float64 x, float64 y, int32 d',
    'int32 z',
    '''
    if (x*x + y*y < 4) {
        z = d + 1;
    }
    else{
        z = d;
    }
    ''',
    'helper_func3')
    
    # @cp.fuse(kernel_name='helper_func2')
    def helper_func2(complex_value):
        try:
            if complex_value.real**2 + complex_value.imag**2 >= 4:
                return np.NaN
            else:
                return complex_value
        except OverflowError:
            return np.NaN
    
    helper_func4 = cp.ElementwiseKernel(
    'complex128 x',
    'complex128 z',
    '''
    #include <math.h>
    if (creal(x)*creal(x) + cimag(x)*cimag(x) >= 4) {
        z = logf(-1);
    }
    else { 
        z = x;
    }
    ''',
    'helper_func4')
    
    helper = np.vectorize(helper_func, otypes=[np.int16])
    helper2 = np.vectorize(helper_func2, otypes=[np.complex128])
    griddim = 64
    blockdim = 64
    if comp_method == 'gpu':
        iterating_values_d = cuda.to_device(iterating_values_h)
        divergence_d = cuda.to_device(divergence_h)
    for iteration in tqdm(range(iteration_count)):
        if comp_method == 'gpu':
            expression[griddim, blockdim](iterating_values_d)
            helper_func5[griddim, blockdim](iterating_values_d, divergence_d)
        else:
            iterating_values_h = expression(iterating_values_h)
            divergence_h = helper(iterating_values_h, divergence_h)
        # iterating_values = helper2(iterating_values)
    if comp_method == 'gpu':
        divergence_h = divergence_d.copy_to_host()
    return divergence_h


def iterate_expression(expression, value: float, precision: int, max_iterations=10000):
    """iterates over a function with the initial value and returns
        the list of iterated values

    Args:
        expression (SymPy object)): function of x to be iterated
        value (float): initial value
        precision (int): decimal precision
        max_iterations (int): amount of iterations

    Returns:
        list: list of iterated values
    """
    iterating_function = lambdify(x, expression)
    iterating_value = iterating_function(value)
    last_iterating_value = iterating_value + 1
    iterate_list = [value, iterating_value]
    count = 0

    while (abs(last_iterating_value - iterating_value) > 10**(-1 * precision) and count < max_iterations):
        last_iterating_value = iterating_value
        iterating_value = iterating_function(
            iterating_value)
        if iterating_value < 1e200 and iterating_value > -1e200:
            iterate_list.append(float(iterating_value))
        else:
            break
        count += 1
    return iterate_list


def full_iterate_expression(expression, value, max_iterations=10000, min_iterations=100):
    """iterates over a function a minimum of 220 times with the initial value and returns
    the list of iterated values

    Args:
        expression (SymPy object)): function of x to be iterated
        value (float): initial value
        precision (int): decimal precision
        max_iterations (int): amount of iterations

    Returns:
        list: list of iterated values
    """
    iterating_function = lambdify(x, expression)
    iterating_value = iterating_function(value)
    iterate_list = [value, iterating_value]
    count = 0

    while count < min_iterations and count < max_iterations:
        iterating_value = iterating_function(iterating_value)
        if abs(iterating_value.real) < 1e200:
            iterate_list.append(iterating_value)
        else:
            break
        count += 1
    return iterate_list


def lab_iterate_expression(expression, value, precision):
    """iterates over a function with the initial value and returns
        the list of iterated values

    Args:
        expression (SymPy object)): function of x to be iterated
        value (float): initial value
        precision (int): amount of iterations

    Returns:
        list: list of iterated values
    """
    fixed_points = find_fixed_points(expression)
    result = expression.subs(x, value)
    iterate_list = [value, result]

    iterating_value = value
    last_iterating_value = iterating_value + 1
    count = 0

    while abs(last_iterating_value - iterating_value) > 10**(-1 * precision) and iterating_value not in fixed_points and count < 100000:
        last_iterating_value = iterating_value
        iterating_value = expression.subs(x, iterating_value)
        if iterating_value < 1e200 and iterating_value > -1e200:
            iterate_list.append(iterating_value)
        else:
            break
        count += 1
    return iterate_list


def calculate_square(value, seed, precision):
    try:
        if value == 0:
            raise ZeroDivisionError
        success = True
        w = seed
        graph1 = []
        graph1.append(w)
        last_w = w + 1
        count = 0

        while round(last_w - w, precision + 1) != 0 and count < 100:
            if w == 0:
                raise ZeroDivisionError
            if count > 98:
                success = False
            last_w = w
            w = w - (w**2 - value) / (2 * w)
            graph1.append(w)
            count += 1
        return success, graph1
    except ZeroDivisionError as err:
        print('F\'(x) is 0:', err)


def iterate_over_range(expression, max):
    iterate_dict = {}
    for item in np.arange(-1 * max - 1, max + 1, .0001):
        iterate_list = iterate_expression(
            expression, item, 20)
        iterate_dict[item] = iterate_list[-1]
    return iterate_dict


def plot_graph_helper(graph_list, title, xlabel, ylabel, window_name, is_logarithm=False, has_grid=False, success=True):
    """takes a list and plots it out. takes arguments that are used in displaying the graph

    Args:
        graph_list (list): list of numbers used as the y values in the graph
        title (string): [description]
        xlabel (string): [description]
        ylabel (string): [description]
        window_name (string): [description]
        is_logarithm (bool, optional): whether to use the log scale on the y graph. Defaults to False.
        has_grid (bool, optional): displays red grid lines. Defaults to False.
        success (bool, optional): [description]. Defaults to True.
    """
    font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
    font2 = {'family': 'serif', 'color': 'darkred', 'size': 15}
    fig, ax = plt.subplots(num=window_name)

    if is_logarithm:
        ax.semilogy(range(0, len(graph_list)), graph_list)
    else:
        ax.plot(range(0, len(graph_list)), graph_list)

    plt.title(title, fontdict=font1)
    plt.xlabel(xlabel, fontdict=font2)
    plt.ylabel(ylabel, fontdict=font2)
    if has_grid:
        ax.grid(color='r', linestyle='-', linewidth=.5)

    if success:
        ax.text(.3, .7, f'Last Value: {float(graph_list[-1])}\nSecond to Last: {float(graph_list[-2])}', transform=ax.transAxes)
    else:
        ax.text(int(len(graph_list) / 2), 0.7 * SEED, 'unable to calculate')
    fig.set_size_inches(19, 9.5)
    return ax, fig


def plot_iterate_graph(precision, expression, value, is_logarithm=False, max_iterations=100000, has_grid=False):
    """puts arguments into iterate() function then plots the resulting graph

    Args:
        precision (int): [description]
        expression (SymPy object): function of x to be iterated
        value (float): initial value
        range_amount (int): times that the function should be iterated
        logarithm (bool, optional): whether the y axis should be logarithmic. Defaults to False.

    Returns:
        list: graphed list
    """
    graph_list = iterate_expression(
        expression, value, precision, max_iterations)

    plot_graph_helper(graph_list, f'F(x) = {expression}, x0 = {value}', "Iterations", "Value",
                      f'Iteration of F(x) = {expression}, x0 = {value}', is_logarithm, has_grid)
    return graph_list


def plot_full_iterate_graph(expression, initial_value, max_iterations=1000, min_iterations=100, is_logarithm=False, has_grid=False):
    """puts arguments into iterate() function then plots the resulting graph

    Args:
        precision (int): [description]
        expression (SymPy object): function of x to be iterated
        initial_value (float): initial value
        range_amount (int): times that the function should be iterated
        logarithm (bool, optional): whether the y axis should be logarithmic. Defaults to False.

    Returns:
        list: graphed list
    """
    graph_list = full_iterate_expression(expression, initial_value, max_iterations, min_iterations)

    plot_graph_helper(graph_list, f'F(x) = {expression}, x0 = {initial_value}', "Iterations", "Value",
                      f'Iteration of F(x) = {expression}, x0 = {initial_value}', is_logarithm, has_grid)
    return graph_list


def plot_square_graph(precision, value):
    success, graph_list = calculate_square(value, SEED, precision)
    plot_graph_helper(graph_list, "Newton's Method", "Iterations", "Value", 'Square of ' + str(value) + ' with precision of ' + str(precision))
    return graph_list


def find_basin_of_attraction(expression, seed, precision):
    """finds the basin of attraction

    Args:
        expression (SymPy object): [description]
        seed (float): seed value
        precision (int): decimal precision

    Returns:
        list: list of iterated values
    """
    try:
        # fixed_points = find_fixed_points(expression)
        derivative = diff(expression, x)
        derivative_function = lambdify(x, derivative)
        expression_function = lambdify(x, expression)
        if seed == 0:
            raise ZeroDivisionError
        w = seed
        graph1 = []
        graph1.append(w)
        last_w = w + 1
        count = 0

        # while round(last_w - w, precision + 1) != 0 and count < 100:
        while count < 10:
            if derivative_function(w) == 0:
                raise ZeroDivisionError
            if count > 98:
                success = False
            last_w = w
            w = w - expression_function(w) / derivative_function(w)
            graph1.append(w)
            count += 1
        return graph1
    except ZeroDivisionError as err:
        print('F\'(x) is 0:', err)


def plot_basin_graph(precision, expression, value, logarithm=False):
    """puts arguments into find_basin_of_attraction() function then plots the resulting graph

    Args:
        precision (int): [description]
        expression (SymPy object): function of x
        value (float): seed value
        logarithm (bool, optional): whether the y axis should be logarithmic. Defaults to False.

    Returns:
        boolean: success
        list: graphed list
    """
    graph_list = find_basin_of_attraction(
        expression, value, precision)

    plot_graph_helper(graph_list, f'F(x) = {expression}, x0 = {value}', "Iterations", "Value", f'Basin of attraction of F(x) = {expression}, seed = {value}', logarithm)
    return graph_list


def plot_cobweb_graph(expression, initial_value, max_iterations, low_range, high_range):
    font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
    font2 = {'family': 'serif', 'color': 'darkred', 'size': 15}
    graph_list = iterate_lambda_expression(expression, initial_value, 8, max_iterations)
    t = np.linspace(low_range - 10, high_range + 10, 10000)

    values = list(map(expression, t))
    fig, ax = plt.subplots(
        num=f'Cobweb plot of F(x) = {expression}, x0 = {initial_value}')
    plt.title(f'F(x) = {expression}, x0 = {initial_value}', fontdict=font2)
    plt.xlabel('x', fontdict=font2)
    plt.ylabel('y', fontdict=font2)
    ax.plot(t, values)
    ax.plot(t, t)
    ax.grid()
    ax.plot([initial_value, initial_value], [
            initial_value, graph_list[1]], color='r')
    ax.plot([initial_value, graph_list[1]], [
            graph_list[1], graph_list[1]], color='r')
    for loop_controller in range(1, len(graph_list) - 1):
        ax.plot([graph_list[loop_controller], graph_list[loop_controller]], [
                graph_list[loop_controller], graph_list[loop_controller + 1]], color='r')
        ax.plot([graph_list[loop_controller], graph_list[loop_controller + 1]],
                [graph_list[loop_controller + 1], graph_list[loop_controller + 1]], color='r')
    plt.xlim(low_range - 1, high_range + 1)
    t = np.linspace(low_range - 1, high_range + 1, 10000)
    values = list(map(expression, t))
    plt.ylim(min(values) - 1, max(values) + 1)
    fig.set_size_inches(19, 9.5)
    return fig, graph_list


def plot_bifurcation_graph(expression, initial_value=0.0, max_iterations=80, minimum_c=-2,
                           maximum_c=-.4, print_progressbar=False, samples=250, density_plot=False):
    time_start = time.perf_counter()
    font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
    font2 = {'family': 'serif', 'color': 'darkred', 'size': 15}

    if(density_plot):
        fig = plt.figure(
            num=f'Bifurcation plot of F(x) = {expression}, between {minimum_c} and {maximum_c}')
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    else:
        fig, ax = plt.subplots(
            num=f'Bifurcation plot of F(x) = {expression}, between {minimum_c} and {maximum_c}')

    plt.title(
        f'F(x) = {expression}, between {minimum_c} and {maximum_c}', fontdict=font1)
    plt.xlabel('y', fontdict=font2)
    plt.ylabel('x', fontdict=font2)

    def helper(bar: Tuple, expression, initial_value: float, max_iterations: int):
        x_y_array = []
        for c_value in bar:
            graph_list = list(bifurcation_iteration_helper(
                expression, initial_value, c_value, max_iterations, 100))
            for iterated_value in graph_list[50:]:
                x_y_array.append((c_value, iterated_value))
        return x_y_array

    if(print_progressbar):
        with progressbar(np.linspace(minimum_c, maximum_c, samples), fill_char='█', width=80, item_show_func=lambda x: f'x = {x} : time passed: {time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - time_start))}') as bar:
            x_y_array = helper(bar, expression, initial_value, max_iterations)
    else:
        x_y_array = helper(np.linspace(minimum_c, maximum_c, samples), expression, initial_value, max_iterations)

    x_array = []
    y_array = []
    for x_y in x_y_array:
        x_array.append(x_y[0])
        y_array.append(x_y[1])

    if(density_plot):
        norm = ImageNormalize(vmin=0., vmax=1000, stretch=LogStretch())
        ax.scatter_density(x_array, y_array, norm=norm)
        ax.set_xlim(minimum_c, maximum_c)
        # ax.set_ylim(-6, 6)
    else:
        ax.scatter(x_array, y_array, s=.1)
    fig.set_size_inches(19, 9.5)
    return ax, fig, x_array, y_array


def multiprocessing_plot_bifurcation_graph(expression, initial_value=0.0, max_iterations=80, minimum_c=-2,
                                           maximum_c=2, print_progressbar=False, samples=250, density_plot=False, cutoff=100):
    font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
    font2 = {'family': 'serif', 'color': 'darkred', 'size': 15}

    if(density_plot):
        fig = plt.figure(
            num=f'Bifurcation plot of F(x) = {expression}, between {minimum_c} and {maximum_c}')
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    else:
        fig, ax = plt.subplots(
            num=f'Bifurcation plot of F(x) = {expression}, between {minimum_c} and {maximum_c}')

    plt.title(
        f'F(x) = {expression}, between {minimum_c} and {maximum_c}, iterations: {max_iterations}, samples: {samples}', fontdict=font1)
    plt.xlabel('y', fontdict=font2)
    plt.ylabel('x', fontdict=font2)

    x_array = []
    y_array = []
    if(print_progressbar):
        x_y_array = process_map(partial(map_bifurcation_helper, expression=expression, initial_value=initial_value, max_iterations=max_iterations, cutoff=cutoff), np.linspace(
            minimum_c, maximum_c, samples), max_workers=psutil.cpu_count(logical=False), chunksize=10)
    else:
        pool = multiprocessing.Pool(psutil.cpu_count(logical=False))
        x_y_array = pool.map(partial(map_bifurcation_helper, expression=expression, initial_value=initial_value, max_iterations=max_iterations, cutoff=cutoff), np.linspace(
            minimum_c, maximum_c, samples), chunksize=10)

    x_y_array = list(itertools.chain.from_iterable(x_y_array))

    for x_y in x_y_array:
        x_array.append(x_y[0])
        y_array.append(x_y[1])

    if(density_plot):
        norm = ImageNormalize(vmin=0., vmax=1000, stretch=LogStretch())
        ax.scatter_density(x_array, y_array, norm=norm)
        ax.set_xlim(minimum_c, maximum_c)
        # ax.set_ylim(-6, 6)
    else:
        ax.scatter(x_array, y_array, s=1)
    fig.set_size_inches(19, 9.5)
    return ax, fig, x_array, y_array


def map_bifurcation_helper(c_value: float, expression, initial_value: float, max_iterations: int, cutoff: int) -> list:
    x_y_array = []
    # expr_function2 = lambdify([y, x], expression, 'numpy')
    # expr_function = lambda x : expr_function2(c_value, x)

    graph_list = bifurcation_iteration_helper(
        expression, initial_value, c_value, max_iterations, cutoff + 100)
    for iterated_value in graph_list[cutoff:]:
        x_y_array.append((c_value, iterated_value))
    return x_y_array


def bifurcation_iteration_helper(iterating_function, value: float, c_value: float, max_iterations: int = 300, min_iterations: int = 100) -> tuple:
    iterating_value = iterating_function(c_value, value)
    last_iterating_value = iterating_value + 1.0
    # iterate_list = []
    # iterate_list.append(value)
    # iterate_list.append(iterating_value)
    iterate_list = ()
    iterate_list = iterate_list + (value,)
    iterate_list = iterate_list + (iterating_value,)
    count = 0

    while ((abs(last_iterating_value - iterating_value) > 10**(-1 * 8) or count < min_iterations) and count < max_iterations):
        last_iterating_value = iterating_value
        iterating_value = iterating_function(c_value, iterating_value)
        if iterating_value < 1e200 and iterating_value > -1e200:
            iterate_list = iterate_list + (iterating_value,)
        else:
            break
        count += 1
    return iterate_list


def plot_iterated_function(expression, min, max, iterations):
    t = np.linspace(min, max, 1000)
    new_expression = lambdify(x, expression)
    values = list(map(partial(composing_fun, expression=new_expression, iterations=iterations), t))

    red_values = list()
    blue_values = list()

    for y_value in values:
        if y_value >= 0:
            blue_values.append(0)
            red_values.append(None)
        else:
            blue_values.append(None)
            red_values.append(0)
    fig, ax = plt.subplots()
    ax.plot(t, values)
    ax.plot(t, t)
    ax.scatter(t, blue_values, color='tab:blue', s=6)
    ax.scatter(t, red_values, color='tab:red', s=6)
    plt.title(f"T^{iterations}(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    ax.grid()
    plt.ylim(-0.1, 2)
    fig.set_size_inches(19, 9.5)


def composing_fun(x, expression, iterations):
    for loop_controller in range(0, iterations):
        x = expression(x)
    return x


def plot_iterate_over_range(expression, min, max):
    t = np.linspace(min, max, 100)
    values = list(map(lambdify(x, expression), t))
    fig, ax = plt.subplots()
    ax.plot(t, values)
    ax.plot(t, t)
    ax.grid()

    red_list = []
    green_list = []
    blue_list = []
    for item in np.linspace(min, max, 3000):
        graph_list = iterate_expression(expression, item, 8, 20)
        LOGGER.info(item)
        LOGGER.info(pp.pformat(graph_list))
        if graph_list[-1] > 1e6:
            blue_list.append(item)
        elif graph_list[-1] < -1e6:
            green_list.append(item)
        else:
            red_list.append(item)
    ax.scatter(blue_list, list(itertools.repeat(0, len(blue_list))), color='tab:blue')
    ax.scatter(green_list, list(itertools.repeat(0, len(green_list))), color='tab:green')
    ax.scatter(red_list, list(itertools.repeat(0, len(red_list))), color='tab:red')
    fig.set_size_inches(19, 9.5)


def liopanov_exponent(expression: Basic, input, amount):
    """
    """
    lambda_expression = lambdify(x, expression)
    iterated_list = iterate_lambda_expression(lambda_expression, input, 8, amount)
    derivative = diff(expression, x)
    lambda_derivative = lambdify(x, derivative)
    derivative_list = list(map(lambda_derivative, iterated_list))

    return np.mean(np.log(np.abs(derivative_list)))


def ddx(x, y, z):
    return 40 * (y - x)


def ddy(x, y, z):
    return (28 - 40) * x - x * z + 28 * y


def ddz(x, y, z):
    return x * y - 3 * z


def attractor(initial_x, initial_y, initial_z, iterations=1000):
    x = ddx(initial_x, initial_y, initial_z)
    y = ddy(initial_x, initial_y, initial_z)
    z = ddz(initial_x, initial_y, initial_z)

    graph_list = [(x, y, z)]

    for loop_control in range(iterations):
        new_x = x + (ddx(x, y, z) * DT)
        new_y = y + (ddy(x, y, z) * DT)
        new_z = z + (ddz(x, y, z) * DT)

        x = new_x
        y = new_y
        z = new_z

        graph_list.append((x, y, z))
        if loop_control < 20:
            LOGGER.info((x, y, z))
    return graph_list


def plot_attractor_graph(initial_x=-0.1, initial_y=0.5, initial_z=-0.6, iterations=10000):
    graph_list = attractor(initial_x, initial_y, initial_z, iterations)
    x_array = []
    y_array = []
    z_array = []
    for x_y in graph_list:
        x_array.append(x_y[0])
        y_array.append(x_y[1])
        z_array.append(x_y[2])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.set_xlim(-40, 40)
    # ax.set_ylim(0, -30)
    # ax.set_zlim(0, 40)

    ax.plot3D(x_array, y_array, z_array, 'gray')
    return fig


def replot_multi_orbit_diagram(expression, min_x, max_x, min_y=None, max_y=None):
    """
        usage example:
        expression = (1-y)*x+(4*x**6)*(np.e)**(-2*x)
        replot_orbit_diagram(expression, 0.75, .8)
        plt.tight_layout()
        plt.savefig('orbit.png', dpi=300)
        plt.show()
    """
    LOGGER.info("plotting orbit diagram")
    start_time = time.perf_counter()
    LOGGER.debug(f'Start time: {start_time}')
    ax, fig, x_array, y_array = multiprocessing_plot_bifurcation_graph(expression, 3, 1000, min_x, max_x, False, 2000, True)
    end_time = time.perf_counter()
    LOGGER.debug(f'End time: {end_time}')
    LOGGER.debug(f'time taken: {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}')
    if min_y is not None and max_y is not None:
        ax.set_ylim(min_y, max_y)


def replot_orbit_diagram(expression, min_x, max_x, min_y=None, max_y=None):
    """
        usage example:
        expression = (1-y)*x+(4*x**6)*(np.e)**(-2*x)
        replot_orbit_diagram(expression, 0.75, .8)
        plt.tight_layout()
        plt.savefig('orbit.png', dpi=300)
        plt.show()
    """
    LOGGER.info("plotting orbit diagram")
    start_time = time.perf_counter()
    LOGGER.debug(f'Start time: {start_time}')
    ax, fig, x_array, y_array = plot_bifurcation_graph(expression, 3, 1000, min_x, max_x, False, 2000, True)
    end_time = time.perf_counter()
    LOGGER.debug(f'End time: {end_time}')
    LOGGER.debug(f'time taken: {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}')
    if min_y is not None and max_y is not None:
        ax.set_ylim(min_y, max_y)


def newtons_method(expression: Basic, initial_value: float, iterations: int) -> float:
    intermediate_value = initial_value
    derivative = diff(expression, x)
    for loop_control in range(0, iterations):
        intermediate_value = intermediate_value - (expression.subs(x, intermediate_value) / derivative.subs(x, intermediate_value))
    return intermediate_value


def main():
    pass


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
