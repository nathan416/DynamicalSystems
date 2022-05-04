"""Midterm Project
    Class: CPSC 455
    By: Nathan Flack
    Version: 1.6
"""
from __future__ import division

import logging
import multiprocessing
import os
import pprint as pp
import random
import sys
import time
import unittest

import cupy as cp
import matplotlib.pyplot as plt
import mpl_scatter_density
import mpld3
import numpy as np
from scipy.interpolate import griddata
from sympy import *
from tqdm import tqdm

from dynamical import *

SEED = 100
LOGGER = logging.getLogger(__name__)
LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -43s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')


x, y, z, t = symbols('x y z t')
k, m, n = symbols('k m n', integer=True)
f, g, h = symbols('f g h', cls=Function)


class DynamicalTest(unittest.TestCase):
    """Test class for dynamical.py
    """

    def setUp(self):
        logging.basicConfig(level=logging.DEBUG,
                            format=LOG_FORMAT, filename='log_file_test.log')
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        logging.getLogger('numba.core').setLevel(logging.WARNING)

    def test_liopanuv(self):
        SUB_INTERVALS = 1000
        ITERATES = 25000
        graph_list = []
        start_time = time.perf_counter()
        LOGGER.debug(f'Start time: {start_time}')
        for c in np.linspace(0, 1, SUB_INTERVALS):
            expression = (1 - c) * x + (4 * x**6) * (np.e)**(-2 * x)
            graph_list.append(liopanov_exponent(expression, 3, ITERATES))
        end_time = time.perf_counter()
        LOGGER.debug(f'End time: {end_time}')
        LOGGER.debug(f'time taken: {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}')
        ax, fig = plot_graph_helper(graph_list, 'title', 'x', 'y', 'window')
        ax.plot([0, SUB_INTERVALS], [0, 0])
        plt.show()

    def test_lyapunov2(self):
        # should equal ln(2)
        expression = 4 * x * (1 - x)
        self.assertAlmostEqual(ln(2), liopanov_exponent(expression, .001, 50000), 2)

    def test_derive(self):
        expression = (1 - y) * x + (4 * x**6) * (np.e)**(-2 * x)
        derivative = diff(expression, x)
        LOGGER.info(derivative)

    def test_iterate(self):
        expression = (1 - y) * x + (4 * x**6) * (np.e)**(-2 * x)
        new_expression = expression.subs(y, .9)
        lambda_expression = lambdify(x, new_expression)
        iterate_list = iterate_lambda_expression(lambda_expression, 3, 8, 10, 10)
        LOGGER.info(pp.pformat(iterate_list))

    def test_complex_basin(self):
        SUB_INTERVALS = 60
        expression: Basic = x**3 + y
        comp_array = np.array([1 + 1j])

        basin_array = list(map(partial(basin_helper, expression=expression, initial_value=comp_array), np.linspace(0, 1, SUB_INTERVALS)))
        root_array = np.array([])
        for sub_array in basin_array:
            root_array = np.append(root_array, sub_array[-1])
        imag_array = np.array([])
        real_array = np.array([])
        for value in root_array:
            real_array = np.append(real_array, value.real)
            imag_array = np.append(imag_array, value.imag)

        # LOGGER.info(pp.pformat(real_array))
        # LOGGER.info(pp.pformat(imag_array))
        fig, ax = plt.subplots(
            num=f'basin plot of F(x) = {expression}, between {0} and {1}')
        ax.scatter(real_array, imag_array, s=1)
        plt.show()

    def test_complex_iteration(self):
        complex_expression = x**2 - 1 + 1 * I
        graph_list = full_iterate_expression(complex_expression, np.array([-1 + 1j]), 10000, 10000)
        real_list = np.array([])
        imag_list = np.array([])
        for value in graph_list:
            real_list = np.append(real_list, value.real)
            imag_list = np.append(imag_list, value.imag)
        fig, ax = plt.subplots(num=f'basin plot of F(x) = {complex_expression}, between {0} and {1}')
        ax.scatter(real_list, imag_list, s=1)
        # ax.plot(real_list, imag_list)
        plt.xlim(min(real_list), max(real_list))
        plt.ylim(min(imag_list), max(imag_list))
        LOGGER.info(pp.pformat(graph_list[-10:]))
        LOGGER.info(pp.pformat(graph_list[:10]))
        LOGGER.info(f'real min: {min(real_list)}')
        LOGGER.info(f'real max: {max(real_list)}')
        LOGGER.info(f'imag min: {min(imag_list)}')
        LOGGER.info(f'imag max: {max(imag_list)}')
        plt.show()

    def test_complex_iteration_alternative(self):
        ITERATION_COUNT = 1000
        complex_expression = x**2 - .5 + .3 * I
        graph_list = full_iterate_expression(complex_expression, np.array([0 - 0j]), ITERATION_COUNT, ITERATION_COUNT)
        combined_list = np.array([])
        for value in graph_list:
            combined_list = np.append(combined_list, min(value.real**2 + value.imag**2, np.array([sys.float_info.max])))  # complex number * complex conjugate = (a+bi)(a-bi) = a**2 + b**2
        fig, ax = plt.subplots(num=f'iteration plot of F(x) = {complex_expression}, between {0} and {1}')

        plt.xlim(0, len(combined_list))
        t = np.arange(0, len(combined_list))
        plt.ylim(0, np.max(combined_list))
        fig.set_size_inches(19, 9.5)
        plt.plot(t, combined_list)
        LOGGER.info(pp.pformat(f'{chr(10)}last 10 elements of graph_list: {chr(10)}{pp.pformat(graph_list[-10:])}'))
        LOGGER.info(pp.pformat(f'{chr(10)}first 10 elements of graph_list: {chr(10)}{pp.pformat(graph_list[:10])}'))

        LOGGER.info(pp.pformat(f'{chr(10)}last 10 elements of combined_list: {chr(10)}{pp.pformat(combined_list[-10:])}'))
        LOGGER.info(pp.pformat(f'{chr(10)}first 10 elements of combined_list: {chr(10)}{pp.pformat(combined_list[:10])}'))

        plt.show()

    def test_julia_set_random_iteration(self):
        """ takes complex expression, iterates the expression with a random set of initial complex points
            and returns the final iterated value.
            removes values that have a large magnitude. plots the remaining points on a complex plane.
        """
        ITERATION_COUNT = 1000
        SEED_COUNT = 100000
        RANGE = [-1, 1]
        complex_expression = x**2 - 1 + 1 * I
        lambda_complex_expression = lambdify(x, complex_expression)
        real_random_set = np.array(random.sample(sorted(np.linspace(RANGE[0], RANGE[1], 1000000)), SEED_COUNT))
        imag_random_set = np.array(random.sample(sorted(np.linspace(RANGE[0], RANGE[1], 1000000)), SEED_COUNT))
        complex_random_set = real_random_set + imag_random_set * 1j
        # LOGGER.info(f'complex_random_set first ten: {complex_random_set[:10]}')
        start_time = time.perf_counter()
        iterated_list = iterate_array_expression(lambda_complex_expression, complex_random_set, ITERATION_COUNT)
        LOGGER.info(f'time to complete: {time.perf_counter() - start_time}')

        fig, ax = plt.subplots(num=f'iteration plot of F(x) = {complex_expression}, between {0} and {1}')
        iterated_list = iterated_list[np.isfinite(iterated_list)]  # remove NaN
        LOGGER.info(f'iterated_list first ten: {iterated_list[:10]}')
        LOGGER.info(f'length of iterated_list: {len(iterated_list)}')
        new_iterated_list = np.array([])
        for number in iterated_list:
            if (number.real**2 + number.imag**2) < 4:
                new_iterated_list = np.append(new_iterated_list, number)
            else:
                new_iterated_list = np.append(new_iterated_list, np.NaN)
        new_iterated_list = new_iterated_list[np.isfinite(new_iterated_list)]
        LOGGER.info(f'new_iterated_list first ten: {new_iterated_list[:10]}')
        LOGGER.info(f'length of new_iterated_list: {len(new_iterated_list)}')
        ax.scatter(list(new_iterated_list.real), list(new_iterated_list.imag), s=1)
        plt.show()

    def test_julia_set_divergence_outline(self):
        """ takes complex expression, iterates the expression with a random set of initial complex points
            and returns the times it was iterated before the value diverged.
            removes values that diverge at points other than 10. plots the remaining points on a complex plane.
            this will plot an outline of the julia set
        """
        ITERATION_COUNT = 11
        SEED_COUNT = 5000000
        RANGE = [-1.5, 1.5]
        C = - .5 + .3j

        LOGGER.info(f'-----------------------------------------')
        LOGGER.info(f'starting test_julia_set_divergence_outline:')
        LOGGER.info(f'iteration count: {ITERATION_COUNT}')
        LOGGER.info(f'seed count: {SEED_COUNT}')
        LOGGER.info(f'range: [{RANGE[0]}, {RANGE[1]}]')
        LOGGER.info(f'expression: x**2 + {C}')

        def complex_expression(x):
            return x**2 + C
        complex_function = np.vectorize(complex_expression)
        real_random_set = np.array(random.sample(sorted(np.linspace(RANGE[0], RANGE[1], 10000000)), SEED_COUNT))
        imag_random_set = np.array(random.sample(sorted(np.linspace(RANGE[0], RANGE[1], 10000000)), SEED_COUNT))
        complex_random_set = real_random_set + imag_random_set * 1j
        # LOGGER.info(f'complex_random_set first ten: {complex_random_set[:10]}')

        start_time = time.perf_counter()
        divergence = iterations_till_divergence(complex_function, complex_random_set, ITERATION_COUNT)
        LOGGER.info(f'iteration time to complete: {time.perf_counter() - start_time}')

        # LOGGER.info(f'{divergence}')
        fig, ax = plt.subplots(num=f'iteration plot of F(x) = {complex_expression}, between {0} and {1}')

        def filter_helper(complex_value, divergence):
            if divergence == 10:
                return complex_value
            else:
                return np.NaN

        start_time = time.perf_counter()
        filter_out = np.vectorize(filter_helper, otypes=[np.complex64])
        filtered_list = filter_out(complex_random_set, divergence)
        LOGGER.info(f'filter time to complete: {time.perf_counter() - start_time}')

        start_time = time.perf_counter()
        cleaned_list = []
        cleaned_divergence = []
        isnan_list = np.isnan(filtered_list)
        for index in range(len(filtered_list)):
            if not isnan_list[index]:
                cleaned_divergence.append(divergence[index])
                cleaned_list.append(filtered_list[index])
        cleaned_list = np.array(cleaned_list)
        cleaned_divergence = np.array(cleaned_divergence)
        LOGGER.info(f'clean time to complete: {time.perf_counter() - start_time}')
        # LOGGER.info(f'{cleaned_list}')
        fig.set_size_inches(13, 13)
        plt.tight_layout()
        ax.axis('equal')
        ax.scatter(cleaned_list.real, cleaned_list.imag, s=1)
        plt.show()

    def test_julia_set_divergence_contour(self):
        """ takes complex expression, iterates the expression with a random set of initial complex points
            and returns the times it was iterated before the value diverged.
            removes values that have a large magnitude. plots the remaining points on a complex plane.
        """
        ITERATION_COUNT = 100
        SEED_COUNT = 100000
        RANGE = [-1.5, 1.5]
        C = - 0.4 + 0.6j

        LOGGER.info(f'-----------------------------------------')
        LOGGER.info(f'starting test_julia_set_divergence_contour:')
        LOGGER.info(f'iteration count: {ITERATION_COUNT}')
        LOGGER.info(f'seed count: {SEED_COUNT}')
        LOGGER.info(f'range: [{RANGE[0]}, {RANGE[1]}]')
        LOGGER.info(f'expression: x**2 + {C}')

        def complex_expression(x):
            return x**2 + C
        complex_function = np.vectorize(complex_expression)
        real_random_set = np.array(random.sample(sorted(np.linspace(RANGE[0], RANGE[1], 2000000)), SEED_COUNT))
        imag_random_set = np.array(random.sample(sorted(np.linspace(RANGE[0], RANGE[1], 2000000)), SEED_COUNT))
        complex_random_set = real_random_set + imag_random_set * 1j
        # complex_random_set = np.array([0 + 0j])
        # LOGGER.info(f'complex_random_set first ten: {complex_random_set[:10]}')
        start_time = time.perf_counter()
        divergence = iterations_till_divergence(complex_function, complex_random_set, ITERATION_COUNT)
        LOGGER.info(f'iteration time to complete: {time.perf_counter() - start_time}')
        LOGGER.info(f'{divergence}')
        fig, ax = plt.subplots(num=f'iteration plot of F(x) = {complex_expression}, between {0} and {1}')

        start_time = time.perf_counter()

        def filter_helper(complex_value, divergence):
            if divergence < 100:
                return complex_value
            else:
                return np.NaN
        filter_out = np.vectorize(filter_helper, otypes=[np.complex64])
        filtered_list = filter_out(complex_random_set, divergence)
        LOGGER.info(f'filter time to complete: {time.perf_counter() - start_time}')

        start_time = time.perf_counter()
        cleaned_list = np.array([])
        cleaned_divergence = np.array([])
        for index in range(len(filtered_list)):
            if not np.isnan(filtered_list[index]):
                cleaned_divergence = np.append(cleaned_divergence, divergence[index])
                cleaned_list = np.append(cleaned_list, filtered_list[index])
        LOGGER.info(f'clean time to complete: {time.perf_counter() - start_time}')

        # LOGGER.info(f'{cleaned_list}')
        fig.set_size_inches(19, 9.5)
        # ax.scatter(filtered_list.real, filtered_list.imag, s=1)
        aX, aY, aZ = plot_contour(cleaned_list.real, cleaned_list.imag, cleaned_divergence, resolution=50, contour_method='linear')
        ax.contour(aX, aY, aZ)
        plt.show()

    def test_julia_set_divergence_plot(self):
        """ takes complex expression, iterates the expression with a random set of initial complex points
            and returns the times it was iterated before the value diverged.
            removes values that have a large magnitude. plots the remaining points on a complex plane.
        """
        ITERATION_COUNT = 180
        SEED_COUNT = 750000
        REAL_RANGE = [-2, 2]
        IMAG_RANGE = [-.75, .75]
        C = -2 + 0j

        LOGGER.info(f'-----------------------------------------')
        LOGGER.info(f'starting test_julia_set_divergence_plot:')
        LOGGER.info(f'iteration count: {ITERATION_COUNT}')
        LOGGER.info(f'seed count: {SEED_COUNT}')
        LOGGER.info(f'real range:      [{REAL_RANGE[0]}, {REAL_RANGE[1]}]')
        LOGGER.info(f'imaginary range: [{IMAG_RANGE[0]}, {IMAG_RANGE[1]}]')
        LOGGER.info(f'expression: x**2 + {C}')

        def complex_expression(x):
            return x**2 + C
        complex_function = np.vectorize(complex_expression)
        real_random_set = np.array(random.sample(sorted(np.linspace(REAL_RANGE[0], REAL_RANGE[1], 2000000)), SEED_COUNT))
        imag_random_set = np.array(random.sample(sorted(np.linspace(IMAG_RANGE[0], IMAG_RANGE[1], 2000000)), SEED_COUNT))
        complex_random_set = real_random_set + imag_random_set * 1j
        # complex_random_set = np.array([0 + 0j])
        # LOGGER.info(f'complex_random_set first ten: {complex_random_set[:10]}')

        start_time = time.perf_counter()
        divergence = iterations_till_divergence(complex_function, complex_random_set, ITERATION_COUNT)
        LOGGER.info(f'iteration time to complete: {time.perf_counter() - start_time}')
        # LOGGER.info(f'{divergence}')
        fig, ax = plt.subplots(num=f'iteration plot of F(x) = {complex_expression}, between {0} and {1}')

        start_time = time.perf_counter()

        def filter_helper(complex_value, divergence):
            if divergence < ITERATION_COUNT:
                return complex_value
            else:
                return np.NaN
        filter_out = np.vectorize(filter_helper, otypes=[np.complex64])
        filtered_list = filter_out(complex_random_set, divergence)
        LOGGER.info(f'filter time to complete: {time.perf_counter() - start_time}')

        start_time = time.perf_counter()
        cleaned_list = []
        cleaned_divergence = []
        isnan_list = np.isnan(filtered_list)
        for index in range(len(filtered_list)):
            if not isnan_list[index]:
                cleaned_divergence.append(divergence[index])
                cleaned_list.append(filtered_list[index])
        cleaned_list = np.array(cleaned_list)
        cleaned_divergence = np.array(cleaned_divergence)
        LOGGER.info(f'clean time to complete: {time.perf_counter() - start_time}')
        # LOGGER.info(f'{cleaned_list}')
        fig.set_size_inches(21.33, 8)
        plt.tight_layout()
        ax.axis('equal')
        ax.scatter(cleaned_list.real, cleaned_list.imag, c=cleaned_divergence, s=1, cmap='YlGnBu_r')
        plt.show()
        # plt.savefig('juliaset12.png')

    def test_julia_set_divergence_plot2(self):
        """ takes complex expression, iterates the expression with a random set of initial complex points
            and returns the times it was iterated before the value diverged.
            removes values that have a large magnitude. plots the remaining points on a complex plane.
        """
        ITERATION_COUNT = 40
        SEED_COUNT = 1000000
        REAL_RANGE = [-5, 5]
        IMAG_RANGE = [-5, 5]

        LOGGER.info(f'-----------------------------------------')
        LOGGER.info(f'starting test_julia_set_divergence_plot:')
        LOGGER.info(f'iteration count: {ITERATION_COUNT}')
        LOGGER.info(f'seed count: {SEED_COUNT}')
        LOGGER.info(f'real range:      [{REAL_RANGE[0]}, {REAL_RANGE[1]}]')
        LOGGER.info(f'imaginary range: [{IMAG_RANGE[0]}, {IMAG_RANGE[1]}]')

        def complex_expression(x):
            return 2 + (x * np.e**(1j * np.abs(x)**2)) / 10
        complex_function = np.vectorize(complex_expression)
        real_random_set = np.array(random.sample(sorted(np.linspace(REAL_RANGE[0], REAL_RANGE[1], 2000000)), SEED_COUNT))
        imag_random_set = np.array(random.sample(sorted(np.linspace(IMAG_RANGE[0], IMAG_RANGE[1], 2000000)), SEED_COUNT))
        complex_random_set = real_random_set + imag_random_set * 1j
        # complex_random_set = np.array([0 + 0j])
        # LOGGER.info(f'complex_random_set first ten: {complex_random_set[:10]}')

        start_time = time.perf_counter()
        divergence = iterations_till_divergence(complex_function, complex_random_set, ITERATION_COUNT)
        LOGGER.info(f'iteration time to complete: {time.perf_counter() - start_time}')
        # LOGGER.info(f'{divergence}')
        fig, ax = plt.subplots(num=f'iteration plot of F(x) = {complex_expression}, between {0} and {1}')

        start_time = time.perf_counter()

        def filter_helper(complex_value, divergence):
            if divergence < ITERATION_COUNT:
                return complex_value
            else:
                return np.NaN
        filter_out = np.vectorize(filter_helper, otypes=[np.complex64])
        filtered_list = filter_out(complex_random_set, divergence)
        LOGGER.info(f'filter time to complete: {time.perf_counter() - start_time}')

        start_time = time.perf_counter()
        cleaned_list = []
        cleaned_divergence = []
        isnan_list = np.isnan(filtered_list)
        for index in range(len(filtered_list)):
            if not isnan_list[index]:
                cleaned_divergence.append(divergence[index])
                cleaned_list.append(filtered_list[index])
        cleaned_list = np.array(cleaned_list)
        cleaned_divergence = np.array(cleaned_divergence)
        LOGGER.info(f'clean time to complete: {time.perf_counter() - start_time}')
        # LOGGER.info(f'{cleaned_list}')
        fig.set_size_inches(11, 11)
        plt.tight_layout()
        ax.axis('equal')
        ax.scatter(cleaned_list.real, cleaned_list.imag, c=cleaned_divergence, s=1, cmap='YlGnBu_r')
        plt.show()
        # plt.savefig('juliaset14.png')
    
    def test_julia_set_plot(self):
        def complex_expression(x):
            return x**2 + 0.7885*np.e**(1j * 2.9)
        
        plot_julia_set(complex_expression, 10, 4000000, -1, 1, -1, 1, 'CMRmap')
        # plt.show()
        plt.savefig('juliaset27.png')
        
    def test_julia_set_root_plot(self):
        complex_expression = x**4 - .4 + 6j
        derivative_expression = diff(complex_expression, x)
        complex_expression = lambdify(x, complex_expression)
        derivative_expression = lambdify(x, derivative_expression)
        
        plot_julia_root_set(complex_expression, derivative_expression, 100, 5000000, -1.5, 1.5, -1.5, 1.5, 'Set1')
        
        # plt.show()
        plt.savefig('juliaset24.png')


    def test_complex_fixed_point(self):
        expression3 = z**2 - 1 + 1j
        expression4 = z
        solution = solve(Eq(expression3, expression4))
        LOGGER.info(solution)

    def test_complex_period_two(self):
        expression3 = (z**2 - 1 + 1j)**2 - 1 + 1j
        expression4 = z
        solution = solve(Eq(expression3, expression4))
        LOGGER.info(solution)

    def test_complex_fixed_point2(self):
        expression3 = 2 + (z * np.e**(1j * z**2)) / 10
        expression4 = z
        solution = solve(Eq(expression3, expression4))
        LOGGER.info(solution)


def basin_helper(c_value, expression, initial_value):
    expression = expression.subs(y, c_value)
    return find_basin_of_attraction(expression, initial_value, 8)


def plot_contour(x, y, z, resolution=50, contour_method='linear'):
    resolution = str(resolution) + 'j'
    aX, aY = np.mgrid[min(x):max(x):complex(resolution), min(y):max(y):complex(resolution)]
    points = [[a, b] for a, b in zip(x, y)]
    aZ = griddata(points, z, (aX, aY), method=contour_method)
    return aX, aY, aZ


def plot_julia_set(expression: callable, iteration_count: int=100, seed_count: int=20000, real_range_min: float=-1.0, real_range_max: float=1.0, imag_range_min: float=-1.0, imag_range_max: float=1.0, cmap='viridis'):
    """ takes complex expression, iterates the expression with a random set of initial complex points
        and returns the times it was iterated before the value diverged.
        removes values that have a large magnitude. plots the remaining points on a complex plane.


    Args:
        expression (callable): _description_
        iteration_count (int): _description_
        seed_count (int): _description_
        real_range_min (float): _description_
        real_range_max (float): _description_
        imag_range_min (float): _description_
        imag_range_max (float): _description_

    Returns:
        _type_: _description_
    """

    complex_function = np.vectorize(expression)
    real_random_set = np.array(random.sample(sorted(np.linspace(real_range_min, real_range_max, 20000000)), seed_count))
    imag_random_set = np.array(random.sample(sorted(np.linspace(imag_range_min, imag_range_max, 20000000)), seed_count))
    complex_random_set = real_random_set + imag_random_set * 1j

    divergence = iterations_till_divergence(complex_function, complex_random_set, iteration_count)
    
    def filter_helper(complex_value, divergence):
        if divergence < iteration_count:
            return complex_value
        else:
            return np.NaN
    filter_out = np.vectorize(filter_helper, otypes=[np.complex64])
    divergence2 = divergence.get()
    filtered_list = filter_out(complex_random_set, divergence2)

    cleaned_list = []
    cleaned_divergence = []
    isnan_list = np.isnan(filtered_list)
    for index in range(len(filtered_list)):
        if not isnan_list[index]:
            cleaned_divergence.append(divergence2[index])
            cleaned_list.append(filtered_list[index])
    cleaned_list = np.array(cleaned_list)
    cleaned_divergence = np.array(cleaned_divergence)
    
    fig, ax = plt.subplots(num=f'Julia set plot')
    fig.set_size_inches(20, 20)
    plt.tight_layout()
    ax.axis('equal')
    ax.scatter(cleaned_list.real, cleaned_list.imag, c=cleaned_divergence, s=.1, cmap=cmap,  marker='s')
    
def plot_julia_root_set(expression: callable, derivative: callable, iteration_count: int=100, seed_count: int=20000, real_range_min: float=-1.0, real_range_max: float=1.0, imag_range_min: float=-1.0, imag_range_max: float=1.0, cmap='viridis'):
    """ takes complex expression, iterates the expression with the newtons method 
        over a random set of initial complex points
        and returns the root at each point.
        removes values that have a large magnitude. plots the remaining points on a complex plane.

    Args:
        expression (callable): _description_
        iteration_count (int): _description_
        seed_count (int): _description_
        real_range_min (float): _description_
        real_range_max (float): _description_
        imag_range_min (float): _description_
        imag_range_max (float): _description_

    Returns:
        _type_: _description_
    """
    
    real_random_set = np.array(random.sample(sorted(np.linspace(real_range_min, real_range_max, 20000000)), seed_count))
    imag_random_set = np.array(random.sample(sorted(np.linspace(imag_range_min, imag_range_max, 20000000)), seed_count))
    complex_random_set = real_random_set + imag_random_set * 1j

    root_list = find_roots(expression, derivative, complex_random_set, iteration_count)
    LOGGER.info(root_list[:20])
    
    # cleaned_roots = []
    # isnan_list = np.isnan(root_list)
    # for index in range(len(root_list)):
    #     if not isnan_list[index]:
    #         cleaned_roots.append(root_list[index])
    # cleaned_roots = np.array(cleaned_roots)
    
    fig, ax = plt.subplots(num=f'Julia set root plot')
    fig.set_size_inches(20, 20)
    plt.tight_layout()
    ax.axis('equal')
    ax.scatter(complex_random_set.real, complex_random_set.imag, c=root_list.imag, s=.1, cmap=cmap)


def find_roots(iterating_function: callable, derivative: callable,  initial_values: np.ndarray, iteration_count=1000) -> list:
    """ finds roots with newtons method

    Args:
        expression (numpy.NDArray): function of x to be iterated
        initial_values (numpy.NDArray): array of initial values
        max_iterations (int): amount of iterations

    Returns:
        list: list of when each value diverges
    """
    root_list = initial_values
    
    for i in tqdm(range(iteration_count)):
        root_list = root_list - (iterating_function(root_list) / derivative(root_list))
    return root_list


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format=LOG_FORMAT, filename='log_file_test.log')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('numba.core').setLevel(logging.WARNING)
    test_client = DynamicalTest()
    test_client.test_julia_set_plot()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
