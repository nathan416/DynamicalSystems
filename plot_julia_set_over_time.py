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
import threading
import time
import unittest
from pympler import asizeof

import cupy as cp
import matplotlib
import matplotlib.pyplot as plt
import mpl_scatter_density
import mpld3
import numpy as np
from numba import cuda
from PIL import Image
from scipy.interpolate import griddata
from sympy import *
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from dynamical import *

SEED = 100
LOGGER = logging.getLogger(__name__)
LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -43s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')


x, y, z, t = symbols('x y z t')
k, m, n = symbols('k m n', integer=True)
f, g, h = symbols('f g h', cls=Function)

REAL_RANGE_MIN = -2.133
REAL_RANGE_MAX = 2.133
IMAG_RANGE_MIN = -1.2
IMAG_RANGE_MAX = 1.2
SEED_COUNT = 4000000

REAL_RANDOM_SET = np.array(random.sample(sorted(np.linspace(REAL_RANGE_MIN, REAL_RANGE_MAX, 20000000)), SEED_COUNT))
IMAG_RANDOM_SET = np.array(random.sample(sorted(np.linspace(IMAG_RANGE_MIN, IMAG_RANGE_MAX, 20000000)), SEED_COUNT))
COMPLEX_RANDOM_SET = np.array((REAL_RANDOM_SET + IMAG_RANDOM_SET * 1j), dtype=np.complex64)

@cuda.jit('void(complex64[:], float32)')
def complex_expression(x, a):
    # i = cuda.grid(1)
    # x[i] = x[i]**2 - 0.4 + 0.6j
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, x.shape[0], stride):
        x[i] = 1 - x[i]**2 + x[i]**2 / (2 + 4 * x[i]) + 0.7885 * np.e**(a * 1j)

def test_plot_julia_set_over_time2():
    start_time = time.perf_counter()
    fig_list = []

    for a in tqdm(np.linspace(4.073019621237019, 4.104593416750483, 200)):
        cleaned_list, cleaned_divergence = plot_julia_set(complex_expression, a, 150, SEED_COUNT, REAL_RANGE_MIN, REAL_RANGE_MAX, IMAG_RANGE_MIN, IMAG_RANGE_MAX, 'CMRmap', 'gpu', fig_name=f'{a}', is_plotted=False, complex_random_set=COMPLEX_RANDOM_SET)
        fig_list.append((cleaned_list, cleaned_divergence, a))
    # pool = multiprocessing.Pool(psutil.cpu_count(logical=False))
    # pool.map(overtime_helper, fig_list, chunksize=2)
    process_map(overtime_helper, fig_list, chunksize=2, max_workers=psutil.cpu_count(logical=False))
    # plt.show()
    # plt.savefig(f'pictures/juliaset{len(os.listdir(os.path.join(os.getcwd(), "pictures")))}.png')
    LOGGER.info(f'size of fig_list: {asizeof.asizeof(fig_list)}')
    
    LOGGER.info(f'time taken test_plot_julia_set_over_time2: {time.perf_counter() - start_time}')
    
def overtime_helper(zip):
    cleaned_list, cleaned_divergence, a = zip
    figure, ax = plt.subplots(num=a)
    figure.set_size_inches(19.2, 10.8)
    ax.axis('equal')
    plt.tight_layout()
    ax.scatter(cleaned_list.real, cleaned_list.imag, c=cleaned_divergence, s=.1, cmap='CMRmap')
    figure.savefig(f'pictures/new/{int(a * 100000)}.png')
    figure.clf()
    
def main():
    logging.basicConfig(level=logging.DEBUG,
                        format=LOG_FORMAT, filename='log_file_test.log')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('numba.core').setLevel(logging.WARNING)
    logging.getLogger('numba.cuda.cudadrv.driver').setLevel(logging.WARNING)
    test_plot_julia_set_over_time2()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()