"""Julia set image
    Class: CPSC 455
    By: Nathan Flack
    Version: 1.0
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

import math
import cupy as cp
import matplotlib
import matplotlib.pyplot as plt
import mpl_scatter_density
import mpld3
import numpy as np
from numba import cuda
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import griddata
from sympy import *
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from matplotlib import cm

from dynamical import *

REAL_RANGE_MIN = -1.0
REAL_RANGE_MAX = 1.0
IMAG_RANGE_MIN = -1.3
IMAG_RANGE_MAX = 1.3

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = (IMAG_RANGE_MAX - IMAG_RANGE_MIN) * (IMAGE_WIDTH) / (REAL_RANGE_MAX - REAL_RANGE_MIN)

FRAMES = 2000
DURATION = 40
ITERATIONS = 110

FILENAME = f'pictures/over_time3/overtime12.gif'

LOGGER = logging.getLogger(__name__)
LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -43s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')

norm = matplotlib.colors.Normalize(vmin=0, vmax=110)

# examples
# x = 1 - x**2 + x**2 / (2 + 4 * x) + 0.7885 * np.e**(a * 1j)
# x = 1 - x + x**2 + 0.7885 * np.e**(a * 1j)
# x = x**4 + x**3/(x-1) + x**2/(x**3 + 4 *x**2 + 5) - 0.5885 * np.e**(a * 1j)
# x = x**4 + x**3/(x-1) + x**2/(x**3 + 4 *x**2 + 5) + 0.755534*math.cos(a) + 0.737292*1j*math.cos(a) - 2*0.737292*1j
# x = 2**x + 0.2885 * np.e**(a * 1j)
# x = x**2 + 0.355534*math.cos(2*a)-0.337292*1j*math.cos(a)
# x**4 + x**3 / (x - 1) + x**2 / (x**3 + 4 * x**2 + 5) + 0.377767 * math.sin(a) + 0.368646 * 1j * math.sin(a) - 0.368646 * 1j + 0.377767
# x**2 + a*.01 - a*.3*1j


@cuda.jit('void(complex64, float32)', device=True)
def iterating_function(x, a):
    return x**4 + x**3 / (x - 1) + x**2 / (x**3 + 4 * x**2 + 5) + 0.377767 * math.sin(a) + 0.368646 * 1j * math.sin(a) - 0.368646 * 1j + 0.377767


@cuda.jit('void(complex64, float32)', device=True)
def divergence_tracker(x, divergence):
    if x.real**2 + x.imag**2 < 300:
        return divergence + 1
    return divergence


@cuda.jit('void(complex64[:,:], int32[:,:], float32, int32)')
def helper_func5_image(x, divergence, a, iter):
    xstart, ystart = cuda.grid(2)
    xstride, ystride = cuda.gridsize(2)
    for k in range(iter):
        for i in range(xstart, x.shape[0], xstride):
            for j in range(ystart, x.shape[1], ystride):
                x[i, j] = iterating_function(x[i, j], a)
                divergence[i, j] = divergence_tracker(x[i, j], divergence[i, j])


def plot_julia_set_image(expression: callable, a, iteration_count: int = 100, real_range_min: float = -1.0, real_range_max: float = 1.0,
                         imag_range_min: float = -1.0, imag_range_max: float = 1.0, cmap='viridis', image_width=1920, image_height=1080, fig_name='Julia set plot'):
    """ takes complex expression, iterates the expression with a random set of initial complex points
        and returns the times it was iterated before the value diverged.
        removes values that did not diverge. plots the remaining points on a complex plane
        with a color matching the times needed for the modulus to go above 4.
        This should be R^2 instead of 4 where R = escape radius; but its hard to code

    Args:
        expression (callable): julia set function
        iteration_count (int): amount of times to iterate
        seed_count (int): amount of points to iterate and plot
        real_range_min (float): min range of real part of complex points
        real_range_max (float): max range of real part of complex points
        imag_range_min (float): min range of imag part of complex points
        imag_range_max (float): max range of imag part of complex points

    """
    start_time = time.perf_counter()

    real_set = np.linspace(real_range_min, real_range_max, int(image_width)).reshape((1, int(image_width)))
    imag_set = np.linspace(imag_range_max, imag_range_min, int(image_height)).reshape((int(image_height), 1))
    complex_set = np.array(real_set + 1j * imag_set, dtype=np.complex64)

    divergence = iterations_till_divergence_image(expression, np.array(complex_set, dtype=np.complex64), a, iteration_count)

    LOGGER.info(f'time taken plot_julia_set: {time.perf_counter() - start_time}')
    return divergence


def iterations_till_divergence_image(expression: callable, initial_values: np.ndarray, a: float, iteration_count=1000) -> list:
    """ iterates over the initial values with a function and returns
        a list of when the modulus of each complex value goes above 4
        array can be multidimensional but the function must change to account
        for it

    Args:
        expression (ufunc): function of x to be iterated
        initial_values (numpy.NDArray): array of initial values
        max_iterations (int): amount of iterations

    Returns:
        list: list of when each value diverges
    """
    start_time = time.perf_counter()
    iterating_values_h = initial_values
    divergence_h = np.zeros(initial_values.shape, dtype=np.int32)

    blockdim = (16, 16)
    griddimx = math.ceil(IMAGE_WIDTH / blockdim[0])
    griddimy = math.ceil(IMAGE_HEIGHT / blockdim[1])
    griddim = (griddimx, griddimy)

    iterating_values_d = cuda.to_device(iterating_values_h)
    divergence_d = cuda.to_device(divergence_h)
    
    expression[griddim, blockdim](iterating_values_d, divergence_d, a, iteration_count)
    
    divergence_h = divergence_d.copy_to_host()
    LOGGER.info(f'time taken iterations_till_divergence_image: {time.perf_counter() - start_time}')
    return divergence_h


def plotting_function():
    start_time = time.perf_counter()

    # for a in tqdm(np.linspace(0, 2*np.pi, 10)):
    #     cleaned_list, cleaned_divergence = plot_julia_set_image(helper_func5_image, a, 110, REAL_RANGE_MIN, REAL_RANGE_MAX, IMAG_RANGE_MIN, IMAG_RANGE_MAX, 'RdYlGn', 19.2, 10.8)
    #     fig_list.append(cleaned_divergence)
    # imgs = process_map(plotting_helper, np.linspace(0, 2*np.pi, FRAMES), chunksize=1, max_workers=psutil.cpu_count(False))
    imgs = []
    for a in tqdm(np.linspace(0, 2*np.pi, FRAMES)):
        imgs.append(plotting_helper(a))
        
    LOGGER.info(f'size of imgs {asizeof.asizeof(imgs)}')
    LOGGER.info(f'time taken plotting_function: {time.perf_counter() - start_time}')
    return imgs


def plotting_helper(a):
    # norm = (cleaned_divergence - np.min(cleaned_divergence)) / (np.max(cleaned_divergence) - np.min(cleaned_divergence))
    divergence = plot_julia_set_image(helper_func5_image, a, ITERATIONS, REAL_RANGE_MIN, REAL_RANGE_MAX, IMAG_RANGE_MIN, IMAG_RANGE_MAX, 'RdYlGn', IMAGE_WIDTH, IMAGE_HEIGHT)
    normalized = norm(divergence)

    im = Image.fromarray(np.uint8(cm.gnuplot(normalized) * 255))
    I1 = ImageDraw.Draw(im)
    my_font = ImageFont.truetype('arial', 20)
    I1.text((28, 36), f'x = x**4 + x**3/(x-1) + x**2/(x**3 + 4 *x**2 + 5) + {0.377767 * math.sin(a) + 0.368646 * 1j * math.sin(a) - 0.368646 * 1j + 0.377767}', fill=(255, 0, 0), font=my_font)
    # im.save(f'pictures/over_time3/{int(a*100000)}.png')
    # plt.imsave(f'pictures/over_time3/{int(a*100000)}.png', cleaned_divergence, cmap='twilight_shifted')
    return im

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format=LOG_FORMAT, filename='log_file_test.log')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('numba.core').setLevel(logging.WARNING)
    logging.getLogger('numba.cuda.cudadrv.driver').setLevel(logging.WARNING)
    imgs = plotting_function()
    img = imgs[0]  # extract first image from iterator
    img.save(fp=FILENAME, format='GIF', append_images=imgs[1:],
             save_all=True, duration=DURATION, loop=0)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()



    
    