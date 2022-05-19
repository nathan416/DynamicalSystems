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

REAL_RANGE_MIN = -2.31
REAL_RANGE_MAX = 2.31
IMAG_RANGE_MIN = -1.3
IMAG_RANGE_MAX = 1.3

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = (IMAG_RANGE_MAX - IMAG_RANGE_MIN)*(IMAGE_WIDTH)/(REAL_RANGE_MAX - REAL_RANGE_MIN)

FRAMES = 30
DURATION = 150
ITERATIONS = 400

LOGGER = logging.getLogger(__name__)
LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -43s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')

norm = matplotlib.colors.Normalize(vmin=0, vmax=399)
# @cuda.jit('void(complex64[:,:], float32)')
# def complex_function(x, a):
#     # i = cuda.grid(1)
#     # x[i] = x[i]**2 - 0.4 + 0.6j
#     # start = cuda.grid(1)
#     # stride = cuda.gridsize(1)
#     xstart, ystart = cuda.grid(2)
#     xstride, ystride = cuda.gridsize(2)
#     for i in range(xstart, x.shape[0], xstride):
#         for j in range(ystart, x.shape[1], ystride):
#             # x[i, j] = 1 - x[i, j]**2 + x[i, j]**2 / (2 + 4 * x[i, j]) + 0.7885 * np.e**(a * 1j)
#             # x[i, j] = 1 - x[i, j] + x[i, j]**2 + 0.7885 * np.e**(a * 1j)
#             x[i, j] = x[i, j]**4 + x[i, j]**3/(x[i, j]-1) + x[i, j]**2/(x[i, j]**3 + 4 *x[i, j]**2 + 5) - 0.5885 * np.e**(a * 1j)

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
    return complex_set, divergence

def iterations_till_divergence_image(expression: callable, initial_values: np.ndarray, a:float, iteration_count=1000) -> list:
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

    griddim = 64
    blockdim = 32
    
    iterating_values_d = cuda.to_device(iterating_values_h)
    divergence_d = cuda.to_device(divergence_h)
    # for iteration in range(iteration_count):
        # expression[griddim, blockdim](iterating_values_d, a)
    expression[griddim, blockdim](iterating_values_d, divergence_d, a, iteration_count)
    divergence_h = divergence_d.copy_to_host()
    LOGGER.info(f'time taken iterations_till_divergence_image: {time.perf_counter() - start_time}')
    return divergence_h

@cuda.jit('void(complex64[:,:], int32[:,:], float32, int32)')
def helper_func5_image(x, divergence, a, iter):
    xstart, ystart= cuda.grid(2)
    xstride, ystride = cuda.gridsize(2)
    for k in range(iter):
        for i in range(xstart, x.shape[0], xstride): 
            for j in range(ystart, x.shape[1], ystride):
                # x[i, j] = x[i, j]**4 + x[i, j]**3/(x[i, j]-1) + x[i, j]**2/(x[i, j]**3 + 4 *x[i, j]**2 + 5) + 0.355534*math.cos(2*a)-0.337292*1j*math.cos(a)
                # x[i, j] = 2**x[i, j] + 0.2885 * np.e**(a * 1j)
                x[i, j] = x[i, j]**2 + 0.355534*math.cos(2*a)-0.337292*1j*math.cos(a)
                if x[i,j].real**2 + x[i,j].imag**2 < 4:
                    divergence[i,j] = divergence[i,j] + 1

def plotting_function(frames):
    start_time = time.perf_counter()

    # for a in tqdm(np.linspace(0, 2*np.pi, 10)):
    #     cleaned_list, cleaned_divergence = plot_julia_set_image(helper_func5_image, a, 110, REAL_RANGE_MIN, REAL_RANGE_MAX, IMAG_RANGE_MIN, IMAG_RANGE_MAX, 'RdYlGn', 19.2, 10.8)
    #     fig_list.append(cleaned_divergence)
    imgs = process_map(plotting_helper, np.linspace(-.1, .1, frames), chunksize=6, max_workers=psutil.cpu_count(False))
    LOGGER.info(f'size of imgs {asizeof.asizeof(imgs)}')
    LOGGER.info(f'time taken plotting_function: {time.perf_counter() - start_time}')
    return imgs
    
def plotting_helper(a):
    # norm = (cleaned_divergence - np.min(cleaned_divergence)) / (np.max(cleaned_divergence) - np.min(cleaned_divergence))
    cleaned_list, cleaned_divergence = plot_julia_set_image(helper_func5_image, a, ITERATIONS, REAL_RANGE_MIN, REAL_RANGE_MAX, IMAG_RANGE_MIN, IMAG_RANGE_MAX, 'RdYlGn', IMAGE_WIDTH, IMAGE_HEIGHT)
    
    normalized = norm(cleaned_divergence)

    im = Image.fromarray(np.uint8(cm.gnuplot2(normalized)*255))
    I1 = ImageDraw.Draw(im)
    my_font = ImageFont.truetype('arial', 15)
    I1.text((28, 36), f'x**2 + {0.355534*math.cos(2*a)-0.337292*1j*math.cos(a)}', fill=(255, 0, 0), font=my_font)
    # im.save(f'pictures/over_time3/{int(a*100000)}.png')
    # plt.imsave(f'pictures/over_time3/{int(a*100000)}.png', cleaned_divergence, cmap='twilight_shifted')
    return im

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format=LOG_FORMAT, filename='log_file_test.log')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('numba.core').setLevel(logging.WARNING)
    logging.getLogger('numba.cuda.cudadrv.driver').setLevel(logging.WARNING)
    imgs = plotting_function(FRAMES)
    img = imgs[0]  # extract first image from iterator
    img.save(fp=f'pictures/over_time3/overtime9.gif', format='GIF', append_images=imgs[1:],
         save_all=True, duration=DURATION, loop=0)
    
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()