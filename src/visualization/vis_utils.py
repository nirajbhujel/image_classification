"""
Created on Thu Jul 13 09:19:39 2023

@author: niraj
"""

import os
import cv2
import sys

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

from collections import defaultdict

def create_figure(nrows=1, ncols=1, figsize=None, subplot_size=None, figwidth=8, xlabel='', ylabel='',
    title='', axis_off=True, dpi=100, layout='constrained'):
    '''
    Parameters
        nrows (int, optional): Number of rows of subplots. Default is 1.
        ncols (int, optional): Number of columns of subplots. Default is 1.
        figwidth (int, optional): Total width of the figure in inches. Default is 8.
        figsize (tuple, optional): Custom size of the figure (width, height). Overrides figwidth and subplot_size if provided.
        subplot_size (tuple, optional): Size of each subplot (width, height) in inches. Default is (4, 4).
        xlabel (str, optional): Label for the x-axis of each subplot. Default is an empty string.
        ylabel (str, optional): Label for the y-axis of each subplot. Default is an empty string.
        title (str, optional): Title of the figure. Default is an empty string.
        axis_off (bool, optional): If True, turns off the axis for each subplot. Default is False.
        dpi (int, optional): Dots per inch for the figure resolution. Default is 100.
    Returns
    fig (Figure): The created Matplotlib figure.
    axs (Axes or array of Axes): The created subplots' axes.
    '''
    if figsize is None:
        # Calculate the total figure size
        if subplot_size is not None:
            figsize = (ncols * subplot_size[0], nrows * subplot_size[1])  # subplot size in inches
        else:
            figsize = (figwidth, nrows/ncols*figwidth)
        
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)

    # make it two dimensional
    if ncols==1 or nrows==1:
        axes = axes.reshape(nrows, ncols)

    if axis_off:
        for ax in axes.flatten():
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.yaxis.set_major_formatter(plt.NullFormatter())

            ax.set_xticks([])
            ax.set_yticks([])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.title(title)
    plt.tight_layout()

    return fig, axes

def get_color_mapper(min_val, max_val, cmap='turbo'):
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)  # cm.*_r is reversed cmap
    return mapper

def get_array_color(array, min_val=None, max_val=None, cmap="turbo"):
    '''
    Generate color for each array values.
    '''
    if max_val is None:
        max_val = np.max(array)
        
    if min_val is None:
        min_val = np.min(array)
        
    mapper = get_color_mapper(min_val, max_val)
    mapper.set_array(errors)
    colors = [mapper.to_rgba(a) for a in array] 
    
    return colors , mapper


def add_color_bar(ax, data, cmap="turbo", fig=None):
    
 
    '''
    Add color bar to axes based on data. 
    '''
    if fig is None:
        fig = plt.gcf()
    
    if data is not None:
        min_val, max_val = np.min(data), np.max(data)
        
    else:
        raise Exception('Min and Max value is required to add color bar')
    
    color_mapper = get_color_mapper(min_val, max_val, cmap)
    y_ticks = [min_val, (max_val - (max_val - min_val) / 2), max_val]
    y_ticks_labels = [f"{int(t)}" for t in y_ticks]
   
    cbar = fig.colorbar(color_mapper, ticks=y_ticks, ax=ax)
    cbar.ax.set_yticklabels(y_ticks_labels)
    
    return ax


def add_grid_to_axes(ax, grid_size=16, labelsize=5, minor_ticks=False):
    
    if isinstance(grid_size, int):
        x_grid, y_grid = grid_size, grid_size
    elif isinstance(grid_size, tuple):
        x_grid, y_grid = grid_size
    else:
        raise TypeError(f"Grid size {grid_size} should be tuple or int")
        
    x_max = max(ax.get_xlim())
    x_major_ticks = np.arange(0, x_max, x_grid)
    ax.set_xticks(x_major_ticks, labels=np.arange(len(x_major_ticks)))
    
    y_max = max(ax.get_ylim())  # (491.5, -0.5)
    y_major_ticks = np.arange(0, y_max, y_grid) # need to swap for arange to work, otherwise empty
    ax.set_yticks(y_major_ticks, labels=np.arange(len(y_major_ticks)))
    
    if minor_ticks:
        x_minor_ticks = np.arange(0, x_max, x_grid//4)
        y_minor_ticks = np.arange(0, y_max, y_grid//4) 
        ax.set_xticks(x_minor_ticks, minor=True)
        ax.set_yticks(y_minor_ticks, minor=True)   
    
    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.4)
    
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    
    return ax

def create_video_from_images(image_dir, video_file_path, fps=1, display_text=False, img_format=''):
    images = [img for img in sorted(os.listdir(image_dir)) if img.endswith(img_format)]
    frame = cv2.imread(os.path.join(image_dir, images[0]))
    
    if len(frame.shape)==3:
        height, width, layers = frame.shape
    elif len(frame.shape)==2:
        height, width = frame.shape
    else:
        raise Exception("Image shape not known!")

    video = cv2.VideoWriter(video_file_path, 0, fps=fps, frameSize=(width, height))

    for image in images:
        img = cv2.imread(os.path.join(image_dir, image))
        if display_text:
            cv2.putText(img, os.path.basename(image).split('.')[0], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        video.write(img)
        
    cv2.destroyAllWindows()
    video.release()
    
def img_gradient(x):

    left = x
    right = np.pad(x, ((0, 0), (0, 1)))[..., :, 1:]
    top = x
    bottom = np.pad(x, ((0, 1), (0, 0)))[..., 1:, :]

    dx, dy = right - left, bottom - top 
    # dx will always have zeros in the last column, right-left
    # dy will always have zeros in the last row,    bottom-top
    dx[..., :, -1] = 0
    dy[..., -1, :] = 0

    return np.abs(dx) + np.abs(dy)
    
