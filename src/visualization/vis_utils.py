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

def create_figure(nrows=1, ncols=1, figwidth=8, figsize=None, xlabel='', ylabel='', title='', axis_off=False, dpi=100):
    
    if figsize is None:
        figsize = (figwidth, nrows/ncols*figwidth)
        
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    
    if axis_off:
        for ax in np.reshape(axes, (-1,)):
            ax.set_axis_off()

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
    
