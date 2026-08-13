#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 18:20:49 2026

"""

import numpy as np 
import cv2 
from cv2 import ximgproc 
 
from pathlib import Path 
import json 
import yaml
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from PIL import Image
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
 
def plot_image(img, title = "", grey= False): 
 
    fig, ax = plt.subplots() 
    plt.title(title) 
    vmin = 0 
    vmax = 1.5 
    if(grey): 
        im = ax.imshow(img, cmap='grey',vmin=vmin, vmax=vmax) 
    else: 
        if len(img.shape) == 3 and img.shape[2] == 3: 
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            im = ax.imshow(img_rgb,vmin=vmin, vmax=vmax) 
        else: 
            im = ax.imshow(img, cmap='turbo',vmin=vmin, vmax=vmax) 
 
    divider = make_axes_locatable(ax) 
    cax = divider.append_axes('right', size='5%', pad=0.15) 
    fig.colorbar(im, cax=cax, orientation='vertical') 
    plt.show() 
 
    return 

# 1. Define the constructor for OpenCV matrices
def opencv_matrix_constructor(loader, node):
    # Extract the mapping (rows, cols, data, etc.)
    mapping = loader.construct_mapping(node, deep=True)
    # Convert the flat data list to a NumPy array
    mat = np.array(mapping["data"])
    # Resize to the correct dimensions
    if mapping["cols"] > 1:
        mat.resize(mapping["rows"], mapping["cols"])
    else:
        mat.resize(mapping["rows"],)
    return mat



def load_calib(calib_file, camid):
        
    
    # 2. Register the constructor for the specific tag
    yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix_constructor)
    
    # 3. Load the file normally
    with open(calib_file, 'r') as file:
        data = yaml.load(file, Loader=yaml.Loader) # Use Loader=yaml.Loader or yaml.FullLoader
    

    
    if camid==0:
        
        res = data['cameras']['camera_0']['resolution']
        K = data['cameras']['camera_0']['K']
        R = data['cameras']['camera_0']['R']
        T = data['cameras']['camera_0']['T']
    else:
        res = data['cameras']['camera_1']['resolution']
        K = data['cameras']['camera_1']['K']
        R = data['cameras']['camera_1']['R']
        T = data['cameras']['camera_1']['T']
    
    return res,K, R, T





# imgl = '/Volumes/WH/data/REAL3EXT 2/s8/l2/zed_left.png'
# imgl = '/Volumes/WH/data/REAL3EXT 2/s8/l2/zed_right.png'

# calib_file = '/Volumes/WH/data/REAL3EXT/calibration.yml'

# # data = load_calib(calib_file, 0)
# res,K_l, R_l, T_l = load_calib(calib_file, 0)
# res,K_r, R_r, T_r = load_calib(calib_file, 1)



#%%

images = '/Volumes/WH/data/animals_sfm/Hare'