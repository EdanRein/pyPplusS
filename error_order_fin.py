#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:57:19 2018

@author: edanrein

v14-May-2019
-Added to pyPplusS

v26-jun-2018
-New Code
"""

if __name__ == "__main__":
    from pyppluss.segment_models import LC_ringed as LC
    import numpy as np
    import matplotlib.pyplot as plt
    #number of points
    n=1000
    #Parameters for the test
    x_planet = np.linspace(-1.1,1.1,2*n)
    y_planet = np.zeros_like(x_planet)+0.0
    radius_planet = np.ones_like(x_planet)*0.11
    radius_in = np.ones_like(x_planet)*10000.44
    radius_out = np.ones_like(x_planet)*10000.7
    ring_inclination = 0
    ring_rotation = 0 
    opacity = 1.0
    #Range of orders for which the errors will be calculated
    orders = np.arange(3,15)
    #"True" order - this will be considered the true value
    trueorder = 20
    err = np.empty((len(orders),len(x_planet)))
    valstrue = vals = LC(radius_planet, radius_in, radius_out, x_planet, y_planet, ring_inclination, ring_rotation, opacity, 0.0, 0.448667, 0.0 ,0.313276,n_center=trueorder,n_gress=trueorder)
    plt.figure()
    for k in range(len(orders)):
        vals = LC(radius_planet, radius_in, radius_out, x_planet, y_planet, ring_inclination, ring_rotation, opacity, 0.0, 0.448667, 0.0 ,0.313276,n_center=orders[k],n_gress=orders[k])
        err[k] = abs(vals-valstrue)
        #Plotting
        plt.semilogy(x_planet,err[k],'o-')
        plt.text(x_planet[n],err[k][n],"n="+orders[k].__str__())
        
    plt.grid()
