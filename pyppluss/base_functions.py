#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 19:52:10 2017

@author: edanrein

v10-Oct-2018
-Cleaned up

v01-Aug-2018
-Cleaned up

v30-Aug-2017
-Notes on the transfer from MATLAB to python:
    1. No intersections became [] (empty) instead of NaN
    2. Still need to make the integrals for the non-uniform source case more accurate
    3. This file has all of the "basic" functions and is called by the other files.
"""

import numpy as np
import math as M
from pyppluss.fastqs import QuarticSolverVec as fastquartroots

def simpledist(u,v):
    """
    A simplified and more efficient 2-D euclidean distance calculation.
    The advantage over the built in (scipy) function is less checks.
    I do not check input validity.
    Assuming u,v are 2xn arrays, returns distance between n points.
    """
    return np.sqrt(np.sum((u-v)**2,0))

#%% Utility Functions: coordinate and argument swapping
def coord_swap (x_planet, y_planet, rotation_angle):
    """Rotation and translation of origin point
    
    Returns the new coordinates for the star's center after:
        1. Translation of the origin to (`x_planet`, `y_planet`)
        2. Rotation of the x axis `rotation_angle` radians
    
    Parameters
    ----------
    x_planet : ``1-D array``
        x coordinate of the planet's center
    y_planet : ``1-D array``
        y coordinate of the planet's center
    rotation_angle : ``float``
        Disk's rotation angle (in radians)

    Returns
    -------
    new_coords : ``2-D array``
        array of the (x,y) values of the new coordinates for the star's center.    
    """
    #This is because we rotate the axis rotation_angle, meaning the points go the other direcrtion
    #Using rotation matrix. Assuming rotation_angle is scalar, while the points may be vector.
    rotation_angle = -rotation_angle
    angle_cos = M.cos(rotation_angle)
    angle_sin = M.sin(rotation_angle)
    new_coords = np.dot(np.array([[angle_cos, -angle_sin],[angle_sin, angle_cos]]),-np.stack((x_planet,y_planet)))
    return new_coords

def get_coord_circle (arg,x_center,y_center,radius):
    """Calculates the argument for a point on a circle
    
    Calculates the coordinates for a point with argument `arg` on a circle with center (`x_center`, `y_center`) and radius `radius`.
    
    Parameters
    ----------
    arg : ``array``
        Argument of the point
    x_center : ``array``
        x coordinate of the circle's center
    y_center : ``array``
        y coordinate of the circle's center
    radius : ``array``
        radius of the circle
    
    Returns
    -------
    coords : ``array``
        a concatenated ``array`` (x,y) of the coordinates
    """
    x = x_center+radius*np.cos(arg)
    y = y_center+radius*np.sin(arg)
    return np.concatenate((x.reshape((1,-1)),y.reshape((1,-1))))

def get_coord_ellipse (arg,x_center,y_center,big_axis,small_axis):
    """Calculates the argument for a point on a ellipse
    
    Calculates the coordinates for a point with argument `arg` on a ellipse with center (`x_center`, `y_center`) and axes `big_axis` and `small_axis`.
    An ellipse behaves like a circle just not the same way (different r) in each axis.
    Parameters
    ----------
    arg : ``1-D array``
        Argument of the point (in radians)
    x_center : ``1-D array``
        x coordinate of the ellipse's center
    y_center : ``1-D array``
        y coordinate of the ellipse's center
    big_axis : ``1-D array``
        Big axis of the ellipse
    small_axis : ``1-D array``
        Small axis of the ellipse
    
    Returns
    -------
    coords : ``2-D array``
        a concatenated ``array`` (x,y) of the coordinates
    """
    k = big_axis/small_axis
    #Make sure atan is in the right quadrant
    angleF = np.arctan(k*np.tan(arg))
    #q4 and q1 are good with atan, only q2,q3 change quadrants inaccurately. This is a fix.
    angleT = angleF+(-np.sign(np.cos(arg))+1)*0.5*M.pi 
    x = x_center+big_axis*np.cos(angleT)
    y = y_center+small_axis*np.sin(angleT)
    return np.concatenate((x.reshape((1,-1)),y.reshape((1,-1))))
    
def get_arg (x, y, x_center, y_center):
    """Calculates the argument for a point relative to a center point
    
    Parameters
    ----------
    x : ``1-D array``
        x coordinate of the point
    y : ``1-D array``
        y coordinate of the point
    x_center : ``1-D array``
        x coordinate of the ellipse's center
    y_center : ``1-D array``
        y coordinate of the ellipse's center
    
    Returns
    -------
    coords : ``1-D array``
        The argument in radians, between 0 and 2*pi
    """
    arg = np.arctan2(y-y_center,x-x_center)
    #Make sure the angle is between 0 and 2*pi
    if np.isscalar(arg):
        if arg<0:
            arg += 2*M.pi
    else:
        arg[np.signbit(arg)] += 2*M.pi
    return arg

#%% Intersection Functions
def get_star_planet_intersection (x_star, y_star, radius_planet, tol):
    """Returns all the intersections between planet and star
    
    Returns all the intersections between:
    Star with center (`x_star`, `y_star`) and radius 1
    Planet with center (0,0) and radius `radius_planet`
    Using algorithm found on the internet - here: http://paulbourke.net/geometry/circlesphere/
    With the title **Intersection of two circles**
    Returns NaN in excess coordinates.
    
    Parameters
    ----------
    x_star : ``1-D array``
        x coordinate of the star's center
    y_star : ``1-D array``
        y coordinate of the star's center
    radius_planet : ``1-D array``
        Planet's radius relative to star's radius
    tol : ``float``
        Error tolerance parameter

    Returns
    -------
    intersections : ``2-D array``
        array of intersection points. Each point has its own column with the following data:
        x coordinate, y coordinate, star argument, planet argument
    """
    d = np.sqrt(x_star**2+y_star**2)
    #Checking existance of solutions
    star_planet_intersections = np.empty((len(x_star),4,2))
    star_planet_intersections[:] = np.NaN
    ind = np.logical_and(np.logical_and(d>abs(1-radius_planet), d<1+radius_planet), np.logical_not(d==0,radius_planet==1))
    if np.any(ind):
        d = d[ind]
        x_tmp = x_star[ind]
        y_tmp = y_star[ind]
        r_tmp = radius_planet[ind]
        #Calculating the solutions
        a = (1-r_tmp**2+d**2)/(2*d)
        h = np.sqrt(1-a**2)
        x2 = x_tmp-a*x_tmp/d
        y2 = y_tmp-a*y_tmp/d
        #Using np.concatenate to create a vector in the right shape.
        s1x = (x2-h*y_tmp/d).reshape(-1,1,1)
        s2x = (x2+h*y_tmp/d).reshape(-1,1,1)
        sx = np.concatenate((s1x,s2x),2)
        s1y = (y2+h*x_tmp/d).reshape(-1,1,1)
        s2y = (y2-h*x_tmp/d).reshape(-1,1,1)
        sy = np.concatenate((s1y,s2y),2)
        star_planet_intersections[ind,:2,:2] = np.concatenate((sx,sy),1)
        #Adding arguments
        mx = np.repeat(np.reshape(x_tmp,(len(x_tmp),1)),2,axis=1)
        my = np.repeat(np.reshape(y_tmp,(len(y_tmp),1)),2,axis=1)
        star_planet_intersections[ind,2,:2] = get_arg(star_planet_intersections[ind,0,:2], star_planet_intersections[ind,1,:2], mx, my)
        star_planet_intersections[ind,3,:2] = get_arg(star_planet_intersections[ind,0,:2], star_planet_intersections[ind,1,:2], 0, 0)
        #Checking duplicates
        veryclose = np.sum((star_planet_intersections[ind,:2,0]-star_planet_intersections[ind,:2,1])**2,1)<=tol
        star_planet_intersections[ind,:,1][veryclose] = np.NaN
    return star_planet_intersections

def get_star_disk_intersection (x_star, y_star, disk_radius, disk_inclination, tol):
    """Returns all the intersections between star and disk
    
    Returns all the intersections between:
    Star with center (`x_star`,`y_star`) and radius 1
    Disk with center (0,0), big axis `disk_radius` and small axis `disk-radius`*`disk_inclination`
    
    Parameters
    ----------
    x_star : ``1-D array``
        x coordinate of the star's center
    y_star : ``1-D array``
        y coordinate of the star's center
    disk_radius : ``1-D array``
        Disk's radius relative to star's radius
    disk_inclination : ``1-D array``
        Disk's inclination
    tol : ``float``
        Error tolerance parameter

    Returns
    -------
    intersections : ``2-D array``
        array of intersection points. Each point has its own column with the following data:
        x coordinate, y coordinate, star argument, disk argument
    """    
    #Calculate the coefficients of the polynomial
    #Polynomial in t using uniform trigonometric substitution (Weierstrass) for angle in disk.
    #NOTE: where a==0, it crashes.
    star_disk_intersections = np.empty([len(x_star),4,4])
    star_disk_intersections[:] = np.nan
    a = disk_radius
    b = a*np.cos(disk_inclination)
    num = x_star**2+y_star**2-1
    coeff_a = a**2+2*a*x_star+num
    coeff_b = -4*b*y_star
    coeff_c = -2*a**2+4*b**2+2*num
    coeff_d = coeff_b
    coeff_e = a**2-2*a*x_star+num
    specialcase = coeff_a==0
    specinds = np.nonzero(specialcase)[0]
    speclen = specinds.size

    if speclen>0:
        star_disk_intersections[specialcase,0,0]=-a[specialcase]
        star_disk_intersections[specialcase,1,0]=0.0
        for j in range(speclen):
            this_ind = specinds[j]
            t = np.empty(3)
            t[:] = np.nan
            tmp = np.roots([coeff_b[this_ind],coeff_c[this_ind],coeff_d[this_ind],coeff_e[this_ind]])
            t[:tmp.size] = tmp
            isnr = np.logical_not(t.imag==0)
            t[isnr] = np.nan
            #Convert from t to (x,y) coordinates
            x = a[this_ind]*(1-t**2)/(1+t**2)
            y = b[this_ind]*2*t/(1+t**2)
            star_disk_intersections[this_ind,0,1:] = x.real.T
            star_disk_intersections[this_ind,1,1:] = y.real.T
            
    #Every row of t corresponds to an input.
    regcase = np.logical_not(specialcase)
    t = fastquartroots(coeff_a[regcase].astype(complex),coeff_b[regcase].astype(complex),coeff_c[regcase].astype(complex),coeff_d[regcase].astype(complex),coeff_e[regcase].astype(complex))
    isnr = np.logical_not(t.imag==0)
    t[isnr] = np.nan
    #Convert from t to (x,y) coordinates
    x = a[regcase]*(1-t**2)/(1+t**2)
    y = b[regcase]*2*t/(1+t**2)

    star_disk_intersections[regcase,0,:] = x.real.T
    star_disk_intersections[regcase,1,:] = y.real.T
    cond = specialcase
    cond[regcase] = np.any(np.logical_not(isnr),0)
    #Calculating arguments
    mx = np.repeat(np.reshape(x_star[cond],(len(x_star[cond]),1)),4,axis=1)
    my = np.repeat(np.reshape(y_star[cond],(len(y_star[cond]),1)),4,axis=1)
    star_disk_intersections[cond,2,:] = get_arg(star_disk_intersections[cond,0,:],star_disk_intersections[cond,1,:],mx,my); #Star argument
    star_disk_intersections[cond,3,:] = get_arg(star_disk_intersections[cond,0,:],star_disk_intersections[cond,1,:],0,0); #Disk argument
    return star_disk_intersections

def get_disk_planet_intersection (radius_planet, disk_radius, disk_inclination, tol):
    """Returns all the intersections between planet and disk
    
    Returns all the intersections between:
    Planet with center (0,0) and radius `radius_planet`
    Disk with center (0,0), big axis `disk_radius` and small axis `disk-radius` * cos(`disk_inclination`)
    Using equation from Wolfram.
    
    Parameters
    ----------
    radius_planet : ``1-D array``
        Planet's radius relative to star's radius
    disk_radius : ``1-D array``
        Disk's radius relative to star's radius
    disk_inclination : ``float``
        Disk's inclination (in radians)
    tol : ``float``
        Error tolerance parameter

    Returns
    -------
    intersections : ``2-D array``
        array of intersection points. Each point has its own column with the following data:
        x coordinate, y coordinate, planet/disk argument
    """    
    n = len(radius_planet)
    disk_planet_intersections = np.zeros([n,3,4])
    b = disk_radius*M.cos(disk_inclination)
    cond = np.logical_or(b>radius_planet,disk_radius<radius_planet)
    disk_planet_intersections[cond,:,:] = np.nan
    cond = np.logical_not(cond)
    n = len(radius_planet[cond])
    rp = radius_planet[cond]
    rd = disk_radius[cond]
    b = b[cond]
    #Calculations
    disk_planet_intersections[cond,0,:2] = np.repeat(rd*np.sqrt((rp-b)*(rp+b)/(rd**2-b**2)),2).reshape((n,2))
    disk_planet_intersections[cond,0,2:4] = -disk_planet_intersections[cond,0,:2]
    disk_planet_intersections[cond,1,:3:2] = np.repeat(b*np.sqrt((rd**2-rp**2)/(rd**2-b**2)),2).reshape((n,2))
    disk_planet_intersections[cond,1,1:4:2] = -disk_planet_intersections[cond,1,:3:2]
    #Void duplicates when x value is too small
    disk_planet_intersections[cond,:,2:][2*disk_planet_intersections[cond,0,0]<=tol,:,:] = np.nan
    #Void duplicates when y value is too small
    disk_planet_intersections[cond,:,:3:2][2*disk_planet_intersections[cond,1,0]<=tol,:,:] = np.nan
    #Planet/Disk argument
    disk_planet_intersections[cond,2,:] = get_arg(disk_planet_intersections[cond,0,:],disk_planet_intersections[cond,1,:],0,0)
    return disk_planet_intersections


#%% Sector area calculation helpers
def triangle_area (x1,y1,x2,y2,x3,y3):
    """Calculates triangle area
    
    The calculation is made based on the (x,y) positions of the vertices.
    
    Parameters
    ----------
    x1 : ``1-D array``
        x coordinate of the first vertex
    y1 : ``1-D array``
        y coordinate of the first vertex
    x2 : ``1-D array``
        x coordinate of the second vertex
    y2 : ``1-D array``
        y coordinate of the second vertex
    x3 : ``1-D array``
        x coordinate of the third vertex
    y3 : ``1-D array``
        y coordinate of the third vertex
        
    Returns
    -------
    area : ``1-D array``
        Area of triangle defined by the three veritces
    """
    #This is a simple case of the polygon area equation.
    area = 0.5*abs(x1*(y3-y2)+x2*(y1-y3)+x3*(y2-y1))
    return area

def ellipse_to_x (big_axis,small_axis,angle):
    """Calculates sector area of the ellipse
    
    This is a helper function intended to aid sector_area_disk in calculations
    assuming all angles are between 0 and 2pi
    
    Parameters
    ----------
    big_axis : ``1-D array``
        The long axis of the ellipse
    small_axis : ``1-D array``
        The short axis of the ellipse
    angle : ``1-D array``
        Angle of the point relative to the positive direction of the long axis
    
    Returns
    -------
    area : ``1-D array``
        Area of sector between the positive direction of the long axis and angle
    """
    #The Formula is present in the paper
    rangle = np.round(angle/np.pi)*np.pi
    area = big_axis*small_axis*rangle*0.5+0.5*big_axis*small_axis*np.arctan(big_axis/small_axis*np.tan(angle-rangle))
    return area
    
#%% Sector area calculations
def sector_area_planet (arg1, arg2, radius_planet):
    """Calculates sector area of the planet
    
    The sector area is between `arg1` and `arg2` in circle with radius `radius planet` and center at origin.
    The calculation is done as required for the sector based algorithm.
    We assume that the area is in the **counterclockwise** direction from `arg1` to `arg2`
    
    Parameters
    ----------
    arg1 : ``1-D array``
        The angle to start going counterclockwise from (in radians)
    arg2 : ``1-D array``
        The angle end of the sector (in radians)
    radius_planet : ``1-D array``
        The radius of the planet
    Returns
    -------
    area : ``1-D array``
        The requested sector area
    """
    if np.isscalar(arg1):        
        if arg2<arg1 :
            area = M.pi*radius_planet**2-0.5*radius_planet**2*(arg1-arg2)
        else:
            area = 0.5*radius_planet**2*(arg2-arg1)
    else:
        area = 0.5*radius_planet**2*(arg2-arg1)
        cond = arg2<arg1
        area[cond] = M.pi*radius_planet[cond]**2+area[cond]
    return area

def sector_area_star (arg1, arg2):
    """Calculates sector area of the star
    
    The sector area is between `arg1` and `arg2` in circle with radius 1 and center at (`x_star`, `y_star`).
    The calculation is done as required for the sector based algorithm.
    We assume that the area is in the **counterclockwise** direction from `arg1` to `arg2`
    
    Parameters
    ----------
    arg1 : ``1-D array``
        The angle to start going counterclockwise from (in radians)
    arg2 : ``1-D array``
        The angle end of the sector (in radians)
    x_star : ``1-D array``
        x coordinate of the star's center
    y_star : ``1-D array``
        y coordinate of the star's center
    Returns
    -------
    area : ``1-D array``
        The requested sector area
    """
    secArea = 0.5*(arg2-arg1)
    cond = arg2<arg1
    secArea[cond] = M.pi+secArea[cond]
    return secArea

def sector_area_disk (arg1, arg2, disk_radius, disk_inclination):
    """Calculates sector area of the disk
    
    The sector area is between `arg1` and `arg2` in ellipse with radius `disk_radius`, inclination `disk_inclination` and center at origin.
    The calculation is done as required for the sector based algorithm.
    We assume that the area is in the **counterclockwise** direction from `arg1` to `arg2`
    
    Parameters
    ----------
    arg1 : ``1-D array``
        The angle to start going counterclockwise from (in radians), between 0 and 2pi
    arg2 : ``1-D array``
        The angle end of the sector (in radians), between 0 and 2pi
    disk_radius : ``1-D array``
        The disk's big axis
    disk_inclination : ``float``
        the inclination of the disk (in radians)
    Returns
    -------
    area : ``1-D array``
        The requested sector area
    """
    if np.isscalar(arg1):
        if arg2>arg1:
            area = ellipse_to_x(disk_radius,disk_radius*M.cos(disk_inclination),arg2)-ellipse_to_x(disk_radius,disk_radius*M.cos(disk_inclination),arg1)
        else:
            area = M.pi*disk_radius**2*M.cos(disk_inclination)-(ellipse_to_x(disk_radius,disk_radius*M.cos(disk_inclination),arg1)-ellipse_to_x(disk_radius,disk_radius*M.cos(disk_inclination),arg2))
    else:
        area = ellipse_to_x(disk_radius,disk_radius*M.cos(disk_inclination),arg2)-ellipse_to_x(disk_radius,disk_radius*M.cos(disk_inclination),arg1)
        cond = arg2<arg1
        area[cond] = M.pi*disk_radius[cond]**2*M.cos(disk_inclination)+area[cond]
    return area

#%% Border helper
def get_dha_border (radius_planet, disk_radius, disk_inclination, x_star, y_star, star_planet_intersections, star_disk_intersections, disk_planet_intersections, tol):
    """Finds which intersection points are on the dha border
    
    Returns the dha_border as required by double hidden area calculations
    
    Parameters
    ----------
    radius_planet : ``1-D array``
        Planet's radius
    disk_radius : ``1-D array``
        Disk's radius
    disk_inclination : ``float``
        Disk's inclination (in radians)
    x_star : ``1-D array``
        x coordinate of the star's center
    y_star : ``1-D array``
        y coordinate of the star's center
    star_planet_intersections : ``2-D array``
        the array of star-planet intersections as returned by get_star_planet_intersection
    star_disk_intersections : ``2-D array``
        the array of star-disk intersections as returned by get_star_disk_intersection
    disk_planet_intersections : ``2-D array``
        the array of disk-planet intersections as returned by get_disk_planet_intersection
    tol : ``float``, optional
        Error tolerance parameter
    
    Returns
    -------
    dha_border : ``2-D array``
        Points that are on the border of the dha. each point is a column consisting of x,y coordinates, argument relative to planet's center and curves.
    """
    # 0=star, 1=planet, 2=disk
    #These identifiers are on row  #3 and #4
    #Taking only the argument relative to the planet's center
    #Stacking them all together on the right dimension
    n = star_planet_intersections.shape[0]
    this_spi = np.concatenate((star_planet_intersections[:,[0,1,3],:],np.zeros((n,1,2)),np.ones((n,1,2)),np.nan*np.empty((n,1,2))),1)
    this_sdi = np.concatenate((star_disk_intersections[:,[0,1,3],:],np.zeros((n,1,4)),2*np.ones((n,1,4)),np.nan*np.empty((n,1,4))),1)
    this_dpi = np.concatenate((disk_planet_intersections,np.ones((n,1,4)),2*np.ones((n,1,4)),np.nan*np.empty((n,1,4))),1)
    all_intersections = np.concatenate((this_spi,this_sdi,this_dpi),-1)
    if all_intersections.ndim<3:
        all_intersections = np.reshape(all_intersections, (1,all_intersections.shape[0], all_intersections.shape[1]))
    #Look for dha border only when there are intersections
    iexist = np.logical_not(np.all(np.isnan(all_intersections[:,0,:]),-1))
    test_intersections = all_intersections[iexist,:,:]
    test_intersections.shape = (-1,all_intersections.shape[1], all_intersections.shape[2])
    if test_intersections.size>0:
        #There are 10 intersection points (including nans)
        sdist_edge = np.sqrt((test_intersections[:,0,:]-np.repeat(x_star[iexist].reshape((-1,1)),10,1))**2+(test_intersections[:,1,:]-np.repeat(y_star[iexist].reshape((-1,1)),10,1))**2)-1
        pdist_edge = np.sqrt(test_intersections[:,0,:]**2+test_intersections[:,1,:]**2)-np.repeat(radius_planet[iexist].reshape((-1,1)),10,1)
        #Signbit is the same as <0
        is_in_star = np.signbit(sdist_edge-tol)
        is_in_planet = np.signbit(pdist_edge-tol)
        disk_dist_1 = np.sqrt((test_intersections[:,0,:]-np.repeat(disk_radius[iexist].reshape((-1,1)),10,1)*M.sin(disk_inclination))**2+test_intersections[:,1,:]**2)
        disk_dist_2 = np.sqrt((test_intersections[:,0,:]+np.repeat(disk_radius[iexist].reshape((-1,1)),10,1)*M.sin(disk_inclination))**2+test_intersections[:,1,:]**2)
        ddist_edge = disk_dist_1+disk_dist_2-2*np.repeat(disk_radius[iexist].reshape((-1,1)),10,1)
        is_in_disk = np.signbit(ddist_edge-tol)
        is_dha_border = np.logical_and(np.logical_and(is_in_star,is_in_planet),is_in_disk)
        #Numpy>=1.10.0, so swapaxes is view and not copy
        np.swapaxes(test_intersections,0,1)[:,np.logical_not(is_dha_border)] = np.nan
        dexist = np.logical_not(np.all(np.isnan(test_intersections[:,0,:]),-1))
        #dborder and test_intersections are not seperate arrays from all_intersections.
        #They seem to function as pointers?
        dborder = test_intersections[dexist,:,:]
        if dborder.size>0:
            #Adding third curve if very close
            dtriple = np.signbit(abs(ddist_edge[dexist,:2])-tol)
            ptriple = np.signbit(abs(pdist_edge[dexist,2:6])-tol)
            striple = np.signbit(abs(sdist_edge[dexist,6:])-tol)
            np.swapaxes(dborder[:,3:,:2],1,2)[dtriple,:] = np.array([0,1,2])
            np.swapaxes(dborder[:,3:,2:6],1,2)[ptriple,:] = np.array([0,1,2])
            np.swapaxes(dborder[:,3:,6:],1,2)[striple,:] = np.array([0,1,2])
        #Ensuring no pointer issues (again)
        test_intersections[dexist,:,:] = dborder
        all_intersections[iexist,:,:] = test_intersections
    else:
        all_intersections[:] = np.nan
    return all_intersections
