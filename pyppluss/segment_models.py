#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 18:12:03 2017

@author: edanrein
This file has finalized models and a version ready for celerite and data generation.

v10-Oct-2018
-Cleaned Up
-Note: some functions are only useful when using commented-out debugging-related parts of the code.

v02-Aug-2018
-Cleaned up

v25-Jun-2018
-Modified celerite model to reflect Aviv's recommendations (re:normalization)
-Note: To avoid bugs, did not "clean up" past variables probably no longer used but you never know

v4-Jun-2018
-Fixed: for mucrit!=0, algorithm was incorrect in using uniform source
-Added: Support for two different Gaussian Quadrature orders: one for ingress/egress and the other for the rest.

v26-May-2018
-Implemented sub-intervals splitting
-Fixed: Bug when y_star=0 in Quartic eq. for min-max disk-star distance
-Fixed: Bug when loss of accuracy in conversion form r->t caused issues with subinterval calculations

"""
import numpy as np
import math as M
from pyppluss.base_functions import coord_swap, get_disk_planet_intersection, get_star_disk_intersection, get_star_planet_intersection
from pyppluss.polygon_plus_segments import intersection_area as double_hidden_area
from pyppluss.polygon_plus_segments import handler, Star, Planet, Disk, border_sort
from pyppluss.fastqs import QuarticSolverVec as fastquartroots
from scipy.special import roots_legendre

#%% Base Models
def planet_hidden_area(radius_planet, x_star, y_star, star_planet_intersections, tol):
    """
    Calculating the area hidden by the planet.
    Models using DoubleHiddenAreaAlgorithm for consistency and lack of errors.
    
    Parameters
    ----------
    radius_planet : ``1-D array``
        Planet's radius relative to star's radius
    x_star : ``1-D array``
        x coordinate of the star's center
    y_star : ``1-D array``
        y coordinate of the star's center
    star_planet_intersections : ``2-D array``
        the array of star-planet intersections as returned by get_star_planet_intersection
    tol : ``float``, optional
        Error tolerance parameter
    Returns
    -------
    planet_area : ``1-D array``
        Area hidden by the planet
    """
    #The commented-out code is a case-specific implememtnation of Step 1.1.1 of the algorithm.
#    #Initilize with zeros to avoid special handling of another special case
#    planet_area = np.zeros_like(x_star)
#    calcr = radius_planet.copy()
#    calcr[calcr<1] = 1
#    #If there are <2 intersections and the planet is hiding part of the star
#    scalc = np.logical_and(np.any(np.isnan(star_planet_intersections),(1,2)),np.sqrt(x_star**2+y_star**2)<=calcr)
#    calcr = radius_planet[scalc].copy()
#    calcr[calcr>1] = 1
#    planet_area[scalc] = M.pi*calcr**2
#    #If there are 2 intersections, call the hidden area calculator.
#    scalc = np.any(np.isnan(star_planet_intersections),(1,2))
#    remcalc = np.logical_not(scalc)
#    border = border_sort(star_planet_intersections[remcalc])
#    pangles = border[:,3,:]
#    this_planet = Planet(radius_planet[remcalc], pangles, np.ones_like(pangles, dtype=bool))
#    this_star = Star(np.concatenate((x_star[remcalc].reshape((1,-1)),y_star[remcalc].reshape((1,-1)))), border[:,2,:], np.ones_like(pangles, dtype=bool))
#    planet_area[remcalc] = double_hidden_area((this_star,this_planet),border,tol)
    
    #Prepare the objects for the algorithm
    border = border_sort(star_planet_intersections)
    pangles = border[:,3,:]
    this_planet = Planet(radius_planet, pangles, np.ones_like(pangles, dtype=bool))
    this_star = Star(np.concatenate((x_star.reshape((1,-1)),y_star.reshape((1,-1)))), border[:,2,:], np.ones_like(pangles, dtype=bool))
    #Calculate the area
    planet_area = double_hidden_area((this_star,this_planet),border,tol)        
    return planet_area

def disk_hidden_area(disk_radius, disk_inclination, x_star, y_star, star_disk_intersections, tol):
    """
    Calculating the area hidden by the planet.
    Models using DoubleHiddenAreaAlgorithm for consistency and lack of errors.
    
    Parameters
    ----------
    disk_radius : ``1-D array``
        Disk's radius relative to star's radius
    disk_inclination : ``float``
        Disk's inclination (in radians)
    x_star : ``1-D array``
        x coordinate of the star's center
    y_star : ``1-D array``
        y coordinate of the star's center
    star_disk_intersections : ``2-D array``
        the array of star-disk intersections as returned by get_star_disk_intersection
    tol : ``float``, optional
        Error tolerance parameter
    Returns
    -------
    disk_area : ``1-D array``
        Area hidden by the disk
    """
    #The commented-out code is a case-specific implememtnation of Step 1.1.1 of the algorithm.
#    disk_area = np.zeros_like(x_star)
#    scalc = np.all(np.isnan(star_disk_intersections),(1,2))
#    simplei = scalc.copy()
#    sx = x_star[scalc]
#    sy = y_star[scalc]
#    sr = disk_radius[scalc]
#    sinc = M.sin(disk_inclination)
#    instar = sx**2+sy**2<=(1+tol)**2
#    indisk = np.sqrt((sx-sr*sinc)**2+sy**2)+np.sqrt((sx+sr*sinc)**2+sy**2)<=2*sr+tol
#    incalc = np.logical_or(instar,indisk)
#    sr = sr[incalc]**2*M.cos(disk_inclination)
#    sr[sr>1] = 1
#    scalc[np.nonzero(scalc)[0][np.logical_not(incalc)]] = False
#    disk_area[scalc] = M.pi*sr
#    scalc = simplei
#    remcalc = np.logical_not(scalc)
#    border = border_sort(star_disk_intersections[remcalc])
#    dangle = border[:,3,:]
#    this_disk = Disk(disk_radius[remcalc], disk_inclination, dangle, np.ones_like(dangle, dtype=bool))
#    this_star = Star(np.concatenate((x_star[remcalc].reshape((1,-1)),y_star[remcalc].reshape((1,-1)))), border[:,2,:], np.ones_like(dangle, dtype=bool))
#    disk_area[remcalc] = double_hidden_area((this_star,this_disk),border,tol)
    
    #Prepare the objects for the algorithm
    border = border_sort(star_disk_intersections)
    dangle = border[:,3,:]
    this_disk = Disk(disk_radius, disk_inclination, dangle, np.ones_like(dangle, dtype=bool))
    this_star = Star(np.concatenate((x_star.reshape((1,-1)),y_star.reshape((1,-1)))), border[:,2,:], np.ones_like(dangle, dtype=bool))
    #Calculate the area
    disk_area = double_hidden_area((this_star,this_disk),border,tol)
    
    return disk_area

def tot_hidden_area(radius_planet, radius_in, radius_out, x_star, y_star, ring_inclination, star_planet_intersections, star_disk_intersections_in, star_disk_intersections_out, disk_planet_intersections_in, disk_planet_intersections_out, opacity, tol=10**-10):
    """
    Calculation of the total hidden area of the planet and ring together
    
    Parameters
    ----------
    radius_planet : ``1-D array``
        Planet's radius
    radius_in : ``1-D array``
        Ring's inner radius
    radius_out : ``1-D array``
        Ring's outer radius
    x_star : ``1-D array``
        x coordinate of the star's center
    y_star : ``1-D array``
        y coordinate of the star's center
    ring_inclination : ``float``
        Ring's inclination (in radians)
    star_planet_intersections : ``2-D array``
        the array of star-planet intersections as returned by get_star_planet_intersection
    star_disk_intersections_in : ``2-D array``
        the array of star-inner disk intersections as returned by get_star_disk_intersection
    star_disk_intersections_out : ``2-D array``
        the array of star-outer disk intersections as returned by get_star_disk_intersection
    disk_planet_intersections_in : ``2-D array``
        the array of inner disk-planet intersections as returned by get_disk_planet_intersection
    disk_planet_intersections_out : ``2-D array``
        the array of outer disk-planet intersections as returned by get_disk_planet_intersection
    opacity : ``float``
        Ring's opacity
    tol : ``float``, optional
        Error tolerance parameter
    
    Returns
    -------
    hidden_area : ``1-D array``
        Total hidden area by the planet and ring.
    """
    #Planet hidden area
    planet_area = planet_hidden_area(radius_planet, x_star, y_star, star_planet_intersections, tol)
    #Disks hidden area
    disk_in_area = disk_hidden_area(radius_in, ring_inclination, x_star, y_star, star_disk_intersections_in, tol)
    disk_out_area = disk_hidden_area(radius_out, ring_inclination, x_star, y_star, star_disk_intersections_out, tol)
    #Double hidden area
    #Initial values assuming no intersections
    double_area_in = np.minimum(planet_area,disk_in_area)
    double_area_out = np.minimum(planet_area,disk_out_area)
    #When there are intersections, call the algorithm to find the double hidden area.
    calcin = np.logical_and(np.logical_and(planet_area>0,disk_in_area>0),np.any(np.logical_not(np.isnan(disk_planet_intersections_in)),(1,2)))
    star, planet, disk, dha_border_in = handler(radius_planet[calcin], radius_in[calcin], ring_inclination, x_star[calcin], y_star[calcin], star_planet_intersections[calcin], star_disk_intersections_in[calcin], disk_planet_intersections_in[calcin], tol)
    double_area_in[calcin] = double_hidden_area((star, planet, disk), dha_border_in, tol)
    calcout = np.logical_and(np.logical_and(planet_area>0,disk_out_area>0),np.any(np.logical_not(np.isnan(disk_planet_intersections_out)),(1,2)))
    star, planet, disk, dha_border_out = handler(radius_planet[calcout], radius_out[calcout], ring_inclination, x_star[calcout], y_star[calcout], star_planet_intersections[calcout], star_disk_intersections_out[calcout], disk_planet_intersections_out[calcout], tol)
    double_area_out[calcout] = double_hidden_area((star, planet, disk), dha_border_out, tol)
    #Conclusions
    ring_area = (disk_out_area-double_area_out)-(disk_in_area-double_area_in)
    hidden_area = opacity*ring_area+planet_area
    return hidden_area
#%% Paper Models
def uniform_rings(radius_planet, radius_in, radius_out, x_planet, y_planet, ring_inclination, ring_rotation, opacity=1.0, tol=10**-7):
    """LC value for planet with rings and uniform source
    
    Calculated using Polygon+Segments algorithm
    Assumes the input vectors are of the same length.

    Parameters
    ----------
    radius_planet : ``1-D array``
        Planet's radius
    radius_in : ``1-D array``
        Inner disk's radius
    radius_out : ``1-D array``
        Outer disk's radius
    x_planet : ``1-D array``
        x coordinate of the planet's center
    y_planet : ``1-D array``
        y coordinate of the planet's center
    ring_inclination : ``float``
        Inclination of the ring
    ring_rotation : ``float``
        Rotation of the ring
    opacity : ``float`` (0 to 1), optional
        Optical opacity for the ring hiding calculations.
    tol : ``float``, optional
        Error tolerance parameter
    Returns
    -------
    LC : ``1-D array``
        Estimated normalized light-curve value for the planet with rings.
    """
    #Input Validation
    LC = np.empty_like(x_planet)
    cond = np.isfinite(x_planet)
    LC[np.logical_not(cond)] = 1.0
    x_planet = x_planet[cond]
    y_planet = y_planet[cond]
    radius_in = radius_in[cond]
    radius_out = radius_out[cond]
    radius_planet = radius_planet[cond]
    #Swapping coordinate systems
    star_coords = coord_swap(x_planet, y_planet, ring_rotation)
    x_star = star_coords[0]
    y_star = star_coords[1]
    #Intersections
    star_planet_intersections = get_star_planet_intersection(x_star, y_star, radius_planet, tol)
    star_disk_intersections_in = get_star_disk_intersection(x_star, y_star, radius_in, ring_inclination, tol)
    star_disk_intersections_out = get_star_disk_intersection(x_star, y_star, radius_out, ring_inclination, tol)
    disk_planet_intersections_in = get_disk_planet_intersection(radius_planet, radius_in, ring_inclination, tol)
    disk_planet_intersections_out = get_disk_planet_intersection(radius_planet, radius_out, ring_inclination, tol)
    #Area Calculation & LC conclusions
    hidden_area = tot_hidden_area(radius_planet, radius_in, radius_out, x_star, y_star, ring_inclination, star_planet_intersections, star_disk_intersections_in, star_disk_intersections_out, disk_planet_intersections_in, disk_planet_intersections_out, opacity, tol)
    LC[cond] = 1-hidden_area/M.pi
    return LC


def LC_ringed(radius_planet, radius_in, radius_out, x_planet, y_planet, ring_inclination, ring_rotation, opacity, c1,c2,c3,c4, mucrit=0.0, n_center=5, n_gress=5, tol=10**-10):
    """LC value(s) for planet with rings and non uniform source
    
    Assuming quadratic limb darkening with paramters u1, u2
    Assign string value to c3 to signify quadratic LD
    
    Parameters
    ----------
    radius_planet : ``1-D array``
        Planet's radius
    radius_in : ``1-D array``
        Inner disk's radius
    radius_out : ``1-D array``
        Outer disk's radius
    x_planet : ``1-D array``
        x coordinate of the planet's center
    y_planet : ``1-D array``
        y coordinate of the planet's center
    ring_inclination : ``float``
        Inclination of the ring
    ring_rotation : ``float``
        Rotation of the ring
    opacity : ``float`` (0 to 1)
        Optical opacity for the ring hiding calculations.
    c1, c2 , c3, c4 : ``float``
        Limb darkening coefficients
    mucrit : ``float``, optional
        A critical mu, after which the LD is constant.
    n_cener,n_gress : ``int``, optional
        Order of Gaussian Quadrature in the integration
    tol : ``float``, optional
        Error tolerance parameter
    Returns
    -------
    LC : ``1-D array``
        Estimated normalized light-curve value for the planet with rings.
    """
    #Convert quadratic limb darkening coefficients into nonlinear
    if str(type(c3))=="<class 'str'>":
        c2 = c2+2*c4
        c4 = -c4
        c3 = 0
        c1 = 0        
    #Caclulate the edge of the interval of the integral
    rcrit = np.sqrt(1-mucrit**2)
    #Calculate Uniform-Source Value
    uniform_val = uniform_rings(radius_planet/rcrit, radius_in/rcrit, radius_out/rcrit, x_planet/rcrit, y_planet/rcrit, ring_inclination, ring_rotation, opacity, tol)
    #Integration
    #Function to be integrated is (c1+2*c2*t+3*c3*t**2+4*c4*t**3)*(uniform_rings(rp/rt,rin/rt,rout/rt,xp/rt,yp/rt,ring_inclination,ring_rotation, opacity, tol))*(1-t**4)
    #Where rt=sqrt(1-t**4)
    #Take stuff from uniform_val (e.g. intersection coords) and pass to integrate_swapped to save time
    y = integrate_swapped(n_center,n_gress, radius_planet, radius_in, radius_out, x_planet, y_planet, ring_inclination, ring_rotation, opacity, c1,c2,c3,c4, mucrit, tol, uniform_val)
    LC = (1-c1*(1-mucrit**0.5)-c2*(1-mucrit)-c3*(1-mucrit**1.5)-c4*(1-mucrit**2))*uniform_val+y
    #See integral in: https://www.wolframalpha.com/input/?i=integral+of+2x(1-c_1(1-(1-x%5E2)%5E(1%2F4))-c_2(1-(1-x%5E2)%5E(1%2F2))-c_3(1-(1-x%5E2)%5E(3%2F4))-c_4(1-(1-x%5E2)))
    star_area = -0.5*c4*rcrit**4-(c1+c2+c3-1)*rcrit**2-0.8*c1*(1-rcrit**2)**(5/4)-2/3*c2*(1-rcrit**2)**(3/2)-4/7*c3*(1-rcrit**2)**(7/4)+0.8*c1+2/3*c2+4/7*c3
    LC=LC/star_area
    return LC

def indefinite_integral_helper(x,c1,c2,c3,c4):
    """Indefinite Integral of for use in the following functions.
    The integral is after the swapping of t=(1-r^2)^(1/4). Assuming Fe=1.
    """
    return c1*x*(1-x**4/5)+c2*x**2*(1-x**4/3)+c3*x**3*(1-3/7*x**4)+c4*x**4*(1-0.5*x**4)

def integrated_swapped_end(t,c1,c2,c3,c4):
    """Integral from t to 1 of inetgral_swapped,
    assuming all area from uniform source is outside
    Remember t=1 is the center of the star and not the limb
    """
    return indefinite_integral_helper(1,c1,c2,c3,c4)-indefinite_integral_helper(t,c1,c2,c3,c4)
    
def integrated_swapped_start(t, c1, c2, c3, c4, c,a):
    """ Integral from a to t of inetgral_swapped,
    assuming all area from uniform source is already inside
    A_h=pi*(1-c) => Fe = 1-(1-c)/r^2
    => f(t) = I(r(t))*(1-(1-c)/(1-t^4))*(1-t^4) = I(r(t))*(1-t^4)-I(r(t))*(1-c)
    """
    return indefinite_integral_helper(t,c1,c2,c3,c4)-indefinite_integral_helper(a,c1,c2,c3,c4)-(1-c)*(c1*t+c2*t**2+c3*t**3+c4*t**4-c1*a-c2*a**2-c3*a**3-c4*a**4)

def vec_fixed_quad(x,w,n,a,b,rp,rin,rout,ring_inclination,ring_rotation,opacity,xp,yp,c1,c2,c3,c4,tol):
    """
    Calculating fixed-order Gaussian Quadrature integration sample values.
    """
    t = (np.dot(0.5*(b-a).reshape((-1,1)),(x+1).reshape((1,-1)))+np.repeat(a.reshape((-1,1)),n,1)).flatten(order='F')
    #Conversion to r
    rt = np.sqrt(1-t**4)
    #Tiling input vectors similarly to t.
    rp = np.repeat(rp.reshape((1,-1)),n,0).flatten()
    rin = np.repeat(rin.reshape((1,-1)),n,0).flatten()
    rout = np.repeat(rout.reshape((1,-1)),n,0).flatten()
    xp = np.repeat(xp.reshape((1,-1)),n,0).flatten()
    yp = np.repeat(yp.reshape((1,-1)),n,0).flatten()
    return ((c1+2*c2*t+3*c3*t**2+4*c4*t**3)*(uniform_rings(rp/rt,rin/rt,rout/rt,xp/rt,yp/rt,ring_inclination,ring_rotation, opacity, tol))*(1-t**4)).reshape((-1,n),order='F')

def integrate_swapped(n_center,n_gress,radius_planet, radius_in, radius_out, x_planet, y_planet, ring_inclination, ring_rotation, opacity, c1,c2,c3,c4, mucrit, tol,c):
    """Function for numerical integration: Calculating weights+function values and returning estimation of the integral.
    General case with changing uniform source.
    """
    rcrit = np.sqrt(1-mucrit**2)
    tcrit = np.sqrt(mucrit)
    #Find values reuired for  interval split calculations from helper functions
    star_coords = coord_swap(x_planet, y_planet, ring_rotation)
    x_star = star_coords[0]
    y_star = star_coords[1]
    mx = np.repeat(np.reshape(x_star,(len(x_star),1)),4,axis=1)
    my = np.repeat(np.reshape(y_star,(len(y_star),1)),4,axis=1)
    mrin = np.repeat(np.reshape(radius_in,(len(y_star),1)),4,axis=1)
    mrout = np.repeat(np.reshape(radius_out,(len(y_star),1)),4,axis=1)
    d = np.sqrt(x_planet**2+y_planet**2)
    #Calculate all of the potential interval splits
    splitr = np.empty((radius_planet.size,16))
    #Start and end of the interval
    splitr[:,0] = np.zeros_like(radius_planet)
    splitr[:,1] = np.ones_like(radius_planet)*rcrit
    #Minimum and maximum star-planet distance
    splitr[:,2] = abs(d-radius_planet)
    splitr[:,3] = d+radius_planet
    #Inner ring-planet intersections
    inner_intersections = get_disk_planet_intersection(radius_planet, radius_in, ring_inclination, tol)
    splitr[:,4:8] = np.sqrt((inner_intersections[:,0,:]-mx)**2+(inner_intersections[:,1,:]-my)**2)
    #Outer ring-planet intersections
    outer_intersections = get_disk_planet_intersection(radius_planet, radius_out, ring_inclination, tol)
    splitr[:,8:12] = np.sqrt((outer_intersections[:,0,:]-mx)**2+(outer_intersections[:,1,:]-my)**2)
 
    #Minimum and maximum inner ring-star distance
    #Solve the quartic
    coeff_a = radius_in*np.cos(ring_inclination)*y_star
    cond = coeff_a==0
    dists = np.empty((len(coeff_a),4),dtype=float)
    #Handler of special case
    #Here, we assume a ring exists:  0<radius_in*np.cos(ring_inclination) therefore cond is true iff y_star=0 iff the solutions for theta are 0 and pi.
    #Therefore the substituiton for t is ineffective, so we will use theta instead
    theta = np.empty((len(coeff_a[cond]),2),dtype=float)
    theta[:,0] = np.pi
    theta[:,1] = 0.0
    dists[cond,:2] = np.sqrt((my[cond,:2]-2*mrin[cond,:2]*np.cos(ring_inclination)*np.sin(theta))**2+(mx[cond,:2]-mrin[cond,:2]*np.cos(theta))**2)
    dists[cond,2:] = dists[cond,:2]

    #Quartic solver
    cond = np.logical_not(cond)
    coeff_b = 2*radius_in[cond]*x_star[cond]+2*radius_in[cond]**2*np.sin(ring_inclination)**2
    coeff_c = np.zeros_like(coeff_a[cond])
    coeff_d = 2*radius_in[cond]*x_star[cond]-2*radius_in[cond]**2*np.sin(ring_inclination)**2
    coeff_e = -coeff_a[cond]
    t = fastquartroots(coeff_a[cond].astype(complex),coeff_b.astype(complex),coeff_c.astype(complex),coeff_d.astype(complex),coeff_e.astype(complex)).T
    #Eliminate complex values
    t[np.iscomplex(t)] = np.nan
    t = t.real
    #Calculate distance at the extremum points
    dists[cond] = np.sqrt((my[cond]-2*mrin[cond]*np.cos(ring_inclination)*t/(1+t**2))**2+(mx[cond]-mrin[cond]*(1-t**2)/(1+t**2))**2)
    #Add the minimum and maximum to the interval split. (Ignoring NaNs)
    splitr[:,12] = np.nanmin(dists,-1)
    splitr[:,13] = np.nanmax(dists,-1)

    #Minimum and maximum outer ring-star distance
    #Solve the quartic: 2 a^2 t^3 - 2 a^2 t + 2 a c t^3 + 2 a c t - 2 b^2 t^3 + 2 b^2 t + b d t^4 - b d
#    + b d t^4 
#    +2 a^2 t^3 + 2 a c t^3 - 2 b^2 t^3 
#    
#    - 2 a^2 t + 2 a c t + 2 b^2 t 
#    - b d

    coeff_a = radius_out*np.cos(ring_inclination)*y_star
    cond = coeff_a==0
    dists = np.empty((len(coeff_a),4),dtype=float)
    #Handler of special case
    #Here, we assume a ring exists:  0<radius_in*np.cos(ring_inclination) therefore cond is true iff y_star=0 iff the solutions for theta are 0 and pi.
    #Therefore the substituiton for t is ineffective, so we will use theta instead
    theta = np.empty((len(coeff_a[cond]),2),dtype=float)
    theta[:,0] = np.pi
    theta[:,1] = 0.0
    dists[cond,:2] = np.sqrt((my[cond,:2]-2*mrout[cond,:2]*np.cos(ring_inclination)*np.sin(theta))**2+(mx[cond,:2]-mrout[cond,:2]*np.cos(theta))**2)
    dists[cond,2:] = dists[cond,:2]
    #Quartic solver
    cond = np.logical_not(cond)
    coeff_b = 2*radius_out[cond]*x_star[cond]+2*radius_out[cond]**2*np.sin(ring_inclination)**2
    coeff_c = np.zeros_like(coeff_a[cond])
    coeff_d = 2*radius_out[cond]*x_star[cond]-2*radius_out[cond]**2*np.sin(ring_inclination)**2
    coeff_e = -coeff_a[cond]
    t = fastquartroots(coeff_a[cond].astype(complex),coeff_b.astype(complex),coeff_c.astype(complex),coeff_d.astype(complex),coeff_e.astype(complex)).T
    #Eliminate complex values
    t[np.iscomplex(t)] = np.nan
    t = t.real
    #Calculate distance at the extremum points
    dists[cond] = np.sqrt((my[cond]-2*mrout[cond]*np.cos(ring_inclination)*t/(1+t**2))**2+(mx[cond]-mrout[cond]*(1-t**2)/(1+t**2))**2)
    #Add the minimum and maximum to the interval split. (Ignoring NaNs)
    splitr[:,14] = np.nanmin(dists,-1)
    splitr[:,15] = np.nanmax(dists,-1)

    #Ensure all splitr values are on the interval [0,rcrit]
    #Check whether there are points beyond one - whether there is a interval [a,rcrit] which requires numerical integration
    splitr_decider = np.zeros_like(splitr[:,2:]).astype(np.bool)
    splitr_decider[np.isnan(splitr[:,2:])] = True
    splitr_decider[~np.isnan(splitr[:,2:])] = splitr[:,2:][~np.isnan(splitr[:,2:])] < rcrit
    todo_start = np.all(splitr_decider, -1)
    #todo_start = np.all(np.logical_or(splitr[:,2:]<rcrit,np.isnan(splitr[:,2:])),-1)
    splitr[np.isnan(splitr)] = rcrit
    splitr[splitr>1] = rcrit
    splitr[splitr<0] = rcrit
    #by default sort acts on the last axis
    splitt = np.sort((1-splitr**2)**(1/4))
    #Calculating Legendre roots for n inputs
    xcen, wcen = roots_legendre(n_center)
    xgress, wgress = roots_legendre(n_gress)
    y = np.zeros_like(x_planet)
    #Calculate integral_swapped_start to initiate.
    #Use splitr instead of splitt in conditionals to avoid loss of accuracy
    start_ind = splitt.shape[1]-np.sum(splitr<rcrit,-1)
    y[todo_start] += integrated_swapped_start(splitt[todo_start,start_ind[todo_start]],c1,c2,c3,c4,c[todo_start],tcrit)
    #Test whether the star's center is NOT inside the planet or ring. Only then integral_swapped_start needs to be added.
    todo_end = np.logical_and(x_star**2+y_star**2>=radius_planet**2,np.logical_or(x_star**2+y_star**2/np.cos(ring_inclination)**2>=radius_out**2,x_star**2+y_star**2/np.cos(ring_inclination)**2<=radius_in**2))
    end_ind = np.sum(splitr>0.0,-1)-1
    y[todo_end] += integrated_swapped_end(splitt[todo_end,end_ind[todo_end]],c1,c2,c3,c4)
    for k in range(15):
        a = splitt[:,k]
        b = splitt[:,k+1]
        #Calculate the integral only when necessary: a,b are different and it is not solved via analytic method.
        i = np.logical_and(np.not_equal(a,b),np.logical_and(k>=start_ind-1+todo_start,k<=end_ind-todo_end))
        if np.any(i):
            icen = np.logical_and(i,todo_start)
            igress = np.logical_and(i,np.logical_not(todo_start))
            ficen = vec_fixed_quad(xcen,wcen,n_center,a[icen],b[icen],radius_planet[icen],radius_in[icen],radius_out[icen],ring_inclination,ring_rotation,opacity,x_planet[icen],y_planet[icen],c1,c2,c3,c4,tol)
            y[icen] += np.dot(ficen,wcen)*0.5*(b[icen]-a[icen])
            figress = vec_fixed_quad(xgress,wgress,n_gress,a[igress],b[igress],radius_planet[igress],radius_in[igress],radius_out[igress],ring_inclination,ring_rotation,opacity,x_planet[igress],y_planet[igress],c1,c2,c3,c4,tol)
            y[igress] += np.dot(figress,wgress)*0.5*(b[igress]-a[igress])
            
    return y
