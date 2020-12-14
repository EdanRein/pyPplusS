#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 16:04:06 2017
Object Oriented Double Hidden Area Algorithm
@author: edanrein
Implements methods from SectorAlgorithm to calculate double hidden area.

v10-Oct-2018
-Cleaned up

v02-Aug-2018
-Cleaned up

v26-May-2018
-Modified the point attribute to include two points
-Implemented use of two test points on the border of each curve instead of one
"""
from pyppluss.base_functions import get_coord_circle, get_coord_ellipse, get_arg, sector_area_disk, sector_area_planet, sector_area_star, get_dha_border, triangle_area
import math as M
import numpy as np
from pyppluss.base_functions import simpledist as edist

class Curve:
    """Parent class for all curves, with a few common methods.
    The Curves commonly implement a few methods:
        (a) Segment area calculation
        (b) Parameter storage
        (c) Parameter limitation (apply a logical index to the parameters)
    
    With regards to (c), the original parameter will be called i<param_name>.
    The limited parameter is called just <param_name>
        
    """    
    def coords(self, arg):
        raise NotImplementedError("overloaded by subclasses")
    
    def angle(self, k):
        """Looks up the angle of a point in input table, self.angles
        
        Parameters
        ----------
        k : ``int``
            Which point to examine
        """
        if np.isscalar(k):
            return self.angles[:,k]
        else:
            return self.angles[np.arange(self.angles.shape[0]),k]
    
    def sector_area(self, ang1, ang2):
        raise NotImplementedError("overloaded by subclasses")
        
    def segment_area(self, ang1, ang2):
        """Calculates segment area
        
        """
        #Calculate sector area using method from subcalss
        area_sec = self.sector_area(ang1,ang2)
        #Calculate triangle area
        p1 = self.coords(ang1)
        p2 = self.coords(ang2)
        area_tri = triangle_area(self.center[0], self.center[1], p1[0],p1[1], p2[0], p2[1])
        #Calculate segment, combining sector and triangle
        area = area_sec+area_tri*np.sign(ang2-ang1)*np.sign(abs(ang2-ang1)-M.pi)
        return area
            
    
    def gen_test(self, ang1, ang2):
        """Generate a test point, counterclockwise direction
        
        """
        test_arg = 0.5*(ang1+ang2)
        test_arg[ang1>ang2] += M.pi 
        return self.coords(test_arg)
    
    def is_on(self, ind):
        """Check using input self.pinds
        self.pinds has inedecies of all points from border on the curve.
        
        Parameters
        ----------
        ind: ``int``
            Which point to examine
        """
        if np.isscalar(ind):
            return self.pinds[:,ind]
        else:
            return self.pinds[np.arange(self.pinds.shape[0]),ind]
    
    def set_limiter(self,cond):
        raise NotImplementedError("overloaded by subclasses")
    
class Star(Curve):
    """
    
    """
    def __init__(self, center, inter_angles, on_inds):
        """Initialize with relevant parameters
        
        """
        self.icenter = center
        self.center = center
        self.iangles = inter_angles
        self.angles = inter_angles
        self.ipinds = on_inds
        self.pinds = on_inds
        self.ipoint = np.vstack((center,center)) + [[0],[1.],[0],[-1.]]
        self.point = self.ipoint
        self.iarea = np.pi*np.ones_like(center[0])
        self.area = self.iarea
    
    def coords(self, arg):
        """
        
        """
        return get_coord_circle(arg, self.center[0], self.center[1], 1)
    
    def sector_area(self, arg1, arg2):
        """
        
        """
        return sector_area_star(arg1,arg2)
    
    def is_in(self, point, tol=10**-8):        
        return edist(point, self.center)<=1+tol    

    
    def set_limiter(self,cond):
        """Required for vector handling. Allows conditionals.
        """
        self.center = self.icenter[:,cond]
        self.angles = self.iangles[cond]
        self.pinds = self.ipinds[cond]
        self.point = self.ipoint[:,cond]
        self.area = self.iarea[cond]

class Planet(Curve):
    """
    
    """
    def __init__(self, radius, inter_angles, on_inds):
        """Initialize this
        
        """
        self.radius = radius
        self.angles = inter_angles
        self.center = np.zeros((2,inter_angles.shape[0]))
        self.iradius = radius
        self.iangles = inter_angles
        self.icenter = self.center
        self.ipinds = on_inds
        self.pinds = on_inds
        self.ipoint = np.stack((np.zeros_like(radius),radius,np.zeros_like(radius),-radius),1).T
        self.point = self.ipoint
        self.iarea = np.pi*radius**2
        self.area=self.iarea
    
    def coords(self, arg):
        """
        
        """
        return get_coord_circle(arg, 0, 0, self.radius)
    
    def sector_area(self, arg1, arg2):
        """
        
        """
        return sector_area_planet(arg1, arg2, self.radius)
    
    def is_in(self, point, tol=10**-8):
        return edist(point, self.center)<=self.radius+tol
    
    def set_limiter(self,cond):
        """Required for vector handling. Allows conditionals.
        """
        self.center = self.icenter[:,cond]
        self.angles = self.iangles[cond]
        self.radius = self.iradius[cond]
        self.pinds = self.ipinds[cond]
        self.area = self.iarea[cond]
        self.point = self.ipoint[:,cond]
        
class Disk(Curve):
    """
    
    """
    def __init__(self, radius, inclination, inter_angles, on_inds):
        """Initialize this
        
        """
        self.radius = radius
        self.inclination = inclination
        self.angles = inter_angles
        self.center = np.zeros((2,inter_angles.shape[0]))
        self.iradius = radius
        self.iangles = inter_angles
        self.icenter = self.center
        self.foci = self.center.copy()
        self.foci[0] = self.radius*M.sin(self.inclination)
        self.ifoci = self.foci#.copy()
        self.ipinds = on_inds
        self.pinds = on_inds
        self.ipoint = np.stack((radius,np.zeros_like(radius),-radius,np.zeros_like(radius)),1).T
        self.point = self.ipoint
        self.iarea = np.pi*radius**2*M.cos(self.inclination)
        self.area=self.iarea

    def coords(self, arg):
        """
        
        """
        
        return get_coord_ellipse(arg, 0, 0, self.radius, self.radius*M.cos(self.inclination))
    
    def sector_area(self, arg1, arg2):
        """
        
        """
        return sector_area_disk(arg1, arg2, self.radius, self.inclination)
    
    def is_in(self, point, tol=10**-8):
        return edist(point, self.foci)+edist(point, -self.foci)<=2*self.radius+tol
    
    def set_limiter(self,cond):
        """Required for vector handling. Allows conditionals.
        """
        self.center = self.icenter[:,cond]
        self.angles = self.iangles[cond]
        self.radius = self.iradius[cond]
        self.foci = self.ifoci[:,cond]
        self.pinds = self.ipinds[cond]
        self.area = self.iarea[cond]
        self.point = self.ipoint[:,cond]

def intersection_area (curves,border, tol=10**-10):
    """Multiple hidden area calculations using object oriented, counterclockwise direction
    Assuming all border points are found before this.
    Assuming the same number of border points in all cases (since it is an array).
    In order to satisfy this, you can add NaNs to border and they will be ignored.
    
    Parameters
    ----------
    curves : ``list``
        A list of curve objects
    border : ``3-D array``
        Every split of the shape [k,:,:] is a single case, already oredered by angle relative to "center of mass".
    tol : ``float``
        Tolerance parameter
    
    Returns
    -------
    double_area : ``1-D array``
        Intersection area of curves.
    """
    seg_area = np.zeros(border.shape[0])
    poly_area = seg_area.copy()
    double_area = seg_area.copy()
    this_size = np.size(border,2)
    if not this_size in [2,3,4,5,6,7,8,9,10]:
        print(border)
        print(not list(border))
        print(this_size)
        raise Exception("Number of border points is not physically possible")
    maxn = border.shape[2]
    pnum = np.sum(np.logical_not(np.isnan(border[:,0,:])),-1)
    cases = border.shape[0]
    lastind = np.maximum(pnum-1,0)
    #Special case: When there are no intersections on the border of the double hidden area (Step 1.1.1)
    nodouble = lastind==0
    if np.any(nodouble):
        for c in curves:
            c.set_limiter(nodouble)
        for i in range(len(curves)):
            #When this curve's point is in all of the curves, set its area as the double-hidden area
            cond = np.empty((len(np.nonzero(nodouble)[0]),len(curves)),bool)
            for j in range(len(curves)):
                #Use Machine Precision as tolerance
                #Test only 1 point on the border. The use of machine precision reduces errors.
                #Unlike the DHA points, curves[i].point is known exactly (to machine precision) so there is no need for increased tolerance.
                #However, in certain cases there is a need for it since machine percision is relative error, i.e. if coordinates are ~200 instead of ~0.9 numerical errors occur.
                cond[:,j] = curves[j].is_in(curves[i].point[:2],tol=2.2204460492503131e-14)#16)
            condi = nodouble.copy()
            condi[nodouble] = np.all(cond,1)
            double_area[condi] = curves[i].area[np.all(cond,1)]
            #Otherwise, zero - double_area is initially set as zero and not changed.
    ydouble = np.logical_not(nodouble)
    poly_area = np.nansum(border[:,0,:-1]*border[:,1,1:]-border[:,1,:-1]*border[:,0,1:],-1)
    poly_area += border[np.arange(cases),0,lastind]*border[:,1,0]-border[np.arange(cases),1,lastind]*border[:,0,0]
    poly_area *= 0.5
    #Handling all-nans
    poly_area[np.isnan(poly_area)] = 0
    #The core loop of the algorithm
    #These two chunks of code are nearly identical, with one difference:
    #Since the top chunk only handles the 2-curve case, it needs one less conditional, checking whether an intersection point is on the curve.
    if len(curves)>2:
        for k in range(maxn):
            nind = k+np.ones(cases, dtype=int)
            nind[np.logical_or(k==maxn-1, k==lastind)] = 0
            is_done = np.isnan(border[:,0,k])
            #If there is no point to this, don't trigger the for loop at all
            if np.all(is_done):
                continue
            addarea = np.full((cases,len(curves)),np.inf)
            addarea[is_done] = 0
            for l in range(len(curves)):
                #Check whether the point and next_point are on curve
                #For increased efficiency, do not carry over all the vector, only a limited part.
                curve = curves[l]
                to_do = np.logical_not(is_done)
                curve.set_limiter(to_do)
                #We only consider cases where the curve connects the two points.
                to_curve = np.logical_and(curve.is_on(k), curve.is_on(nind[to_do]))
                if np.any(to_curve):
                    to_do[np.nonzero(to_do)[0][np.logical_not(to_curve)]] = False
                    curve.set_limiter(to_do)
                    ang = curve.angle(k)
                    next_ang = curve.angle(nind[to_do])
                    addarea[to_do,l] = curve.segment_area(ang, next_ang)
            seg_area += np.min(addarea,1)
    else:
        for k in range(maxn):
            nind = k+np.ones(cases, dtype=int)
            nind[np.logical_or(k==maxn-1, k==lastind)] = 0
            is_done = np.isnan(border[:,0,k])
            #If there is no point to this, don't trigger the for loop at all
            if np.all(is_done):
                continue
            addarea = np.full((cases,len(curves)),np.inf)
            addarea[is_done] = 0
            for l in range(len(curves)):
                #Check whether the point and next_point are on curve
                #For increased efficiency, do not carry over all the vector, only a limited part.
                curve = curves[l]
                to_do = np.logical_not(is_done)
                curve.set_limiter(to_do)
                #There are only 2 curves so all curves are on all points.
                ang = curve.angle(k)
                next_ang = curve.angle(nind[to_do])
                addarea[to_do,l] = curve.segment_area(ang, next_ang)
            seg_area += np.min(addarea,1)
    #Finalize the result.
    double_area[ydouble] = poly_area[ydouble]+seg_area[ydouble]
    return double_area

def border_sort(dha_border):
    """Sorting the border by angle relative to "center of mass"
    
    """
    maxn = dha_border.shape[2]
    pnum = np.sum(np.logical_not(np.isnan(dha_border[:,0,:])),-1)
    #If pnum=0, there are no points to begin with
    cond = np.logical_not(pnum==0)
    dborder = dha_border[cond]
    #Calculate the center of mass
    polyx = np.nansum(dborder[:,0,:], -1)/pnum[cond]
    polyy = np.nansum(dborder[:,1,:], -1)/pnum[cond]
    #Duplicate center of mass * (maximum number of points) for angle calculations
    xmat = np.repeat(polyx.reshape((-1,1)),maxn,1)
    ymat = np.repeat(polyy.reshape((-1,1)),maxn,1)
    #Sort the angles relative to the center of mass.
    angs = get_arg(dborder[:,0,:],dborder[:,1,:],xmat,ymat)
    inds = np.argsort(angs)
    dborder = np.moveaxis(dborder[np.arange(dborder.shape[0]),:,inds.T],0,2)
    dha_border[cond] = dborder
    return dha_border


def handler(radius_planet, disk_radius, disk_inclination, x_star, y_star, star_planet_intersections, star_disk_intersections, disk_planet_intersections, tol):
    """ Converts all the standard inputs into the appropriate classes (Star, Planet, Disk).
    """
    dha_border = get_dha_border(radius_planet, disk_radius, disk_inclination, x_star, y_star, star_planet_intersections, star_disk_intersections, disk_planet_intersections, tol)
    #Immediately re-order with respect to polygon center of mass.
    dha_border = border_sort(dha_border)
    indicators = dha_border[:,3:,:]
    is_star = (np.sum(indicators==0,1)>0)
    is_planet = (np.sum(indicators==1,1)>0)
    is_disk = (np.sum(indicators==2,1)>0)
    #Setup the curve instances
    sborder = np.swapaxes(dha_border,0,1).copy()
    pborder = sborder.copy()
    dborder = sborder.copy()
    sborder[:,np.logical_not(is_star)] = np.nan
    pborder[:,np.logical_not(is_planet)] = np.nan
    dborder[:,np.logical_not(is_disk)] = np.nan
    #Convert argument for star points to star argument
    sborder[2] = get_arg(sborder[0],sborder[1],np.repeat(x_star.reshape((-1,1)),10,1),np.repeat(y_star.reshape((-1,1)),10,1))
    star = Star(np.concatenate((x_star.reshape((1,-1)),y_star.reshape((1,-1)))), sborder[2], is_star)
    planet = Planet(radius_planet, pborder[2], is_planet)
    disk = Disk(disk_radius, disk_inclination, dborder[2], is_disk)
    
    return star, planet, disk, dha_border