"""
Utility functions for detrital modeling.
"""

__author__= ("Boris Avdeev, borisaqua@gmail.com", "Mike")


import pymc as pm
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
#import matplotlib.pyplot as plt
import sys, os, pickle


def sig2tau(sig):
    """Standard deviation to precision"""
    return 1./(sig**2)

def tau2sig(tau):
    """Precision to standard deviation"""
    return (1./tau)**.5




def nre_like(x, mu, err):
    '''Normal likelihood from relative error instead of tau'''
    tau = sig2tau(mu*err)
    #print x.shape, mu.shape, tau.shape
    #print; print
    return pm.normal_like(x, mu, tau)
def rnre(mu, err):
    '''Random normal from relative error instead of tau'''
    #if mu.shape[0]>1: size=
    return pm.rnormal(mu, sig2tau(mu*err))
NormRelErr = pm.stochastic_from_dist("NormRelErr", logp=nre_like, random=rnre,
                                            dtype=np.float,mv=False)







def sample_wr(population, size):
    """Chooses 'size' random elements (with replacement) from a population"""
    n = len(population)
    js = np.array(np.random.random(size) * n).astype('int') 
    return population[js]

def parm2ha(parm, ub, ub_type):
    """Convert parameter vector of an exhumation model [ e1, e2, e3, ... ,hc, abr1, abr2, ... ]
    into ([elevation], [age]) array for plotting"""
    parm = np.array([p.flatten()[0] for p in parm]) #Use of AdaptiveMetropolis results in inconsistent array dimentions
    #parm = np.array(parm).flatten()
    segs = len(parm)/2
    e = parm[0:segs]
    hc= -parm[segs]   # switch to depth from elevatioon
    a = [0]
    h = [hc]
    if segs > 1:
        a.extend(parm[segs+1:])
        da = np.diff(a)
        for i in range(segs-1):
            h.append(h[i] + e[i]*da[i])
    if not (ub_type=='h' or ub_type=='a'): 
        raise NameError('ub_type must be either a or h')
    if ub_type=='h' and ub > h[-1]:
        h.append(ub) 
        a.append(a[-1] + (ub - h[-2]) / e[-1])
    elif ub_type=='a' and ub > a[-1]:
        a.append(ub) 
        h.append(h[-1] + (ub - a[-2]) * e[-1])
    return np.array(h), np.array(a)




def h2a(hyps, parm):
    """Convert elevations to ages given parm = [ e1, e2, e3, ... ,hc, abr1, abr2, ... ] """
    #import pdb;pdb.set_trace()
    h,a = parm2ha(parm,max(hyps),'h')
    if len(a)==1: return np.repeat(np.nan,len(hyps)) 
    fun = interp1d(h, a, bounds_error=False, fill_value=np.nan) #or 0???
    return fun(hyps)

#def a2h(ages, parm):
    #"""Convert ages to elevations given parm = [ e1, e2, e3, ... ,hc, abr1, abr2, ... ] """
    #[h,a] = parm2ha(parm,max(ages),'a')
    #fun = interp1d(a, h, bounds_error=False, fill_value=0)
    #return fun(ages)
    



def tilt( xyz, strike, angle):
    """
    Converts elevations into tilted coordinates.
    The xref and yref are the coordinates for the rotation axis,
    should probably be near the center of the catchment.
    """
    if angle==0: return xyz[:,2]
    
    #import pdb;pdb.set_trace() 
    
    strike, angle = np.deg2rad(strike), np.deg2rad(angle)
    sina, ccosa = np.sin(angle), 1 - np.cos(angle)

    ux, uy, uz = np.sin(strike), np.cos(strike), 0

    # Rotation matrix 
    # http://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    rot = np.array([
        [ 1-ccosa+ux**2*ccosa, ux*uy*ccosa-uz*sina, ux*uz*ccosa+uy*sina ],
        [ ux*uy*ccosa+uz*sina, 1-ccosa+uy**2*ccosa, uy*uz*ccosa-ux*sina ],
        [ ux*uz*ccosa-uy*sina, uy*uz*ccosa+ux*sina, 1-ccosa+uz**2*ccosa ] 
        ]) 

    return np.dot( xyz, rot)[:,2]


def untilt(xyz, strike, angle):
    '''
    (-angle) because we rotate present topography into the past state,
    but angle is the direct rotation
    '''
    return tilt(xyz, strike, -angle)







#@pm.randomwrap
#def detrital_random(hyps,hyps_w,parm,err,size=None):
    #"""Random detrital sample. hyps -- bins. BAD? Should probably
       #supply weighting function and DEM and sample from there."""
    #if len(hyps_w)==1: idx=range(len(hyps))  #WARNING!
    #else: idx = hyps_w > pm.runiform(0,max(hyps_w),len(hyps_w))
    #ta = h2a(hyps[idx],parm)
    #ta_samp = sample_wr(ta, size)    
    #return pm.rnormal(ta_samp,sig2tau(err*ta_samp))

def statistics(mcmc, samples):
    """Output statistics for e1, e2, ..., hc"""
    f=open('statistics.csv', 'w')
    f.write("Parameter, Mean, Standard Deviation, 95% Lower, 95% Upper\n")
    for parm in mcmc.stochastics:
        if parm.plot:
            stats = parm.stats()
            f.write("%s, %f, %f, %f, %f\n" % (str(parm), stats['mean'], stats['standard deviation'], stats['95% HPD interval'][0], stats['95% HPD interval'][1]))
    f.close()






BINMODE = True #False

def autopickle(__init__):
    """
    Decorator for instantiating pickled instances transparently.
    Borrowed from code.activestate.com
    """

    def new__init__(self, *args, **kwds):
        picklename = self.getPickleFilename(*args, **kwds)
        if os.path.exists(picklename):
            print "Reading data from cache file " + picklename
            newSelf = pickle.load(open(picklename))
            #import ipdb; ipdb.set_trace()
            print "TODO: assert type(newSelf) is type(self)  #temp fix"
            # copy newSelf to self
            if hasattr(newSelf, '__getstate__'):
                state = newSelf.__getstate__()
            else:
                state = newSelf.__dict__
            if hasattr(self, '__setstate__'):
                self.__setstate__(state)
            else:
                self.__dict__.update(state)
        else:
            __init__(self, *args, **kwds)
            picklefile = open(picklename, BINMODE and 'wb' or 'w')
            try: pickle.dump(self, picklefile, BINMODE)
            finally: picklefile.close()
    return new__init__








### Convex hull code is from scipy examples

def _angle_to_point(point, centre):
    '''calculate angle in 2-D between points and x axis'''
    delta = point - centre
    res = np.arctan(delta[1] / delta[0])
    if delta[0] < 0:
        res += np.pi
    return res

def area_of_triangle(p1, p2, p3):
    '''calculate area of any triangle given co-ordinates of the corners'''
    return np.linalg.norm(np.cross((p2 - p1), (p3 - p1)))/2.

def convex_hull(points):
    '''Super slowly calculate subset of points that make a convex hull around points

Recursively eliminates points that lie inside two neighbouring points until only convex hull is remaining.

:Parameters:
    points : ndarray (2 x m)
        array of points for which to find hull

:Returns:
    hull_points : ndarray (2 x n)
        convex hull surrounding points
'''
    n_pts = points.shape[1]
    assert(n_pts > 5)
    centre = points.mean(1)
    angles = np.apply_along_axis(_angle_to_point, 0, points, centre)
    pts_ord = points[:,angles.argsort()]
    pts = [x[0] for x in zip(pts_ord.transpose())]
    prev_pts = len(pts) + 1
    k = 0
    while prev_pts > n_pts:
        prev_pts = n_pts
        n_pts = len(pts)
        i = -2
        while i < (n_pts - 2):
            Aij = area_of_triangle(centre, pts[i],     pts[(i + 1) % n_pts])
            Ajk = area_of_triangle(centre, pts[(i + 1) % n_pts], \
                                   pts[(i + 2) % n_pts])
            Aik = area_of_triangle(centre, pts[i],     pts[(i + 2) % n_pts])
            if Aij + Ajk < Aik:
                del pts[i+1]
            i += 1
            n_pts = len(pts)
        k += 1
    return np.asarray(pts)

def _point_in_poly(point, poly):
    

    x, y = point
    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside


def points_in_poly(points,poly):
    #import ipdb; ipdb.set_trace()
    idx = np.apply_along_axis(_point_in_poly,1,points,poly)
    return idx







def InitExhumation(settings):
    """Initialize piece-wise linear exhumation parameters"""
    #Create erosion rate parameters (e1, e2, ...)
    e = []
    for i in range(1,settings.breaks+2):
        e.append(pm.Uniform("e%i" % i, settings.erate_prior[0], settings.erate_prior[1],plot=True))
    #Create age break parameters (abr1, ...)
    abr_i = settings.abr_prior[0]
    abr = []
    for i in range(1,settings.breaks+1):
        abr_i = pm.Uniform("abr%i" % i, abr_i, settings.abr_prior[1],plot=True)
        abr.append(abr_i)
    hc = {}
    try:     hc['AFT'] = pm.Uniform('hc_AFT', settings.hc_AFT_prior[0], settings.hc_AFT_prior[1],plot=False)
    except:
        print "Automatic hc_AFT prior"
        hc['AFT'] = pm.TruncatedNormal('hc_AFT', mu=3.7, tau=1./0.8**2, b=6.0, a=2.9,plot=False)
    try:     hc['AHe'] = pm.Uniform('hc_AHe', settings.hc_AHe_prior[0], settings.hc_AHe_prior[1],plot=False)
    except:
        print "Automatic hc_AHe prior"
        hc['AHe'] = pm.TruncatedNormal('hc_AHe', mu=2.2, tau=1./0.5**2, b=3.7, a=1.6,plot=False)
#    hc={'AFT':pm.TruncatedNormal('hc_AFT', mu=3.7, tau=1./0.8**2, b=6.0, a=2.9,plot=False),
#        'AHe':pm.TruncatedNormal('hc_AHe', mu=2.2, tau=1./0.5**2, b=3.7, a=1.6,plot=False)}

    return e, hc, abr



