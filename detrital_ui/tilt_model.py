import sys; sys.path[0:0] = './' # Puts current directory at the start of path
import settings


import pymc as pm
import numpy as np
import scipy as sp

import detrital_ui.common as ba
from detrital_ui.data_types import *

CATEGORICAL=True    # Setting to False is probably a bad idea because it will cause
                    # a Deterministic node TiltSample behave like a Stochastic


def _untilt_sample(x,i,s,a):
    if isinstance(i, int):  # make random sample here (this might be a bad idea)
        i = pm.rcategorical(np.ones(x.shape[0])/x.shape[0],size=i)
    return ba.untilt(x[i,:],s,a)
 
class TiltSample(pm.Deterministic):
    """
    z_prime = Tilt(name, xyz, idx, strike, angle[, doc, dtype=None,
        trace=True, cache_depth=2, plot=None])

    A Deterministic returning elevations in pre-tilted coordinates.

   :Parameters:
      xyz : Nx3 array of points
      idx: random sample index, or length of a random sample
      strike : bearing of the rotation axis
      angle : rotation angle (corkscrew, from past to present)
    """
    def __init__(self, name, xyz, idx, strike, angle, doc = 'Elevations in tilted coordinates', *args, **kwds):
        pm.Deterministic.__init__(self,
                                eval=_untilt_sample,
                                doc=doc,
                                name = name,
                                parents = {'x':xyz, 'i':idx, 's':strike, 'a':angle}, 
                                *args, **kwds)








err = pm.Uniform('RelErr',settings.error_prior[0],settings.error_prior[1],plot=True)

strike = pm.Uniform('Strike',settings.axis_prior[0],settings.axis_prior[1],plot=True)
angle  = pm.Uniform('Angle',settings.tilt_prior[0],settings.tilt_prior[1],plot=True)


e, hc, abr = ba.InitExhumation(settings)

idx, sim_idx, zprime, sim_zprime = [], [], [], []
exp, obs, sim = [], [], []

for sample in settings.samples:
    #import ipdb;ipdb.set_trace()
    hc[sample.tc_type].plot=True

    parms = e+[hc[sample.tc_type]]+abr

    if hasattr(sample, 'xyz'): #Bedrock sample
        xyz = sample.xyz - settings.rot_center  #Warning: ignoring ref_z
        idx.append(range(xyz.shape[0]))
        
        #Store expected ages for plotting
        #This is slower that computing expected profiles from parms posterior.
        #exp_profile = pm.Lambda('ExpProfile_%s' % sample.name,
        #                        lambda parms=parms:
        #                            ba.h2a(sample.get_zrange(), parms),
        #                        plot=False)
        #nodes.append(exp_profile)


    else: #Detrital sample
        xyz = sample.catchment.xyz - settings.rot_center  #Warning: ignoring ref_z
        p_samp = np.ones(xyz.shape[0])/xyz.shape[0] #Uniform

        if CATEGORICAL:
            idx.append(pm.Categorical("Idx_" + sample.name,
                                    p = p_samp, size=len(sample.ages), 
                                    plot=False, trace=False))
            sim_idx.append(pm.Categorical("SimIdx_" + sample.name,
                                    p = p_samp, size=len(sample.ages), 
                                    plot=False, trace=False))
                                    
        else: idx.append(len(sample.ages)); sim_idx.append(len(sample.ages))

        sim_zprime.append(TiltSample('SimZ_'  + sample.name, xyz, sim_idx[-1], strike, angle, 
                                    plot=False, trace=True))



    zprime.append(TiltSample('Z_' + sample.name, xyz, idx[-1], strike, angle, 
                                    plot=False, trace=True))

    if hasattr(sample, 'xyz'): sim_zprime.append(zprime[-1])
        



    exp.append(pm.Lambda("ExpAge_" + sample.name, 
        lambda z=zprime[-1], parms=parms: ba.h2a(z, parms),
        plot=False))

    
    
    if sample.use: 
        obs.append(ba.NormRelErr("ObsAge_" + sample.name,
                            value = sample.ages, mu=exp[-1], err = err,
                            observed=True))
    
#    import ipdb;ipdb.set_trace()
    @pm.deterministic(name="SimAge_" + sample.name, plot=False)
    def sim_i(parms=parms, zprime=sim_zprime[-1], err=err):
        exp = ba.h2a(zprime,parms)
        return pm.rnormal(mu = exp, tau = ba.sig2tau(exp*err)),
    
    sim.append(sim_i)
