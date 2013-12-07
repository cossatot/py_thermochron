import sys; sys.path[0:0] = './' # Puts current directory at the start of path
import settings


import pymc as pm
import numpy as np
import scipy as sp

import detrital_ui.common as ba
from detrital_ui.data_types import *

BIN = True




err = pm.Uniform('RelErr',settings.error_prior[0],settings.error_prior[1],plot=True)


e, hc, abr = ba.InitExhumation(settings)

idx, sim_idx = [], []
exp, obs, sim = [], [], []



for sample in settings.samples:
    hc[sample.tc_type].plot=True

    parms = e+[hc[sample.tc_type]]+abr

    if hasattr(sample, 'xyz'): #Bedrock sample
        hyps = sample.xyz[:,2] - sample.ref_z
        #idx.append(range(len(hyps)))
        #sim_idx.append(range(len(hyps)))
        
        #Store expected ages for plotting
        #This is slower that computing expected profiles from parms posterior.
        #exp_profile = pm.Lambda('ExpProfile_%s' % sample.name,
        #                        lambda parms=parms:
        #                            ba.h2a(sample.get_zrange(), parms),
        #                        plot=False)
        #nodes.append(exp_profile)
        exp.append(pm.Lambda("ExpAge_" + sample.name, 
            lambda parms=parms: ba.h2a(hyps, parms),
            plot=False))
        
        sim.append(pm.Lambda("SimAge_" + sample.name, 
            lambda p=parms, mu=exp[-1], re=err: ba.rnre(mu, re), 
            plot=False))


    else:#Detrital sample
        if BIN:
            hyps   = sample.catchment.bins['h']-sample.ref_z
            p_samp = sample.catchment.bins['w']
        else:
            hyps = sample.catchment.xyz[:,2]-sample.ref_z
            p_samp = np.ones(len(hyps))/len(hyps) #Uniform

        idx.append(pm.Categorical("Idx_" + sample.name,
                                p = p_samp, size=len(sample.ages), 
                                plot=False, trace=False))
        sim_idx.append(pm.Categorical("SimIdx_" + sample.name,
                                p = p_samp, size=len(sample.ages), 
                                plot=False, trace=False))

        exp.append(pm.Lambda("ExpAge_" + sample.name, 
            lambda i=idx[-1], parms=parms: ba.h2a(hyps[i], parms),
            plot=False))

        @pm.deterministic(name="SimAge_" + sample.name, plot=False)
        def sim_i(parms=parms, i=sim_idx[-1], re=err):
            mu = ba.h2a(hyps[i],parms)
            return ba.rnre(mu,re),
        
        sim.append(sim_i)
    #import ipdb;ipdb.set_trace()
    if sample.use: 
        obs.append(ba.NormRelErr("ObsAge_" + sample.name,
                            value = sample.ages, mu=exp[-1], err = err,
                            observed=True))
    




