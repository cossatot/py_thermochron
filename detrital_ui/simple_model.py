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




idx, sim_idx, zprime, sim_zprime = [], [], [], []
exp, obs, sim = [], [], []

for sample in settings.samples:
    #import ipdb;ipdb.set_trace()
    hc[sample.tc_type].plot=True

    parms = e+[hc[sample.tc_type]]+abr

    if hasattr(sample, 'xyz'): #Bedrock sample
        z = sample.xyz[:,2] - sample.ref_z  #Warning: ignoring ref_z
        idx.append(range(len(z)))
        
        #Store expected ages for plotting
        #This is slower that computing expected profiles from parms posterior.
        #exp_profile = pm.Lambda('ExpProfile_%s' % sample.name,
        #                        lambda parms=parms:
        #                            ba.h2a(sample.get_zrange(), parms),
        #                        plot=False)
        #nodes.append(exp_profile)


    else: #Detrital sample
        if BIN:
            z   = sample.catchment.bins['h']-sample.ref_z
            p_samp = sample.catchment.bins['w']
        else:
            z = sample.catchment.xyz[:,2]-sample.ref_z
            p_samp = np.ones(len(z))/len(z) #Uniform

        idx.append(pm.Categorical("Idx_" + sample.name,
                                p = p_samp, size=len(sample.ages), 
                                plot=False, trace=False))
        sim_idx.append(pm.Categorical("SimIdx_" + sample.name,
                                p = p_samp, size=len(sample.ages), 
                                plot=False, trace=False))
        sim_zprime.append(pm.Lambda('SimZ_'  + sample.name, lambda z=z, i=sim_idx[-1]: z[i], 
                                    plot=False, trace=True))
                                    


    # This is a leftover from the tilt model. 
    zprime.append(pm.Lambda('Z_' + sample.name, lambda z=z, i=idx[-1]: z[i],
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







