#!/usr/bin/python


import matplotlib
matplotlib.use('Agg')
import pylab as plt


import sys
import common as ba
import plots 
import pymc as pm
import numpy as np
import scipy as sp

# TODO: Clean up
                   
def run_MCMC(model):
    """Run MCMC algorithm"""
    burn = model.settings.iterations/2
    thin = (model.settings.iterations-burn) / model.settings.finalChainSize
    name = "%s" % model.settings.model_name + "_%ibrk" % model.settings.breaks
    attempt = 0
#    model=None
#    while attempt<5000:
#        try:
#            model = ExhumationModel(settings)
#            break
#        except pm.ZeroProbability, ValueError:
#            attempt+=1
#            print "Init failure %i" % attempt

    db = pm.database.hdf5.load(name + '.hdf5')
    mcmc = pm.MCMC(model, db=db, name=name)    
    
    pm.graph.graph(mcmc, 'png', consts=False, legend=False,collapse_deterministics=False)

    corellated = {mcmc.get_node('Strike'):10, #WARNING!
                  mcmc.get_node('Angle'):1,
                  mcmc.get_node('hc_AHe'):1,
                  mcmc.get_node('hc_AFT'):1,
                  mcmc.get_node('e1'):0.02,
                  mcmc.get_node('e2'):0.02,
                  mcmc.get_node('abr1'):5}

    try: corellated.pop(None) # Remove inexisting nodes. Is this a reliable construct?
    except: pass #this will happen when all corellated nodes are found in the model (no Nones)
    
    mcmc.use_step_method(pm.AdaptiveMetropolis, corellated.keys(),
                         scales = corellated, interval=model.settings.iterations/10)  

    
    #import ipdb;ipdb.set_trace()

    # Make sure there are at least two chains
    while True:
        mcmc.sample(model.settings.iterations,burn=burn,thin=thin)
        if mcmc.db.chains > 1: break

    return mcmc
                    
if __name__ == '__main__':
    sys.path[0:0] = './' # Puts current directory at the start of path
    import model
    
    if len(sys.argv)>1: model.settings.iterations = int(sys.argv[1])
    
    plots.data_summary(model.settings)
    
    M=run_MCMC(model)
    
    
    try: pm.Matplot.summary_plot(M)
    except: print "Cannot plot auto summary"
   
    try:
        ba.statistics(M, model.settings.samples)
    except TypeError:
        print "Cannot compute stats without resampling (PyMC bug?)."
    plots.chains(M, model.settings.iterations, model.settings.samples, model.settings.output_format)
    plots.summary(M, model.settings.samples, model.settings.output_format)
    #plots.ks_gof(M, ms.samples, ms.output_format)
#    plots.histograms(ms.samples, ms.output_format)    
    
    try:
        plt.figure()
        plt.axes(polar=True)
        plots.tilt(M)
        plt.savefig('tilt.'+model.settings.output_format)
    except: print "No tilt plot."
#    plots.discrepancy(M, ms.samples, ms.output_format)
    
##    ps.unorthodox_ks(M, ms.output_format)

    #M.db.close()


    
