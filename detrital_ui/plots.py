"""
Plotting functions for detrital modeling
"""

import matplotlib
matplotlib.use('Agg')
import pylab as plt

import pymc as pm
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
import sys
import common as ba
import mpl_toolkits.axes_grid as ag
from data_types import *


def ecdf_lines(observations, a=None, b=None, color='k',alpha=0.01):
    """Plots ecdf as a line"""
    def __ecdf(x):
        counter = 0.0
        for obs in observations:
            if obs <= x:
                counter += 1
        return counter / len(observations)

    if a == None: a = observations.min() - observations.std()/2
    if b == None: b = observations.max() + observations.std()/2
    X = np.linspace(a, b, 100)
    f = np.vectorize(__ecdf)
    plt.plot(X, f(X), color=color,alpha=alpha)
    
def ecdf_points(obs,hpd=None,color='w',edgecolors='k'):
    """Plots ecdf as a point"""
    obs=np.array(obs)
    idx = np.argsort(obs)
    cdf = np.linspace(0,1,len(obs))
    if hpd is None:
        plt.scatter(obs[idx],cdf,facecolors=color,edgecolors=edgecolors)
    else:
        plt.errorbar(obs[idx],cdf,xerr=np.abs(hpd[:,idx]-obs[idx]),fmt='.')

def gof(obs,sim,obs_hpd=None, obs_color='w',sim_color='k'):
    """Creates gof plot for detrital data"""
    for sim_i in sim:
        ecdf_lines(sim_i,color=sim_color)
    ecdf_points(obs,obs_hpd,color=obs_color,edgecolors=sim_color)
    
def ah(parms, h_max, color='black'):
    """Generates and plots age/elevation lines between 0 and h_max"""
    for parm in parms:
        h, a = ba.parm2ha(parm, h_max, 'h')
        plt.plot(a,h,color=color,alpha=0.01)

def ks_gof(mcmc, samples, format="png"):
    """Runs ks_2samp test and plots Observed/Expected vs. Simulated/Expected"""
    size = len(samples)
    #Check for bedrock data
    for sample in samples:
        if isinstance(sample, BedrockSample):
            size = len(samples)-1
    #Set size and create grid
    if size == 1:
        fig = plt.figure(figsize=(10, 10))
    else:
        fig = plt.figure(figsize=(6, 10))
    grid = ag.axes_grid.Grid(fig, 111, nrows_ncols = (size, 1), axes_pad = 1, share_x=False, share_y=False, label_mode = "all" )
    j = 0 
    for sample in samples:
        if isinstance(sample, DetritalSample):
            #obs = mcmc.get_node("ObsAge_" + sample.name)
            sim = mcmc.get_node("SimAge_" + sample.name)
            exp = mcmc.get_node("ExpAge_" + sample.name)
            d_simExp=[]
            d_obsExp=[]
            #import ipdb; ipdb.set_trace()
            for i in range(len(exp.trace()[:,-1])):
                D, P = sp.stats.ks_2samp(mcmc.trace(exp)[i,-1], mcmc.trace(sim)[i,-1])
                d_simExp.append(D)
                D, P = sp.stats.ks_2samp(mcmc.trace(exp)[i,-1], sample.ages)
                d_obsExp.append(D)
                
            #The test statistics generated from ks_2samp plot as a grid. The following adds
            #random uniform noise for a better visual representation on the plot.
            noise=(0.5/len(sample.ages))
            for i in range(len(d_simExp)):
                d_simExp[i] = d_simExp[i] + np.random.uniform(-noise, noise, 1)
                d_obsExp[i] = d_obsExp[i] + np.random.uniform(-noise, noise, 1)

            #Calculate p-value (The proportion of test statistics above the y=x line)
            count=0
            for i in range(len(d_simExp)):
                if (d_simExp[i]>d_obsExp[i]):
                    count=count+1
            count=float(count)
            p_value=(count/len(d_simExp))
    
            #Plot
            grid[j].scatter(d_obsExp, d_simExp, color='gray', edgecolors='black')
            grid[j].set_xlabel('Observed/Expected', fontsize='small')
            grid[j].set_ylabel('Simulated/Expected', fontsize='small')
            label = sample.name + ", " + "p-value="+"%f"%p_value
            grid[j].set_title(label, fontsize='medium')

            #Add a y=x line to plot
            if (max(d_simExp)>=max(d_obsExp)):
                grid[j].plot([0,max(d_simExp)],[0,max(d_simExp)], color='black')
            else:
                grid[j].plot([0,max(d_obsExp)],[0,max(d_obsExp)], color='black')
            j+=1
            
            
    plt.suptitle('Kolmogorov-Smirnov Statistics', fontsize='large')
    fig.savefig("KS_test."+format)

def get_map_idx(mcmc):
    return np.argmin(mcmc.trace('deviance')[:])

def summary(mcmc, samples, format='png'):
    fig = plt.figure(figsize=(6,4.5*len(samples)+1))

    had_br=False
    ax = None
    map_idx = get_map_idx(mcmc)
    for i,sample in enumerate(samples):
        #import ipdb;ipdb.set_trace()
        ax = plt.subplot(len(samples), 1, i+1, sharex=ax)
        plt.title(sample.name, fontsize='medium')
        sim = mcmc.trace("SimAge_"+sample.name)[:].squeeze()
        exp = mcmc.trace("ExpAge_"+sample.name)[:].squeeze()
        if hasattr(sample,'xyz'):  #bedrock sample
            try: z = mcmc.trace('Z_' + sample.name)[:].squeeze()
            except KeyError: z = sample.xyz[:,2]; print "Not a tilt model"
            had_br=True
            lowerElev = sample.ref_z - np.max(mcmc.trace("hc_%s"%sample.tc_type)[:])
            for i in range(len(exp)):
                if len(z.shape)>1: zi = z[i]; z_map=z[map_idx]
                else: zi = z; z_map=z
                plt.scatter(sim[i], zi, s=3, c='k', edgecolors=None, alpha=0.01)
                plt.plot(exp[i], zi, c='k', alpha=0.01)
                #plt.scatter(sample.ages, tilt_z_i, color='k', s='.',alpha=0.1)
            #for tr in mcmc.trace("ExpProfile_"+sample.name):
            #    plt.plot(tr, sample.get_zrange(), color='k',alpha=0.01)
            #Plot MAP estimate
            #parm_map=[mcmc.trace(p)[map_idx,] for p in parm_list]
            #h, a = ba.parm2ha(parm_map, upperElev, 'h')
            
            #plt.scatter(sample.ages, sample.z, color='white', edgecolors='black')
            #plt.plot(mcmc.trace("ExpProfile_"+sample.name)[map_idx], sample.get_zrange(), color='red')
            #import ipdb;ipdb.set_trace()
            plt.plot(exp[map_idx], z_map, color='red')
            plt.scatter(sample.ages, z_map, color='w', edgecolors='r') #plot observed age ~ map tilted elevation
            
            upperElev = 1.1 * max(sample.z)
            #plt.ylim(lowerElev,upperElev)
            plt.ylabel('Relative elevation (km)')
            

            
        else: #Detrital sample
            gof(sample.ages, sim)
            #ecdf_lines(mcmc.trace("ExpAge_"+sample.name)[map_idx],color='r',alpha=1)
            plt.ylim(0,1)
            plt.ylabel('Empirical CDF')
        if not had_br: # plot age-elev relationship
            pass
    
        plt.grid()
        
    plt.xlabel('Age (Ma)')
    plt.xlim(0, 1.1*np.max([np.max(s.ages) for s in samples]))
    fig.savefig("gof_summary." + format)
     
def histograms(samples, format="png"):
    """Plots histograms of age data and hypsometry data for each detrital sample"""
    size = len(samples)
    #Check for bedrock data
    for sample in samples:
        if isinstance(sample, BedrockSample): size -= 1
    if size == 0: return  # No detrital samples
    #Set size and create grid
    fig = plt.figure(figsize=(10, size*4+2))
    grid = ag.axes_grid.Grid(fig, 111, nrows_ncols = (size, 2), axes_pad = 1, share_x=False, share_y=False, label_mode = "all" )
    i = 0 
    for sample in samples:
        if isinstance(sample, DetritalSample):
            #Elevation
            grid[i].hist(sample.catchment.xyz[:,2], facecolor='gray')
            grid[i].set_xlabel(sample.name + ' Elevation (km)')
            #grid[i].set_ylabel('Frequency')
            i+=1
            
            #Ages
            grid[i].hist(sample.ages, facecolor='gray')
            grid[i].set_xlabel(sample.name + ' Age (Ma)')
            #grid[i].set_ylabel('Frequency')
            i+=1
            
    fig.savefig("Histograms." + format)
        
def chains(mcmc, iterations, samples, format="png"):
    """Plots the thinned chain and a histogram for each parameter"""
    fig = plt.figure(figsize=(15, 20))
    map_idx = get_map_idx(mcmc)
    
    N = len(mcmc.trace('deviance')[:])
    #Create a custom x-axis for the chains
    x=np.linspace(iterations/2,iterations,N)/1000.

    parms = []
    for st in mcmc.stochastics:
        if st.plot: parms.append(st)
        
	  
	
    grid = ag.axes_grid.Grid(fig, 111, nrows_ncols = (len(parms), 2), axes_pad = .3, share_all=False, label_mode = "L" )
    
    i=0
    for parm in parms:
        #import ipdb; ipdb.set_trace()
        grid[i].plot(x, parm.trace(), c='k')
        grid[i].set_ylabel(parm)
        grid[i].set_xlabel("1e3 Iteration")
        parm_map = parm.trace()[map_idx]
        grid[i].plot((x[0],max(x)),(parm_map, parm_map), c='r') #Adds MAP estimate
        i+=1
        
        
        grid[i].hist(mcmc.trace(parm)[:], orientation='horizontal', fc='gray')
        grid[i].plot((0,0.7*N),(parm_map, parm_map), c='r')  #Adds MAP estimate
        grid[i].set_xlabel("Frequency")

        #if isinstance(parm, pm.TruncatedNormal):
            ##print parm.__name__
            #x_ = np.linspace(parm.parents['a'], parm.parents['b'])
            #y_ = [parm.set_value(xi) or parm.logp for xi in x_]
            ##y_ = pm.truncnorm_like(x_, **parm.parents)
            ##import ipdb; ipdb.set_trace()
            #grid[i].scatter(y_,x_,c='k')


        i+=1
    
    plt.savefig("chains."+format)

def discrepancy(mcmc, samples, format="png"):    
    """Discrepancy plots for detrital and bedrock sample"""
    #Set size and create grid
    size = len(samples)
    if size == 1:
        fig = plt.figure(figsize=(6, 6))
    else:
        fig = plt.figure(figsize=(6, 10))
    grid = ag.axes_grid.Grid(fig, 111, nrows_ncols = (size, 1), axes_pad = 1, share_x=False, share_y=False, label_mode = "all" )
    j = 0 
    for sample in samples:
        name = "D_gof_"

        #obs = mcmc.get_node("ObsAge_" + sample.name)
        sim = mcmc.get_node("SimAge_" + sample.name)
        exp = mcmc.get_node("ExpAge_" + sample.name)
        #import ipdb; ipdb.set_trace()
        if len(sim.trace().shape) == 3: #Happens when using AdaptiveMetropolis
            D_obs, D_sim = pm.discrepancy(sample.ages, sim.trace()[:,:,-1], exp.trace()[:,:,-1])
        else:
            D_obs, D_sim = pm.discrepancy(sample.ages, sim, exp)

        #Calculate p-value (The proportion of test statistics above the y=x line)
        count=0
        for i in range(len(D_obs)):
            if (D_sim[i]>D_obs[i]):
                count=count+1
        count=float(count)
        p_value=(count/len(D_obs))

        #Plot
        grid[j].scatter(D_obs, D_sim, color='gray', edgecolors='black')
        grid[j].set_xlabel('Observed deviates', fontsize='small')
        grid[j].set_ylabel('Simulated deviates', fontsize='small')
        label = sample.name + ", " + "p-value="+"%f"%p_value
        grid[j].set_title(label, fontsize='medium')
            
        #Add a y=x line to plot
        if (max(D_sim)>=max(D_obs)):
            grid[j].plot([0,max(D_sim)],[0,max(D_sim)], color='black')
        else:
            grid[j].plot([0,max(D_obs)],[0,max(D_obs)], color='black')
        j+=1
       
    #plt.suptitle('Discrepancy Statistics', fontsize='large')
    fig.savefig("discrepancy."+format)


##def unorthodox_ks(mcmc, format="png"):
##    """Runs test similar to ks_2samp, but compares horizontal distance between CDF's"""
##    d_simExp=[]
##    d_obsExp=[]
##    sorted_obs=np.sort(mcmc.dt_obs.value)
##    for i in range(len(mcmc.dt_exp.trace())):
##
##        sorted_exp=np.sort(mcmc.dt_exp.trace()[i])
##        sorted_sim=np.sort(mcmc.dt_sim.trace()[i])
##
##        j=np.argmax(abs(sorted_exp-sorted_sim))
##        d_stat=abs(sorted_exp[j]-sorted_sim[j]) 
##        d_simExp.append(d_stat)
##        
##        j=np.argmax(abs(sorted_exp-sorted_obs))
##        d_stat=abs(sorted_exp[j]-sorted_obs[j]) 
##        d_obsExp.append(d_stat)
##    
##    fig = plt.figure()
##    plt.scatter(d_simExp, d_obsExp)
##    fig.savefig("Alternate_ks."+format)



def data_summary(settings):  
    fig=plt.figure()
    pl = plt.subplot(122)
    for v in settings.__dict__.values():
        if isinstance(v,Catchment):
            v.plot()
    for sample in settings.samples:
        sample.plot()

    pl.grid()
    pl.set_xlabel("x, km")
    pl.set_ylabel("y, km")
    
    pl.yaxis.tick_right()
    pl.yaxis.set_label_position("right")


    #pr = plt.subplot(122)
    i=0
    pi=None
    for sample in settings.samples:
        i+=1
        pi = plt.subplot(len(settings.samples),2,i*2-1,sharex=pi,title=sample.name)
        if isinstance(sample,DetritalSample):
            pi.hist(sample.ages, facecolor='gray')
            pi.set_ylabel("Frequency")
        elif isinstance(sample,BedrockSample):
            pi.scatter(sample.ages,sample.xyz[:,2],facecolors='gray')
            pi.set_ylabel("Elevation (km)")
        pi.grid()
    
    
    #import ipdb;ipdb.set_trace()
    pi.set_xlabel('Age (My)')
    plt.subplots_adjust(hspace=0.3)
    #plt.tight_layout()
    
    fig.savefig('data_summary.'+settings.output_format)
        



def tilt(mcmc,color='k',map_color='r'):
    '''Make polar plot of strike and angle of rotation'''
    #ax = plt.axes(projection='polar')
    ax=plt.gca()
    strike = np.deg2rad(mcmc.trace('Strike')[:])

    angle = mcmc.trace('Angle')[:]

    strike[angle<0]+=np.pi
    angle=np.abs(angle)
    strike[strike<0]+=2*np.pi
    strike = 2*np.pi - strike
    angle=90-angle
    ax.scatter(strike,angle,color=color,alpha=.01)
    map_idx = get_map_idx(mcmc)
    ax.scatter(strike[map_idx],angle[map_idx],edgecolors=map_color,facecolors='w')
    ax.set_rmax(90)
    ax.yaxis.set_ticklabels([])
    ax.xaxis.set_ticklabels([90,45,0,315,270,225,180,135])
    plt.grid(True)

    
    
    