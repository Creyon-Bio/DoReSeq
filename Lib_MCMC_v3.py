#!/usr/bin/env python
# coding: utf-8

# # Version 3 of the MCMC library

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import emcee
import time
from multiprocessing import Pool
import multiprocessing
from scipy.stats import nbinom
import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import pickle


# In[16]:


# Helper functions

# These helper functions are used for processing MCMC data
def sum_dims(data, dim):
    # this function sums the data on the dims specified in dims
    return(np.sum(data, axis=tuple(dim)))

def sum_dims_table(data):
    # make a table, for each row we sum over all dimensions except for the row index
    all_dims=np.arange(len(data.shape))

    return([sum_dims(data,np.delete(all_dims,d)) for d in all_dims])

def find_contour(hist, target):
    mx=np.max(hist)

    cf_low=0
    cf_high=mx

    val_low=1
    val_high=0

    while (cf_high-cf_low > 0.0000001) and (val_low!=val_high):
        cf_mid=0.5*(cf_low+cf_high)
        res=np.sum(hist[hist>cf_mid])
        if res>target:
            cf_low=cf_mid
            val_low=res
        else:
            cf_high=cf_mid
            val_high=res

    return(0.5*(cf_low+cf_high))

def analyze_MCMC(s1, binspec='Automatic'):
    ''' 
    This function analyzes the distribution of points in a MCMC sample
    The function works on data of any dimension, but has only been tested on 1D and 2D data
    The function 
        (a) histograms the data
        (b) computes the MAP estimate from the histogram
        (c) computes the 99%,95%, and 50% credibility regions
    
    Input:
        s1 -- numpy array of data
        binspec -- manually specify the bins for histogramming
        
    Output:
        mv                   -- mean values of the input data
        hist, bins           -- histogram of input data
        mapEstimate          -- map estimate derived from the histogram
        contours             -- heights at which to cut the histogram to obtain the 99%, 95%, and 50% credibility contours
        binSelect            -- bins that appear inside the contours
        credibilityIntervals -- min and max of the credibility region in each dimension
    '''

    # compute mean
    mv=np.mean(s1,axis=0).reshape(-1)
    #    print('mv=',mv)

    # compute histogram
    if binspec=='Automatic':
        hist, bins=np.histogramdd(s1,bins=51)
    else:
        hist, bins=np.histogramdd(s1,bins=binspec)
    hist=hist/np.sum(hist)

    # compute MAP estimate
    binMax=np.unravel_index(np.argmax(hist),hist.shape)
    #    print('binMax=',binMax)
    mapEstimate=[0.5*(bins[i][binMax[i]]+bins[i][binMax[i]+1]) for i in range(s1.shape[1])]
    #    print('mapEstimate=',mapEstimate)

    # compute credibility interval
    cuts=(0.99,0.95,0.5)
    contours=np.array([find_contour(hist,c) for c in cuts])

    binSelect=[hist > c for c in contours]        # select the bins that appear inside the contours

    # figure out the credibility intervals (as opposed to regions) for each dimension
    credibilityIntervals=[]
    for bs in binSelect:
        sdt=sum_dims_table(bs)
        credibilityIntervals.append([[np.min((bins[i][:-1])[sdt[i]>0]),np.max((bins[i][1:])[sdt[i]>0])] for i in range(len(sdt))])

    return mv, hist, bins, mapEstimate, contours, binSelect, credibilityIntervals


def estimate_p_value(hist, bins, threshold_value):
    # this function sums up the weight of the histogram over bins abover the threshold
    inds_of_interest = np.where(bins[0] >= threshold_value)
    p_value = np.sum(hist[inds_of_interest[0][:-1]])
    return p_value


# In[28]:


'''
DoReSeq class

The objective of this class is to evaluate the probability distribution over fit parameters of 
our noise model conditioned by the number of raw reads for the different samples

The class is designed to work on data for one gene at a time
Probability is evaluated using emcee MCMC package
The resultant MCMC sample is analyzed within the class
'''
class DoReSeq():
    def __init__(self, settings, 
                 scale=np.array([], dtype='float64'),
                 dose=np.array([], dtype='float64'),
                 time=np.array([], dtype='float64'),
                 plates=np.array([], dtype='int64')
                ):
        ''' 
        This function initializes the settings and the metadata
        
        Input: 
            settings -- dictionary of settings for the calculation
            scale    -- scale factor for each sample (i.e. the total number of non-duplicate reads)
            dose     -- dose for each sample
            time     -- time for each sample [only needs to be specified if 'use_time':True]
            plates   -- plate index for each sample [only needs to be specified if 'use_plate_dependent_tpm':True
        '''
        # settings
        self.settings=settings             # dictionary of settings used for calculation and analysis

        # variables that store the metadata 
        self.all_scale=scale               # scale factor for each sample
        self.all_dose=dose                 # dose for each sample
        self.all_time=time                 # time for each sample
        self.all_plates=plates             # plate index for each sample
        
        # variables used in calculations
        # __n_plates is used to determine the number of tpm fitting parameters
        if self.settings['use_plate_dependent_tpm']:  
            self.__n_plates=len(set(self.all_plates)) # To initialize __n_plates, we count the number of plates
        else:
            self.__n_plates=1                         # If plate-dependent tpm is not used

        
    def _log_posterior(self, theta, args):
        ''' 
        This function computes the posterior that is sampled by runMC
        
        Input: theta -- the list of fit parameters
        
        Output: floating point number specifying the log posterior
        '''
        all_data=args
        
        # decode the fit parameters
        tpm_list=theta[:self.__n_plates]
        phi=theta[self.__n_plates]
        kd=theta[self.__n_plates+1]
        ic50=theta[self.__n_plates+2]
        if self.settings['use_time']:
            delta=theta[self.__n_plates+3]
            
        # compute the log prior -- here we implement a top-hat prior for all fitting parameters
        for tpm_v in tpm_list: 
            if (tpm_v < 0.001) or (tpm_v > 50000): return -np.inf
        
        if (phi < 0.001) or (phi>10): return -np.inf
        if (kd < 0.001) or (kd>1): return -np.inf
        if (ic50 < 0.001) or (ic50>15): return -np.inf
    
        if self.settings['use_time']:
            if (delta < 0.04) or (delta>1): return -np.inf
        
        # compute the log likelihood        
        if self.settings['use_plate_dependent_tpm']:
            tpm_p=tpm_list[self.all_plates]    # if using plate-dependent tpm, construct a list where each sample is assigned its correct tpm
        else:
            tpm_p=tpm_list[0]                  # if not using plate dependent tpm, make a float64 variable equal to the tpm
       
        # if using time dependence compute the attenuation factor
        if self.settings['use_time']:
            af=1.-np.exp(-self.all_time*delta)
        else:
            af=1.
    
        # construct the expect mean number of counts
        mu_ = (self.all_scale) * (tpm_p*(1.E-6)) * (1-self.all_dose*(1.-kd)/(self.all_dose+ic50)*af)        
        mu_ = np.asarray(mu_)
   
        r1=nbinom._pmf(all_data, 1./phi, 1./(1.+mu_*phi))
    
        # if use_outliers is turned on modify the probability 
        if self.settings['use_outliers']:
            frac_out=self.settings['outlier_fraction']
            phi_out=self.settings['outlier_phi']
            r1=(1.-frac_out)*r1 + frac_out*nbinom._pmf(all_data, 1./phi_out, 1./(1.+mu_*phi_out))

        # compute the log of the likelihood
        res=0
        for an_r in r1:
            if an_r==0:
                res+=-1000.
            else:
                res+=np.log(an_r)

        return res

    

    def run_mc(self, all_data):
        ''' 
        This runs the MCMC calculation
        
        Input: all_data -- a numpy array of integers specifying the raw number of reads for each sample
        
        Output: the function returns a dictionary of analysis results as specified in the settings
        '''
            
        ndim = 3+self.__n_plates       # number of parameters in the model: [tpm_0, ... tpm_nplates, phi, kd, ic50]
        
        if self.settings['use_time']:  # add one more dimension if we need to fit delta
            ndim=ndim+1

        nwalkers = self.settings['mcmc_nwalkers']  # number of MCMC walkers
        nburn = self.settings['mcmc_nburn']   # "burn-in" period to let chains stabilize
        nsteps = self.settings['mcmc_nsteps']  # number of MCMC steps to take

        # set theta near the maximum likelihood for the SNCA gene 
        np.random.seed(0)
        if self.settings['use_time']:
            theta1=np.array([50]*self.__n_plates+[0.05, 0.35, 2., 0.5])
        else:
            theta1=np.array([50]*self.__n_plates+[0.05, 0.35, 2.])
            
        starting_guesses = np.array([np.random.normal(theta1, theta1/10) for i in range(nwalkers)])

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._log_posterior, args=[all_data])
        sampler.run_mcmc(starting_guesses, nsteps)

        sample = sampler.chain[:, nburn:, :].reshape(-1, ndim)     # extract the MCMC sample, dropping the burn-in period
        lnp=sampler.lnprobability[:, nburn:].reshape(-1)           # extract the log probability data, dropping the burn-in period
        pos_max=np.argmax(lnp)                                     # figure out the location of the sample with highest probability
        MAPestimate=sample[pos_max]                                # sample with highest probability = MAP estimate
        
        return_data={}

        if self.settings['return_MCMC_sample']:
            return_data['MCMC_sample']=sample

        if self.settings['return_MCMC_emcee_sampler']:
            return_data['MCMC_emcee_sampler']=sampler

        if self.settings['return_analysis']:
            return_data['analysis']=self.analyze_gene(MAPestimate, sample)

        return(return_data)
    
    
    def analyze_gene(self, MAPestimate, sample):
        ''' 
        This function analyzes the MCMC sample data
                
        Output: a python dictionary of analysis results 
        '''
        doses=self.settings['analysis_rfd_dose_list']  # change this number to get response at a different dose(s)
        step_rfd=0.01
 
        p_val_thresh = self.settings['analysis_rfd_p_val_threshold']

        hist_store={'MAP estimate':MAPestimate}
        
        # process the tpm data
        tpm_dict = {}
        for i in range(self.__n_plates):
            mv, hist, bins, mapEstimate, contours, binSelect, credibilityIntervals=analyze_MCMC(sample[:,i:i+1])
            tpm_dict[f'tpm_{i}'] = {
                'hist': hist,
                'bins': bins,
                'contours': contours,
                'mapEstimate': mapEstimate,
                'credibilityIntervals': credibilityIntervals[1]
            }
        hist_store['tpm'] = tpm_dict

        # process the phi data
        mv, hist, bins, mapEstimate, contours, binSelect, credibilityIntervals=analyze_MCMC(sample[:,self.__n_plates:self.__n_plates+1])
        hist_store['phi'] = {
            'hist': hist,
            'bins': bins,
            'contours': contours,
            'mapEstimate': mapEstimate,
            'credibilityIntervals': credibilityIntervals[1]
        }

        # process the kd-ic50 data
        mv, hist, bins, mapEstimate, contours, binSelect, credibilityIntervals=analyze_MCMC(sample[:,self.__n_plates+1:self.__n_plates+1+2], binspec=[np.arange(0,1.0001,1/50),np.arange(0,15.0001,15/50)])
        estimated_p_kd = estimate_p_value(hist[0], bins, p_val_thresh)
        hist_store['kd_ic50'] = {
            'hist': hist,
            'bins': bins,
            'contours': contours,
            'mapEstimate': mapEstimate,
            'credibilityIntervals': credibilityIntervals[1],
            'estimated_p_val_kd': estimated_p_kd
        }
        
        # process the delta data if available
        if self.settings['use_time']:
            mv, hist, bins, mapEstimate, contours, binSelect, credibilityIntervals=analyze_MCMC(sample[:,self.__n_plates+3:self.__n_plates+4])
            hist_store['delta'] = {
                'hist': hist,
                'bins': bins,
                'contours': contours,
                'mapEstimate': mapEstimate,
                'credibilityIntervals': credibilityIntervals[1]
            }

        # process the response at fixed dose data
        rfd_dict = {}
        for dose in doses:
            sample_rfd=(sample[:, self.__n_plates+1]+(1.-sample[:, self.__n_plates+1])/(1.+dose/sample[:, self.__n_plates+2]))
            mv, hist, bins, mapEstimate, contours, binSelect, credibilityIntervals=analyze_MCMC(sample_rfd.reshape(-1,1), binspec=[np.arange(0,1.0001,1/50)])
            estimated_p = estimate_p_value(hist, bins, p_val_thresh)
            rfd_dict[f'rfd_{dose}'] = {
                'hist': hist,
                'bins': bins,
                'contours': contours,
                'mapEstimate': mapEstimate,
                'credibilityIntervals': credibilityIntervals[1],
                'estimated_p_val': estimated_p
            }
        hist_store['rfd'] = rfd_dict

        return(hist_store)

