# DoReSeq
Dose Response analysis using next generation Sequencing data 

Lib_MCMC_v3.ipynb contains the key functions for running Markov Chain Monte Carlo using the emcee package and analyzing the results
Lib_MCMC_v3.py is generated using jupyter from Lib_MCMC_v3.ipynb

Plots_for_paper.ipynb is provided as an example for how to run Lib_MCMC_v3 package, it generates all MCMC plots in the paper

To analyze data:
# 1 Make a DoReSeq object
drs=DoReSeq(settings,                               # dictionary of settings explained below
        scale=np.array(df_meta_select['scale']),    # array of scale parameter values, one for each sample
        dose=np.array(df_meta_select['dose']),      # array of dose values, one for each sample
        time=np.array(df_meta_select['time']))      # optional: array of time values, one for each sample
        
# 2 Make an array of raw number of reads
args=np.array(df_raw_select)                        # array of the raw number of reads, rows are genes and columns are samples

# 3 Run DoReSeq on each row of the array
pool = multiprocessing.Pool(n_cpu)                  # process data using multiprocessing
res = list(tqdm.tqdm(pool.imap(drs.run_mc, args), total=len(args))) # res is a list of dictionaries with results of DoReSeq analysis

# settings dictionary
'use_plate_dependent_tpm': True/False -- whether to use different zero dose tpm fit parameter for each plate 
'use_outliers': True/False -- whether to use a mixture of two negative binomial distributions to describe outliers as well as inliers 
'outlier_fraction': floating point number -- specifies the mixture fraction for the negative binomial districution of outliers (0.1 is a good value)
'outlier_phi': floating point number -- dispersion parameter for the negative binomial districution of outliers (5 is a good value)
'use_time': True/False -- whether to use time dependence and to dit delta parameter
'mcmc_nwalkers': integer -- parameter for running emcee package (40 is a good value)
'mcmc_nburn': integer -- parameter for running emcee package (1000 is a good value)
'mcmc_nsteps': integer -- parameter for running emcee package (5000 is a good value)
'return_MCMC_sample': True/False -- whether to return the MCMC sample
'return_MCMC_emcee_sampler':True/False -- whether to return the emcee sampler
'return_analysis': True/False -- whether to perform data analysis on the MCMC sample and return the results
'analysis_rfd_dose_list': list of floats -- dose values at which to compute the knockdown at fixed dose ([0.625, 1.25, 2.5, 5, 10] is a good value)
'analysis_rfd_p_val_threshold': floating point number -- threshold for computing p-values (0.8 is a good value)
