#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np
import powerlaw
import warnings
warnings.filterwarnings("ignore")

# In[2]:


paths = (["./parsed/", "./cotag_data/"])


# In[3]:


def lognorm_fit(filename, df_cotag, xmin = 1):
    data = df_cotag["ct"]

    fit = powerlaw.Fit(data, discrete=True, estimate_discrete=True, xmin=xmin)
    new_row = {'filename':filename,
                   'mu': fit.lognormal.mu, 
                   'sigma':fit.lognormal.sigma, 
                   'ks':fit.lognormal.D}
    R,p = fit.distribution_compare('lognormal', 'power_law', normalized_ratio=True)
    new_row['pl_R'] = R
    new_row['pl_p'] = p
    new_row['pl_ks'] = fit.power_law.D

    R,p = fit.distribution_compare('lognormal', 'truncated_power_law', normalized_ratio=True)
    new_row['tpl_R'] = R
    new_row['tpl_p'] = p
    new_row['tpl_ks'] = fit.truncated_power_law.D
    

    R,p = fit.distribution_compare('lognormal', 'stretched_exponential', normalized_ratio=True)
    new_row['sexp_R'] = R
    new_row['sexp_p'] = p
    new_row['sexp_ks'] = fit.stretched_exponential.D
    
    return new_row


# In[4]:


def calculate_stats(paths, save_file = "./results/tagfreq.csv"):
    SOURCE_PATH = paths[0]
    SOURCE_PATH_COTAG = paths[1]
    final_df = pd.DataFrame(columns=['filename','mu', 'sigma', 'ks'])
    
    ct = 0
    for (dirpath, dirnames, filenames) in os.walk(SOURCE_PATH):
        for filename in filenames:
            if filename.endswith('.txt'): 
                ct += 1
                #print(ct, filename)
                
                df_cotag = pd.read_csv(SOURCE_PATH_COTAG+"%s_cotag.csv" % filename[:-4], sep = ',', index_col = 0)
                new_row = lognorm_fit(filename[:-4], df_cotag)


                final_df = final_df.append(new_row, ignore_index=True)
                final_df.to_csv(save_file)
    final_df['pl_pm'] =  final_df['pl_p']
    final_df.loc[final_df['pl_pm']<1e-16,['pl_pm']] = 1e-16

    final_df['tpl_pm'] =  final_df['tpl_p']
    final_df.loc[final_df['tpl_pm']<1e-16,['tpl_pm']] = 1e-16

    final_df['sexp_pm'] =  final_df['sexp_p']
    final_df.loc[final_df['sexp_pm']<1e-16,['sexp_pm']] = 1e-16
    final_df.to_csv(save_file)
    print('done')
    return final_df


# In[5]:


final_df = calculate_stats(paths)


# ## Patent is different from other datasets

# In[6]:


def calculate_stats2(paths, save_file = "./results/patents_is_different.csv"):
    SOURCE_PATH = paths[0]
    SOURCE_PATH_COTAG = paths[1]
    final_df = pd.DataFrame()
    
    ct = 0
    for (dirpath, dirnames, filenames) in os.walk(SOURCE_PATH):
        for filename in filenames:
            if filename.endswith('.txt'): 
                ct += 1
                #print(ct, filename)
                
                df_cotag = pd.read_csv(SOURCE_PATH_COTAG+"%s_cotag.csv" % filename[:-4], sep = ',', index_col = 0)
                ct = df_cotag['ct']
                
                ratio = (np.sum(df_cotag['ct']==1)/np.sum(df_cotag['ct']==2))

                new_row = {"filename":filename[:-4], 'ratio':ratio}
                final_df = final_df.append(new_row, ignore_index=True)
                final_df.to_csv(save_file)

    print('done')
    return final_df


# In[7]:


final_df = calculate_stats2(paths)

