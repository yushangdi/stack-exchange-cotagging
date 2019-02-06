#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import errno
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


# ## Path Config

# In[2]:


# PATH to the source txt files
SOURCE_PATH_COTAG = "cotag_data/"
CSV_DIR = "results/"
CSV_NAME = CSV_DIR + "poly_params.csv"


# ## polynomial fit

# In[3]:


def fit_post_cotag(df_cotag):
    """
    csv_file is the file name, ends with .csv
    """
    
    ct = df_cotag["ct"]
    cotag_u = df_cotag["cotag_u"]
    # log to ensure positive 
    
    z,residuals, _, _, _ = np.polyfit(np.log(ct+1), np.log(cotag_u+1), 3, full=True)
    _,residuals_lin, _, _, _ = np.polyfit(np.log(ct+1), np.log(cotag_u+1), 1, full=True)
    p = np.poly1d(z)

    fitted = np.exp(p(np.log(ct+1)))
    return p, residuals[0]/len(cotag_u), residuals_lin[0]/len(cotag_u)


# In[4]:


try:
    os.makedirs(CSV_DIR)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

path = SOURCE_PATH_COTAG
final_df = pd.DataFrame(columns=['filename','d3','d2','d1','d0','mse'])
#ct = 0
for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
        if filename.endswith('_cotag.csv'): 
            #ct += 1
            #print(ct, filename)
            df = pd.read_csv(path + filename, index_col = 0)
            p, mse, mse_lin = fit_post_cotag(df)
            new_row = {'filename':filename[:-10],'d3':p[3],'d2':p[2],'d1':p[1],'d0':p[0], 'mse':mse, 'mse_lin':mse_lin}
            final_df = final_df.append(new_row, ignore_index=True)

final_df.to_csv(CSV_NAME)
print("done")


# # PCA

# In[9]:


NPY_FILENAME = CSV_DIR + 'param_PC.npy'


# In[10]:


pca = PCA(n_components=2)
principalComponents = pca.fit(final_df[['d3','d2','d1','d0']])
print(principalComponents.explained_variance_ratio_)
print("explains", np.sum(principalComponents.explained_variance_ratio_))
print("PC:\n",principalComponents.components_)


# In[11]:


PC1 = principalComponents.components_[0]
PC2 = principalComponents.components_[1]
PC_mean = principalComponents.mean_
np.save(NPY_FILENAME,{"PC1":PC1 , "PC2":PC2, "PC_mean": PC_mean, "explained":principalComponents.explained_variance_ratio_}) 

