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


# ## polynomial fit

# In[2]:


def fit_post_cotag(df_cotag, path, csv_file = None):
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
    

    if len(residuals) == 0:
        return p, None, None
    return p, residuals[0]/len(cotag_u), residuals_lin[0]/len(cotag_u)


# In[7]:


path = './files/generated/cotag_files/'
final_df = pd.DataFrame(columns=['filename','d3','d2','d1','d0','mse','mse_lin'])
#ct = 0
for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
        if filename.endswith('_cotag.csv'): 
            #ct += 1
            #print(ct, filename)
            df = pd.read_csv(path + filename, index_col = 0)
            p, mse, mse_lin = fit_post_cotag(df, path="files/gen_poly/",csv_file=filename[:-9])
            if mse is None:
                print(filename)
            new_row = {'filename':filename[:-10],'d3':p[3],'d2':p[2],'d1':p[1],'d0':p[0], 'mse':mse, 'mse_lin':mse_lin}
            final_df = final_df.append(new_row, ignore_index=True)

final_df.to_csv("./gen_results/gen_poly_params.csv")
print("done")


# # PCA

# In[5]:


pca = PCA(n_components=2)
principalComponents = pca.fit(final_df[['d3','d2','d1','d0']])
print(principalComponents.explained_variance_ratio_)
print("explains", np.sum(principalComponents.explained_variance_ratio_))
print("PC:\n",principalComponents.components_)


# In[6]:


PC1 = principalComponents.components_[0]
PC2 = principalComponents.components_[1]
PC_mean = principalComponents.mean_
np.save('gen_results/gen_param_PC.npy',{"PC1":PC1 , "PC2":PC2, "PC_mean": PC_mean, "explained":principalComponents.explained_variance_ratio_}) 


# In[ ]:




