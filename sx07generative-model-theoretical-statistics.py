#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import errno
from scipy.stats import hypergeom
from scipy.special import comb


# ## Helper Functions

# In[2]:


def find_n(N,T):
    left = N
    right = N/(1-np.exp(-1))
    
    def search_n(l,r,T):
        f = lambda x: x*(1-np.exp(-T/x))
        while(f(r) < N):
            l = r
            r = r * 1.5
        n = (l+r)/2
        while(np.abs(f(n)-N) > 1):
            if f(n) - N < 0:
                l = n
                n = (l+r)/2
            else:
                r = n
                n = (l+r)/2
        return n
    return round(search_n(left,right,T)).astype(int)


def plot_CDF_helper(data):
    h, edges = np.histogram(data, density=True, bins=100,)
    h = np.cumsum(h)/np.cumsum(h).max()

    X = edges.repeat(2)[:-1]
    y = np.zeros_like(X)
    y[1:] = h.repeat(2)
    return X,y 


# In[3]:


def calculate_theory_unique_cotag(file, info):
    
    cotag_file = "./cotag_data/" + file + "_cotag.csv"
    N = int(info[info["filename"]==(file+".txt")]["num_post"])
    
    cotag_df = pd.read_csv(cotag_file, index_col = 0)
    ct_lst = []
    expected_lst = []
    for row in cotag_df.iterrows():
        x_i = row[1]["ct"]
        temp = 1- hypergeom(N, x_i,cotag_df["ct"]).pmf(0)
        #print(temp)
        result = np.sum(temp) - (1- hypergeom(N, x_i,x_i).pmf(0))
        ct_lst.append(x_i)
        expected_lst.append(result)
    
    result_df = pd.DataFrame(
        {'ct':       ct_lst,
         'expected': expected_lst
        })
    
    result_df.to_csv("./files/theory_unique/"+file+".csv")
    return result_df


# In[4]:


info = pd.read_csv("results/stats.csv", index_col = 0)


# In[5]:


try:
    os.makedirs("./files/theory_unique/")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
for filename in os.listdir("./cotag_data"):   

    if filename.endswith('.csv'): 
        print(filename)
        file = filename[:-10]
        df = calculate_theory_unique_cotag(file,info)


# In[6]:


def calculate_theory_mse(file):
    
    cotag_df = pd.read_csv("./cotag_data/" + file + "_cotag.csv").sort_values('ct').reset_index()
    result   = pd.read_csv("./files/theory_unique/"+file+".csv").sort_values('ct').reset_index()
    
 
    assert (cotag_df['ct'] - result["ct"] == 0).all()
    
    return (
           np.sum(-np.log(cotag_df['cotag_u']+1) + np.log(result['expected']+1))/len(cotag_df['cotag_u']),
           np.sum(-cotag_df['cotag_u'] + result['expected'])/len(cotag_df['cotag_u']))


# In[7]:


final_df = pd.DataFrame()
    
for filename in os.listdir("./cotag_data"):   

    if filename.endswith('.csv'): 
        #print(filename)
        file = filename[:-10]
        mre_log, mre = calculate_theory_mse(file)
        new_row = {"filename": file, "mre_log":mre_log,  'mre':mre}
        final_df = final_df.append(new_row, ignore_index=True)
final_df.to_csv("results/theory_cotag_u.csv")
print('done')


# ## Variance of Number of posts with 0 tag

# In[8]:


def get_var(N,M):
    p = (1-1/N)**M
    var = N * p * (1-p)
    return var


# In[9]:


info = pd.read_csv("results/stats.csv", index_col = 0)
info = info[["filename",'num_post', "num_post_cor", "total_tag"]]


# In[11]:


final_df = pd.DataFrame()
for row in info.iterrows():
    N = row[1]["num_post_cor"]
    M = row[1]["total_tag"]
    var = get_var(N,M)
    new_row = {"filename": row[1]["filename"], 'var':var, 'var_p':var/ row[1]["num_post_cor"], 'sqrt_p':np.sqrt(var)/ row[1]["num_post_cor"]}
    final_df = final_df.append(new_row, ignore_index=True)
final_df.to_csv("results/var_zero_post.csv")
print('done')

