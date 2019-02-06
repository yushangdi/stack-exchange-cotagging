#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import scipy.stats as st
import os
import errno
from helpers import *


# ## Path Config

# In[18]:


# PATH to the source txt files (for generated)
SOURCE_PATH = "./files/generated/"
SOURCE_PATH_COTAG = "./files/generated/cotag_files/"
CSV_NAME = 'gen_results/gen_stats.csv'


# ## Read in Data

# In[19]:


def import_data(filename):
    
    """
    import dataset corresponding to the dataset defined
    return dataframe with columns `"ques", "tag", "id"`
    
    """

    df = pd.read_csv(filename, sep = ',', index_col = 0)    
    return df


# In[20]:


"""
Return the number of posts, tags and unique tags
"""
def get_data_stats(df):
    num_post = len(np.unique(df['id']))
    total_tag = len(df)
    num_tag = len(np.unique(df['tag']))
    
    return num_post,total_tag,num_tag


# In[21]:


def get_cotag_post_fit_params(df_cotag):
    """
    return linear fit of cotag-post
    """
    counts = df_cotag['ct']
    cotag = df_cotag['cotag']
    slope, intercept, r_value, p_value, std_err  = st.linregress(counts,cotag)
    return slope, intercept, r_value, p_value, std_err 

# def get_lognormal_fit_params(df_cotag):
#     """
#     return lognormal fit of how many times a tag appears(the post counts of tags)
#     """
#     cts = np.array(df_cotag['ct'])
#     param = st.lognorm.fit(cts)
#     test_stat, p_value = st.kstest(cts,'lognorm',args=param)
#     s,loc,scale = param
#     return s,loc,scale, test_stat, p_value

def get_lognormal_fit_params_cotag(df_cotag):
    """
    return lognormal fit of how many times a cotag appears
    """
    cotag = df_cotag['cotag']
    param = st.lognorm.fit(cotag)
    s,loc,scale  = param
    test_stat, p_value = st.kstest(cotag,'lognorm',args=param)
    return s,loc,scale,test_stat, p_value


# In[22]:


def calculate_stats(paths, save_file):
    final_df = pd.DataFrame(columns=['filename','num_post','total_tag','num_tag',
                                     'slope', 'intercept', 'r_value', 'lin_p', 'std_err',
                                     # 's','loc','scale',"ks","logn_p",
                                     's_t','loc_t','scale_t',"ks_t","logn_p_t",])

    try:
        os.makedirs("gen_results/")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    SOURCE_PATH = paths[0]
    SOURCE_PATH_COTAG = paths[1]
    ct = 0
    for (dirpath, dirnames, filenames) in os.walk(SOURCE_PATH):
        for filename in filenames:
            if (not filename.endswith('_cotag.csv')) and filename.endswith('.csv'): 
                ct += 1
                #print(ct, filename)

                df = import_data(SOURCE_PATH + filename)

                df_cotag = pd.read_csv(SOURCE_PATH_COTAG+filename[:-4]+"_cotag.csv")

                num_post,total_tag,num_tag = get_data_stats(df)

                slope, intercept, r_value, lin_p, std_err = get_cotag_post_fit_params(df_cotag)
                #s,  loc,  scale, ks, logn_p           = get_lognormal_fit_params(df_cotag)
                s_t,loc_t,scale_t, ks_t, logn_p_t     = get_lognormal_fit_params_cotag(df_cotag)


                new_row = {'filename':filename,'num_post':num_post,'total_tag':total_tag,'num_tag':num_tag,
                           'slope':slope, 'intercept':intercept, 'r_value':r_value, 'lin_p':lin_p, 'std_err':std_err,
                           #'s':s,     'loc':loc,    'scale':  scale,  "ks": ks,     "logn_p":logn_p,
                           's_t':s_t, 'loc_t':loc_t,'scale_t':scale_t, "ks_t": ks_t, "logn_p_t":logn_p_t }
                final_df = final_df.append(new_row, ignore_index=True)
                final_df.to_csv(save_file)
    print("done")
    return final_df


# In[23]:


paths = [SOURCE_PATH,SOURCE_PATH_COTAG]
final_df = calculate_stats(paths,CSV_NAME)
final_df['num_post_cor'] = final_df.apply(lambda x: find_n(x['num_post'], x['total_tag']), axis=1)
final_df['theory_slope'] = final_df.apply(lambda x: ( x['total_tag']- (x['total_tag']/x['num_tag']))/x['num_post_cor'] , axis=1)
final_df.to_csv("./gen_results/gen_stats.csv")

