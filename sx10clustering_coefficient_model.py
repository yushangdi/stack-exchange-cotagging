#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import scipy.sparse
import numpy as np
import os
import errno
from pandasql import sqldf
import pandas as pd


# In[2]:


COTAG_PAIRS = ("select c1.tag as t1, c2.tag as t2, count(*) as ct "
                "from Cul as c1, Cul as c2 "
                "where c1.id = c2.id and c1.tag < c2.tag "
                "group by t1,t2")


# In[3]:


def get_data_stats(df):
    num_post = df['id'].nunique()# len(np.unique(df['id']))
    total_tag = max(df['tag'])+1
    num_tag = df['tag'].nunique()#len(np.unique(df['tag']))

    return num_post,total_tag,num_tag


# In[4]:


def compute_Clustering(df,total_tag):
    row = np.array(df['t1'])
    col = np.array(df['t2'])
    data = np.array(df['ct'])

    adj_matrix = scipy.sparse.coo_matrix((data, (row, col)),shape = (total_tag,total_tag))
    G = nx.from_scipy_sparse_matrix(adj_matrix)
    
    
    unweighted_cc = nx.average_clustering(G)

    weighted_cc = nx.average_clustering(G,weight = "weight")
    tri = nx.triangles(G)
    
    data = np.log(np.array(df['ct']))
    adj_matrix = scipy.sparse.coo_matrix((data, (row, col)),shape = (total_tag,total_tag))
    G = nx.from_scipy_sparse_matrix(adj_matrix)
    weighted_log = nx.average_clustering(G,weight = "weight")

    return unweighted_cc,weighted_cc, weighted_log, tri


# In[5]:


def compute_Clustering_log1(df,total_tag):
    # total_tag is the number of unique tags
    row = np.array(df['t1']) 
    col = np.array(df['t2']) 
    assert np.all(df['ct']>0)
    
    data = np.log(np.array(df['ct'])+1)
    adj_matrix = scipy.sparse.coo_matrix((data, (row, col)),shape = (total_tag,total_tag))
    G = nx.from_scipy_sparse_matrix(adj_matrix)
    weighted_log = nx.average_clustering(G,weight = "weight")

    return weighted_log


# In[8]:


path = 'files/generated/'
final_df = pd.DataFrame()
for filename in os.listdir(path):
    if filename.endswith('.csv'): 
        print(filename)

        try:
            Cul = pd.read_csv(path + filename, sep=",", index_col = 0)
            num_post,total_tag,num_tag = get_data_stats(Cul)
            df_pairs = sqldf(COTAG_PAIRS, locals())

            weighted_log = compute_Clustering_log1(df_pairs,total_tag)

            new_row = {'filename':filename,
                       'num_post':num_post,'total_tag':total_tag,'num_tag':num_tag,
                       'weighted_log':weighted_log}
            final_df = final_df.append(new_row, ignore_index=True)
        except Exception as e:
            print(e)
            print(filename, " failed")
        final_df.to_csv("gen_results/gen_clustering2.csv")
print("done")


# In[7]:


try:
    os.makedirs("gen_results/")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

path = 'files/generated/'
final_df = pd.DataFrame()
for filename in os.listdir(path):
    if filename.endswith('.csv'): 
        print(filename)

        try:
            Cul = pd.read_csv(path + filename, sep=",", index_col = 0)
            num_post,total_tag,num_tag = get_data_stats(Cul)
            df_pairs = sqldf(COTAG_PAIRS, locals())

            unweighted_cc,weighted_cc, weighted_log, tri= compute_Clustering(df_pairs,total_tag)
            triangle_counts = np.sum(list(tri.values()))/3
            new_row = {'filename':filename,
                       'num_post':num_post,'total_tag':total_tag,'num_tag':num_tag,
                       'unweighted':unweighted_cc,'weighted':weighted_cc,'weighted_log':weighted_log, 
                       "triangle_counts":triangle_counts}
            final_df = final_df.append(new_row, ignore_index=True)
        except Exception as e:
            print(e)
            print(filename, " failed")
        final_df.to_csv("gen_results/gen_clustering.csv")
print("done")

