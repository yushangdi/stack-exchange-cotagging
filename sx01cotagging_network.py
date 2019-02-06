#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandasql import sqldf
import os
import errno

# In[2]:


# PATH to the source txt files
SOURCE_PATH = "parsed/"
TARGET_PATH_COTAG = "cotag_data/"


# In[3]:


QUERY_COTAG = ("with tags as (select t1.tag as q1, t2.tag as q2 "
                "from Cul as t1, Cul as t2  "
                "where t1.id = t2.id and t1.tag <> t2.tag), "
                " t_cotag as (select q1, count(*) as cotag, count(distinct q2) as cotag_u "
                "from tags  "
                "group by q1), "
            " t_ct as (select tag, count(distinct id) as ct "
                "from Cul "
                "group by tag) "
            "select t_ct.tag, ct, ifnull(cotag,0) as cotag, ifnull(cotag_u,0) as cotag_u "
            "from t_ct "
            "left join t_cotag on t_ct.tag = t_cotag.q1 ")


# In[4]:


def get_cotag_data(filename, data_path = None, col_names = ["ques", "tag", "id","time_str"]):
    if not data_path:
        data_path = SOURCE_PATH 
    
    df = pd.read_csv(data_path + filename + ".txt", sep = ' ', names = col_names)
    Cul = df[["id","tag"]]
    Cul = sqldf(QUERY_COTAG, locals())
    Cul.to_csv(TARGET_PATH_COTAG + filename + "_cotag.csv")
    return Cul


# In[5]:


def calculate_cotags(path, save_file = "stats.csv"):

    
    try:
        os.makedirs("./cotag_data")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    ct = 0
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith('.txt'): 
                ct += 1
                print(ct, filename)
                try:
                    df_cotag = get_cotag_data(filename[:-4])
                except Exception as e:
                    print(e)
                    print("~~~~~~~~~~~ %s~~~~~~~~" % filename)
    print("done")


# In[6]:


path = SOURCE_PATH
calculate_cotags(path)

