#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import errno


# In[2]:
CSV_DIR = "results/"
SOURCE_PATH = "./parsed/"

def get_tags_count(filename, path, gen = False):
    
    if gen: 
        df = pd.read_csv(path + filename, sep = ',', index_col = 0)
        print
    else:
        df = pd.read_csv(path + filename, sep = ' ', names =["ques", "tag", "id", "time_str"])[["tag", "id"]]
    result = df.groupby('id').agg('count')
    n1 = int(np.sum(result==1))
    n2 = int(np.sum(result==2))
    n3 = int(np.sum(result==3))
    n4 = int(np.sum(result==4))
    n5 = int(np.sum(result==5))
    nm = int(np.sum(result>5))
    
    num_post = df['id'].nunique()
    return {"filename":filename,
            "n1":n1, 
            "n2":n2, 
            "n3":n3, 
            "n4":n4, 
            "n5":n5,
            'nm':nm,
            "n1p":n1/num_post, 
            "n2p":n2/num_post, 
            "n3p":n3/num_post, 
            "n4p":n4/num_post, 
            "n5p":n5/num_post,
           }


# In[3]:

try:
    os.makedirs(CSV_DIR)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

final_df = pd.DataFrame()
path = SOURCE_PATH
for filename in os.listdir(path):
    if filename.endswith('.txt'): 
        #print(filename)

        new_row = get_tags_count(filename, path)
        final_df = final_df.append(new_row, ignore_index=True)
final_df.to_csv(CSV_DIR + "tag_num.csv")
print("done")

