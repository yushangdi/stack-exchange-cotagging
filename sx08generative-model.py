#!/usr/bin/env python
# coding: utf-8

# ### using the powerlaw package's lognormal fit and generating method,  x_min = 1

# In[1]:


import numpy as np
import pandas as pd
import random
from pandasql import sqldf
import powerlaw
import os
import errno


# In[2]:


DEBUG = False


# ## Calculate Cotag

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
def cotag_calculate(tag_dist, tag_supply):
    """
    take in tag_dist, tag_supply, and calculate count, cotag_count, and unique_cotag_count
    """

    Cul = pd.DataFrame({'id':tag_dist, 'tag':tag_supply})
    Cul = sqldf(QUERY_COTAG, locals())
    return Cul


# In[4]:


def get_params(filename, param_df):
    "e.g. filename = 'gis.txt'  "
    mu,sigma = param_df[param_df['filename']==filename][["mu","sigma"]].items()
    mu = float(mu[1])
    sigma = float(sigma[1])
    return [mu,sigma]


# ## PATH Configuration

# The specified path must exist before running the code

# In[5]:


MODEL = "bakset_relaxed"
PARAM_DF_PATH = './results/tagfreq.csv'

# PATH to the source txt files
SOURCE_PATH = "parsed/"
SOURCE_PATH_COTAG = "cotag_data/"

# PATH to store the generated networks
# also need folder ""/cotag_files
MODEL_PATH_MAP = {
    "bakset_relaxed": "files/generated/"
}

# prefix of generated graphs
MODEL_PREFIX_MAP = {
    "bakset_relaxed": "gen_"
}

# PATH to store the generated graphs
MODEL_GRAPH_MAP = {
    "bakset_relaxed": "files/plots/"
}

"""
return PATH to store the generated networks
"""
def get_path_from_model(model):
    assert model in MODEL_PATH_MAP.keys()
    data_path = MODEL_PATH_MAP[model]
    return data_path

"""
return prefix of generated graphs
"""
def get_prefix(model):
    assert model in MODEL_PREFIX_MAP.keys()
    graph_prefix = MODEL_PREFIX_MAP[model]
    return graph_prefix


"""
return PATH to store the generated graphs
"""
def get_graph_folder(model):
    assert model in MODEL_GRAPH_MAP.keys()
    graph_path = MODEL_GRAPH_MAP[model]
    return graph_path


# In[6]:


def get_db_name(filename):
    return filename + '.db'

"""
return the result of `query` on `df` as a dataframe

The name of the table in the query must be Cul
"""
def execute_query(query,Cul):
    return sqldf(query, locals())


# ## Read in Data

# In[7]:


"""
file `data_path + filename + ".txt"` should be in format 

0 6 1 1 2014-04-15T18:11:01.93
0 6 2 1 2014-04-15T18:11:01.93
0 6 3 1 2014-04-15T18:11:01.93

otherwise please specify the column names with col_names

return the dataframe created from the txt file

if remove_time is True, do not include the time strings

e.g. filename = apple
"""
def data_preprocess(filename, data_path = None, col_names = ["ques", "tag", "id","time_str"], remove_time = True):
    if not data_path:
        data_path = SOURCE_PATH 
    
    df = pd.read_csv(data_path + filename + ".txt", sep = ' ', names = col_names)
    #db_name = data_path + "db_files/"+get_db_name(filename)
    #write_to_db(df,db_name, output_cols = col_names , tablename = 'Cul')
    #return db_name, df
    if remove_time:
        df = df[["ques", "tag", "id"]]
    return df


# ## Get post number, tag number, unique tag number, tag distribution

# In[8]:


"""
Return the number of posts, tags and unique tags
"""
def get_data_stats(df):
    num_post = df['id'].nunique()# len(np.unique(df['id']))
    total_tag = len(df)
    num_tag = df['tag'].nunique()#len(np.unique(df['tag']))
    
    return num_post,total_tag,num_tag


# ## Generate Random Graph

# In[9]:


"""
given post number and tag number, based on the probability of getting
0 tag post, generate a post number that will result in the orginal
number of nonzero tag posts
"""
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


# Assign tags to posts according to the note:
# For each tag, randomly assign it to one of the posts. 
def tag_dist_generate(tag_supply, num_post):
    post_tags_dict = dict()
    num_tag_supply = len(tag_supply)
    
    for tag,N in enumerate(tag_supply):
        posts = np.random.choice(num_post, N)
        for post in posts:
            current_dict = post_tags_dict.get(post, set([]))
            c = post_tags_dict.get(post,set([]))
            c.add(tag)
            
            post_tags_dict[post] = c
            
    return post_tags_dict


# This function generate a post-tag network with the following procedure:
# Get the post number(num_post), the sum of number of tags in each post(total_tag), and the number of unique tags(num_tag)
# Generate a list of tags based on num_tag, whose post count is in lognormal distribution
# A tag can have at most num_post post count, if more, truncate
# Normalize the tag number such that the total number of tags is similar to total_tag
# Assign tags to posts using the function <tag_dist_generate>
def cotag_network_gen(num_post, num_tag, total_tag, params, normalize = True):

    
    theoretical_distribution = powerlaw.Lognormal(xmin=1, parameters=params)#, discrete=True)
    tag_supply = theoretical_distribution.generate_random(num_tag)
    
    normalized_factor = None
    if normalize:
        max_app = num_post
        tag_supply[tag_supply > max_app] = max_app
        normalized_factor = total_tag/np.sum(tag_supply)
        tag_supply = np.round(tag_supply*normalized_factor).astype(int)


        #print("larger than max app:", np.sum(tag_supply > max_app))
        #print("normalized factor  :", normalized_factor)
    else:
        tag_supply = tag_supply.astype(int)
        
    total_tag_discrep = abs(total_tag-np.sum(tag_supply))/total_tag
    
    # general new tag assignment 
    post_tags_dict = tag_dist_generate(tag_supply, num_post)
    
    def modify(post_tags_dict):
        tag_supply = []
        tag_dist   = []
        for k,values in post_tags_dict.items():
            tag_dist += [k] * len(values)
            tag_supply += list(values)
        return tag_supply, tag_dist
    
    #print("bbb",type(tag_supply))
    tag_supply, tag_dist = modify(post_tags_dict)
    
    return tag_supply, tag_dist, post_tags_dict, total_tag_discrep, normalized_factor


# In[10]:


def generate_data(file_name,num_post,num_tag,total_tag, params,gen_path = None, gen_path_cotag = None, normalize = True):  

    
    # post number correction
    num_post_new = find_n(num_post,total_tag)
    
    # generate new graph
    results = cotag_network_gen(num_post_new, num_tag, total_tag, params, normalize = normalize)
    tag_supply, tag_dist, post_tags_dict, total_tag_discrep, normalized_factor = results
    
    # calculate discrepancies 
    num_tag_per_post = np.array([len(i) for i in list(post_tags_dict.values())]) 
    over_five_percent = np.sum(num_tag_per_post>5)/num_post
    num_post_discrep = (abs(len(num_tag_per_post)-num_post)/num_post)

    
    # write results to csv files
    df_gen = pd.DataFrame({'id':tag_dist, 'tag':tag_supply})
    df_gen.to_csv(gen_path)
    df_gen_cotag = cotag_calculate(tag_dist, tag_supply)
    df_gen_cotag.to_csv(gen_path_cotag)
    
    

    return df_gen_cotag, [over_five_percent,num_post_discrep, total_tag_discrep, normalized_factor]


# ## Plot

# In[11]:


def log_pre(df, x_log = True, y_log = True):
    return df[(df.T != 0).all()]



# # Run

# In[12]:


def run_model(file_name, param_df, model = None,  normalize = True):

    df = data_preprocess(file_name)
    num_post,total_tag,num_tag = get_data_stats(df)

    if not model:
        model = MODEL
    data_path = get_path_from_model(model)
    prefix = get_prefix(model)
    graph_path = get_graph_folder(model)
    
    gen_path = data_path + prefix + file_name + ".csv"
    gen_path_cotag = data_path + "cotag_files/" + prefix + file_name + "_cotag.csv"
    
    print (file_name)
    params = get_params(file_name, param_df)

    # over_five_percent,num_post_discrep, total_tag_discrep,normalized_factor = discrep_info
    df_gen_cotag, discrep_info = generate_data(file_name, num_post,num_tag,total_tag, 
                                               params = params,
                                               gen_path = gen_path, gen_path_cotag = gen_path_cotag, 
                                              normalize = normalize)
    
    
    return discrep_info


# In[13]:


try:
    os.makedirs("./files/generated/cotag_files")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        

param_df = pd.read_csv(PARAM_DF_PATH)
final_df = pd.DataFrame()
for (dirpath, dirnames, filenames) in os.walk("./parsed"):
    for filename in filenames:
        if filename.endswith('.txt'): 
            file = filename[:-4]
            results = run_model(file, param_df = param_df, normalize = True)
            over_five_percent,num_post_discrep, total_tag_discrep, normalized_factor = results
        
            new_row = {'filename':filename,
                        "num_post":over_five_percent,
                        "num_post_discrep":num_post_discrep,
                        "total_tag_discrep":total_tag_discrep,
                        "normalized_factor":normalized_factor}
            #print(new_row)
            final_df = final_df.append(new_row, ignore_index=True)
final_df.to_csv("results/discrep.csv")

