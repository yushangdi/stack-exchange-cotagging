# Modeling and Analysis of Tagging Networks in Stack Exchange Communities

This code and data repository accompanies the paper:

- [Modeling and Analysis of Tagging Networks in Stack Exchange Communities](...) - , <a href="https://dainves.github.io/">Xiang Fu*</a>, <a href="http://yushangdi.github.io/">Shangdi Yu*</a>, <a href="http://www.cs.cornell.edu/~arb/">Austin R. Benson</a>. (2018)

For questions, please email Shangdi at sy543@cornell.edu.

The code for analyzing Stack Exchanges communities, as well as the code to generate the synthetic graphs for section 4, is written in Python 3.

We used the following versions of external python libraries:

* `networkx=2.2`
* `numpy=1.15.4`
* `pandas=0.23.4`
* `pandasql==0.7.3`
* `scipy=1.2.0`
* `scikit-learn==0.20.1`
* `matplotlib==3.0.2`
* `powerlaw==1.4.6` - [Power-law distributions in empirical data](https://arxiv.org/abs/0706.1062)


### Reproducing results and figures

To produce the results presented in the paper, run the python files in the following order,
and results will be generated through running the code. It will take hours to run all code on a
normal computer for all the 168 stack-exchange networks. The code for calculating Clustering
Coefficients takes significant longer time than other code.

1.
```python
python sx01cotagging_network.py
```
Generate: "./cotag_data/\*\_cotag.csv" for each stack-exchange community "./parsed/\*.txt".

Each ".csv" file describes the cotagging network of the corresponding stack-exchange community.

2.
```python
python sx02number-of-tags-on-posts_data.py
```

Generate: "./results/tag_num.csv".


3.
```python
python sx03clustering_coefficient_data.py
```

Generate: "./results/clustering.csv" and "./results/clustering2.csv".

<!-- 1st -->

4.
```python
python sx04statistics_data.py
```

Generate: "./results/stats.csv"

5.
```python
python sx05tag-frequency.py
```

Generate: "./results/tagfreq.csv" and "./results/patents_is_different.csv"


6.
```python
python sx06polynomial-fit_data.py
```

Generate: "./results/poly_params.csv" and "./results/param_PC.npy".


7.
```python
python sx07generative-model-theoretical-statistics.py
```

Generate: "./results/theory_cotag_u.csv", "./results/var_zero_post.csv", and
"./files/theory_unique/\*.csv" for for each stack-exchange community "./parsed/\*.txt".

<!-- 2nd -->

8.
```python
python sx08generative-model.py
```

Generate: "./files/generated/gen\_\*.csv" and "./files/generated/cotag_files/gen\_\*\_cotag.csv"
for each stack-exchange community "./parsed/\*.txt".

<!-- 3rd -->

9.
```python
python sx09number-of-tags-on-posts_model.py
```

Generate: "./gen_results/gen_tag_num.csv".


10.
```python
python sx10clustering_coefficient_model.py
```

Generate: "./gen_results/gen_clustering.csv", "./gen_results/gen_clustering2.csv",
and "./gen_results/gen_param_PC.npy".



11.
```python
python sx11statistics_model.py
```

Generate: "./gen_results/gen_stats.csv"

<!-- 4th -->

12.
```python
python sx12polynomial-fit_model.py
```

Generate: "./gen_results/gen_poly_params.csv" and "./gen_results/gen_param_PC.npy".

<!-- 5th -->

Now all data files should be generated correctly, and we are ready to make plots.

Run the jupyter notbook data_description_plot.ipynb for Figure 1.

Run make_plots_data.ipynb for data-related plots.

Run make_plots_model.ipynb for model-related plots.
