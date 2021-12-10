import os, sys, time
import allel
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import math

from collections import Counter
import gzip
from scipy.interpolate import interp1d


def get_meta_data(chm, model_pos, query_pos, n_wind, wind_size, gen_map_df=None):
    """
    Transforms the predictions on a window level to a .msp file format.
        - chm: chromosome number
        - model_pos: physical positions of the model input SNPs in basepair units
        - query_pos: physical positions of the query input SNPs in basepair units
        - n_wind: number of windows in model
        - wind_size: size of each window in the model
        - genetic_map_file: the input genetic map file
    """

    model_chm_len = len(model_pos)
    
    # chm
    chm_array = [chm]*n_wind

    # start and end pyshical positions
    if model_chm_len % wind_size == 0:
        spos_idx = np.arange(0, model_chm_len, wind_size)#[:-1]
        epos_idx = np.concatenate([np.arange(0, model_chm_len, wind_size)[1:],np.array([model_chm_len])])-1
    else:
        spos_idx = np.arange(0, model_chm_len, wind_size)[:-1]
        epos_idx = np.concatenate([np.arange(0, model_chm_len, wind_size)[1:-1],np.array([model_chm_len])])-1

    spos = model_pos[spos_idx]
    epos = model_pos[epos_idx]

    # start and end positions in cM (using linear interpolation, truncate ends of map file)
    if gen_map_df is not None:
        end_pts = tuple(np.array(gen_map_df.pos_cm)[[0,-1]])
        f = interp1d(gen_map_df.pos, gen_map_df.pos_cm, fill_value=end_pts, bounds_error=False) 
        sgpos = np.round(f(spos),5)
        egpos = np.round(f(epos),5)
    else:
        sgpos = [1]*len(spos)
        egpos = [1]*len(epos)

    # number of query snps in interval
    wind_index = [min(n_wind-1, np.where(q == sorted(np.concatenate([epos, [q]])))[0][0]) for q in query_pos]
    window_count = Counter(wind_index)
    n_snps = [window_count[w] for w in range(n_wind)]

    #print(len(chm_array), len(spos), len(epos), len(sgpos), len(egpos), len(n_snps))
    # Concat with prediction table
    meta_data = np.array([chm_array, spos, epos, sgpos, egpos, n_snps]).T
    meta_data_df = pd.DataFrame(meta_data)
    meta_data_df.columns = ["chm", "spos", "epos", "sgpos", "egpos", "n snps"]

    return meta_data_df

  
    
def write_msp_tsv(msp_prefix, meta_data, pred_labels, populations, query_samples, write_population_code=False):
    
    msp_data = np.concatenate([np.array(meta_data), pred_labels.T], axis=1).astype(str)
    
    with open(msp_prefix+".msp.tsv", 'w') as f:
        if write_population_code:
            # first line (comment)
            f.write("#Subpopulation order/codes: ")
            f.write("\t".join([str(pop)+"="+str(i) for i, pop in enumerate(populations)])+"\n")
        # second line (comment/header)
        f.write("#"+"\t".join(meta_data.columns) + "\t")
        f.write("\t".join([str(s) for s in np.concatenate([[s+".0",s+".1"] for s in query_samples])])+"\n")
        # rest of the lines (data)
        for l in range(msp_data.shape[0]):
            f.write("\t".join(msp_data[l,:]))
            f.write("\n")
            
    return