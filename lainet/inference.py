import os, sys, time
import allel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from collections import Counter
import gzip
from scipy.interpolate import interp1d

from .utils.eval import compute_accuracy, AccuracyLogger
from .utils.output_writer import get_meta_data, write_msp_tsv






def snp_intersection(pos1, pos2, verbose=False):
    """
    Finds interception of snps given two arrays of snp position 
    """

    if len(pos2) == 0:
        print("Error: No SNPs of specified chromosome found in query file.")
        print("Exiting...")
        sys.exit(0)
    
    # find indices of intersection
    idx1, idx2 = [], []
    for i2, p2 in enumerate(pos2):
        match = np.where(pos1==p2)[0]
        if len(match) == 1:
            idx1.append(int(match))
            idx2.append(i2)

    intersection = set(pos1) & set(pos2)
    if len(intersection) == 0:
        print("Error: No matching SNPs between model and query file.")
        print("Exiting...")
        sys.exit(0)

    if verbose:
        print("- Number of SNPs from model:", len(pos1))
        print("- Number of SNPs from file:",  len(pos2))
        print("- Number of intersecting SNPs:", len(intersection))
        intersect_percentage = round(len(intersection)/len(pos1),4)*100
        print("- Percentage of model SNPs covered by query file: ",
              intersect_percentage, "%", sep="")

    return idx1, idx2

# TODO: move to input/output file
def read_query_vcf(query_vcf_file_path, snp_pos_fmt=None, snp_ref_fmt=None, miss_fill=2, return_idx=False, verbose=True):
    """
    Converts vcf file to numpy matrix. 
    If SNP position format is specified, then comply with that format by filling in values 
    of missing positions and ignoring additional positions.
    If SNP reference variant format is specified, then comply with that format by swapping where 
    inconsistent reference variants.
    Inputs
        - vcf_data: already loaded data from a vcf file
        - snp_pos_fmt: desired SNP position format
        - snp_ref_fmt: desired reference variant format
        - miss_fill: value to fill in where there are missing snps
    Outputs
        - npy matrix on standard format
    """
    vcf_data = allel.read_vcf(query_vcf_file_path)
    # reshape binary represntation into 2D np array 
    data = vcf_data["calldata/GT"]
    chm_len, n_ind, _ = data.shape
    data = data.reshape(chm_len,n_ind*2).T
    mat_vcf_2d = data
    vcf_idx, fmt_idx = np.arange(n_ind*2), np.arange(n_ind*2)

    if snp_pos_fmt is not None:
        # matching SNP positions with standard format (finding intersection)
        vcf_pos = vcf_data['variants/POS']
        fmt_idx, vcf_idx = snp_intersection(snp_pos_fmt, vcf_pos, verbose=verbose)
        # only use intersection of variants (fill in missing values)
        fill = np.full((n_ind*2, len(snp_pos_fmt)), miss_fill)
        fill[:,fmt_idx] = data[:,vcf_idx]
        mat_vcf_2d = fill

    if snp_ref_fmt is not None:
        # adjust binary matrix to match model format
        # - find inconsistent references
        vcf_ref = vcf_data['variants/REF']
        swap = vcf_ref[vcf_idx] != snp_ref_fmt[fmt_idx] # where to swap w.r.t. intersection
        if swap.any() and verbose:
            swap_n = sum(swap)
            swap_p = round(np.mean(swap)*100,4)
            print("- Found ", swap_n, " (", swap_p, "%) different reference variants. Adjusting...", sep="")
        # - swapping 0s and 1s where inconsistent
        fmt_swap_idx = np.array(fmt_idx)[swap]  # swap-index at model format
        mat_vcf_2d[:,fmt_swap_idx] = (mat_vcf_2d[:,fmt_swap_idx]-1)*(-1)

    # make sure all missing values are encoded as required
    missing_mask = np.logical_and(mat_vcf_2d != 0, mat_vcf_2d != 1)
    mat_vcf_2d[missing_mask] = miss_fill

    # return npy matrix
    if return_idx:
        return mat_vcf_2d, vcf_idx, fmt_idx

    return mat_vcf_2d, vcf_data






def inference_main(config, query_vcf_file_path, pretrained_folder, output_folder_prefix):
    print('Starting inference...')

    
    # Loading Network and meta-data ---------------------------------------
    print('Loading best performing networks...')

    #nets = []
    #for k in range(ensemble):
    #    network_path = pretrained_folder+'_{}_'.format(k)+'_network_model.pth'
    #    print('Loading... {}'.format(network_path))
     #   net = torch.load(network_path)
     #   #net = nn.DataParallel(net)
     #   nets.append(net)
    network_path = pretrained_folder+'_{}_'.format(0)+'_network_model.pth'
    net = torch.load(network_path)
    #nets.append(net)
    
    print('Loading Meta-data...')
    info_path = pretrained_folder+'_info.npy' 
    info = np.load(info_path, allow_pickle=True).item()

    anc_names_path = pretrained_folder+'_ancestry_names.txt'
    populations = np.loadtxt(anc_names_path, dtype=str)
    
    
    
    # Loading Query VCF ---------------------------------------------------
    print('Loading query vcf in...{}'.format(query_vcf_file_path))
    snp_npy, vcf_data = read_query_vcf(query_vcf_file_path, snp_pos_fmt=info['pos'], snp_ref_fmt=info['ref'], miss_fill=0.5)
    val_snps = torch.tensor(snp_npy)

    # Running Network ---------------------------------------------------
    print('Running network...')
    predicted, prob = run_net_in_query(config, net, val_snps)
    
    print(predicted.shape)

    pred_labels = predicted.cpu().numpy()
    if len(pred_labels.shape) == 3:
        pred_labels = pred_labels.transpose(1,0,2)
        pred_labels = pred_labels.reshape(pred_labels.shape[0], pred_labels.shape[1]*pred_labels.shape[2]).T


    # Writing output file (.msp.tsv) ----------------------------------
    print('Writing .msp file...' )
    chm = info['chm'][0]
    pos = info['pos']
    query_samples = vcf_data['samples']
    if hasattr(net, 'module'):
        n_wind = net.module.num_windows
        wind_size = net.module.windows_size
    else:
        n_wind = net.num_windows
        wind_size = net.windows_size

    meta = get_meta_data(chm, pos, pos, n_wind, wind_size)
    write_msp_tsv(output_folder_prefix, meta, pred_labels, populations, query_samples)
    print('Done writing')
    
    return net, predicted, prob, val_snps



def run_net_in_query(config, net, snp_tensor):
   
    batch_size = config['TRAINING']['BATCH_SIZE']
    device = config['TRAINING']['DEVICE']
    
    net = net.to(device)
    net.eval()
    
    val_snps = snp_tensor

    with torch.no_grad():
        if hasattr(net, 'module'):
            haploid2diploid = net.module.haploid2diploid
            is_haploid = net.module.is_haploid
        else:
            haploid2diploid = net.haploid2diploid
            is_haploid = net.is_haploid        

        num_b = np.max([math.ceil(val_snps.shape[0]/batch_size), 1])
        predicted_list = []
        prob_list = []
        for k in range(num_b):
            s, e = k * batch_size, np.min([(k+ 1) * batch_size, val_snps.shape[0]])

            inputs = val_snps[s:e, ...].to(device).float()

            if not is_haploid:
                inputs = haploid2diploid(inputs)

            _, output = net(inputs)
            prob_list.append(F.softmax(output, dim=1))
            _, predicted = torch.max(output, 1)
            predicted_list.append(predicted)

        predicted = torch.cat(predicted_list)
        prob = torch.cat(prob_list)
    return predicted, prob







