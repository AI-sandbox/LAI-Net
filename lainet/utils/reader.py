import os, sys, time
import allel
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def load_map_file(map_file):
    sample_map = pd.read_csv(map_file, sep="\t", header=None)
    sample_map.columns = ["sample", "ancestry"]
    ancestry_names, ancestry_labels = np.unique(sample_map['ancestry'], return_inverse=True)
    samples_list = np.array(sample_map['sample'])
    return samples_list, ancestry_labels, ancestry_names


def load_vcf_samples_in_map(vcf_file, samples_list):
    # Reading VCF
    vcf_data = allel.read_vcf(vcf_file)
    
    # Intersection between samples from VCF and samples from .map
    inter = np.intersect1d(vcf_data['samples'], samples_list, assume_unique=False, return_indices=True)
    samp, idx = inter[0], inter[1]

    # Filter only interecting samples
    snps = vcf_data['calldata/GT'].transpose(1,2,0)[idx,...]
    samples = vcf_data['samples'][idx]
    
    # Save header info of VCF file
    info = {
        'chm' : vcf_data['variants/CHROM'],
        'pos' : vcf_data['variants/POS'],
        'id'  : vcf_data['variants/ID'],
        'ref' : vcf_data['variants/REF'],
        'alt' : vcf_data['variants/ALT'],
    }
    
    return samples, snps, info


def split_train_val(snps, ancestry_labels, random_split=False):
    num_samples = snps.shape[0]
    n_val = int(num_samples*0.1)
    
    if random_split:
        perm = np.random.permutation(num_samples)

        val_snps = snps[perm[0:n_val],...]
        val_labels = ancestry_labels[perm[0:n_val],...]
        train_snps = snps[perm[n_val:],...]
        train_labels = ancestry_labels[perm[n_val:],...]
        
    else: # Per-class balanced split is done
        unique_labels =  np.unique(ancestry_labels)
        
        train_snps, val_snps, train_labels, val_labels = [], [], [], []
        for pop in unique_labels:

            pop_labels = ancestry_labels[ancestry_labels == pop,...]
            pop_snps = snps[ancestry_labels == pop,...]
            
            # If only one sample is available is included in train, val and test
            if len(pop_labels) == 1:
                train_snps.append(pop_snps)
                val_snps.append(pop_snps)
                train_labels.append(pop_labels)
                val_labels.append(pop_labels)
                
            # If more than 1 are available, a 90-10 split is done - at least 1 is included in val
            else:
                len_val = int(max(int(0.1 * len(pop_labels)), 1)) #Make sure at least 1 element is included
                len_train = len(pop_labels) - len_val

                val_labels.append(pop_labels[0:len_val,...])
                val_snps.append(pop_snps[0:len_val,...])
                train_labels.append(pop_labels[len_val:])
                train_snps.append(pop_snps[len_val:])

        train_snps, val_snps = np.concatenate(train_snps, axis=0), np.concatenate(val_snps, axis=0)
        train_labels, val_labels = np.concatenate(train_labels, axis=0), np.concatenate(val_labels, axis=0)
    
    return train_snps, train_labels, val_snps, val_labels




def load_founders_from_vcf_and_map(founders_vcf_file_path, founders_map_file_path, make_haploid=True, random_split=False, verbose=True):

    # Load .map file
    if verbose:
        print('Loading vcf and .map files...')
    samples_list, ancestry_labels, ancestry_names = load_map_file(founders_map_file_path)
    samples_vcf, snps, info = load_vcf_samples_in_map(founders_vcf_file_path, samples_list)
    
    if verbose:
        print('Done loading vcf and .map files...')

    # Order alphabetically vcf and .map samples
    argidx = np.argsort(samples_vcf)
    samples_vcf = samples_vcf[argidx]
    snps = snps[argidx, ...]

    argidx = np.argsort(samples_list)
    samples_list = samples_list[argidx]
    ancestry_labels = ancestry_labels[argidx, ...]
    
    # Check if samples in VCF and .map are consistent
    for s1, s2 in zip(samples_vcf, samples_list):
        assert s1 == s2

    num_samples = len(samples_list)
    
    if verbose:
        print('A total of {} diploid individuals where found in the vcf and .map'.format(num_samples))
        print('A total of {} ancestries where found: {}'.format(len(ancestry_names),ancestry_names))
    
    if make_haploid:
        nsamples, nseq, nsnps = snps.shape
        snps = snps.reshape(nsamples*nseq, nsnps)
        ancestry_labels = np.repeat(ancestry_labels, 2)

    # Train/val split - 90/10
    train_snps, train_labels, val_snps, val_labels = split_train_val(snps, ancestry_labels, random_split=random_split)
    
    
    return train_snps, train_labels, val_snps, val_labels, ancestry_names, info


def read_genetic_map(genetic_map_path, chm=None):
    
    gen_map_df = pd.read_csv(genetic_map_path, sep="\t", comment="#", header=None, dtype="str")
    gen_map_df.columns = ["chm", "pos", "pos_cm"]
    gen_map_df = gen_map_df.astype({'chm': str, 'pos': np.int64, 'pos_cm': np.float64})

    if chm is not None:
        chm = str(chm)
        if len(gen_map_df[gen_map_df.chm == chm]) == 0:
            gen_map_df = gen_map_df[gen_map_df.chm == "chr"+chm]
        else:
            gen_map_df = gen_map_df[gen_map_df.chm == chm]

    return gen_map_df

def load_results_file(map_path):
    ff = open(map_path, "r", encoding="latin1")
    matrix = []
    loc_func = lambda x: int(x.rstrip("\n"))
    for i in ff.readlines()[1:]:
        row = i.split("\t")[2:]
        row = np.vectorize(loc_func)(row)
        matrix.append(row-1)
    matrix = np.asarray(matrix).T
    ff.close()

    return matrix