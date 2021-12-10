import torch
import numpy as np



def gmap_to_rate_per_snp(info, gmap):
    cm_interp = np.interp(info['pos'], gmap['pos'], gmap['pos_cm'])
    prob = np.gradient(cm_interp/100.0)
    return cm_interp, prob


def get_random_simulation_batch(snps, labels, device, bsize=200, num_generation_max=100, num_generation=50, balanced=False,
                                mode='uniform', generation_num_list=[1,2,4,8,16,32,64,128], rate_per_snp=None, chm=None, cM=None,
                                simulate_in_device=True):
    with torch.no_grad():
        
        # Make sure input arrays are pytorch tensors
        if not torch.is_tensor(snps):
            snps = torch.tensor(snps)
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels)

        
        # Obtain number of samples and SNPs - make sure dimensions of labels and SNPs match
        num_samples = snps.shape[0]
        num_snps = snps.shape[1]
        
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(1).repeat(1,num_snps)

        
        # Obtain batch of samples - if bsize is smaller than or equal to 0, all the samples are used
        if bsize>0:
            if not balanced:
                rand_idx = torch.randint(num_samples, (bsize,))
            else:
                unique_labels = torch.unique(labels[:,0])
                samples_per_class = int(bsize/len(unique_labels))
                bsize = samples_per_class * len(unique_labels) # Updating batch size so its divisible
                rand_idx = []
                for c in unique_labels:
                    idx_c = torch.nonzero(labels[:,0]==c).squeeze()#.numpy()
                    rand_idx_c = np.random.choice(idx_c.numpy(), samples_per_class)
                    rand_idx.extend(rand_idx_c)
        else: #Use all samples
            bsize = num_samples
            rand_idx = torch.randperm(bsize)
            
        batch_snps = snps[rand_idx,:]
        batch_labels = labels[rand_idx,:]

        # If simulate_in_device, batch is moved into device before simulation, else is moved after simulation
        if simulate_in_device:
            batch_snps = batch_snps.to(device)
            batch_labels = batch_labels.to(device)
        
        
        # Select number of generations
        if mode == 'uniform':
            num_generation = np.random.randint(0, num_generation_max)
        elif mode == 'exponential':
            num_generation = np.exp(np.random.uniform(0, np.log(num_generation_max)))
        elif mode == 'pre-defined':
            num_generation = np.random.choice(generation_num_list)
        elif mode == 'fix':
            num_generation = num_generation # Only used for mode 'fix'
        else:
            assert False, 'Simulation mode not valid - use "uniform", "pre-defined", or "fix"'

        
        # Obtain a list of switch point indexes
        # if rate_per_snp (from genetic map) is available, a binomial is sampled
        if rate_per_snp is not None:
            switch = np.random.binomial(num_generation, rate_per_snp) % 2
            split_point_list = np.flatnonzero(switch)
            
        # else if cM is available, a uniform distribution is used
        elif cM is not None:
            switch_per_generation = cM/100
            split_point_list = torch.randint(num_snps, (int(num_generation*switch_per_generation),))  
            
        # else if chm number is available, a hardcoded cM (for humans) is used
        elif chm is not None:
            cM_list = [286.279234, 268.839622, 223.361095, 214.688476, 204.089357, 192.039918, 187.2205, 168.003442, 166.359329, 181.144008, 158.21865, 174.679023, 125.706316, 120.202583, 141.860238, 134.037726, 128.490529, 117.708923, 107.733846, 108.266934, 62.786478, 74.109562]
            switch_per_generation = cM_list[(int(chm)-1)]/100
            split_point_list = torch.randint(num_snps, (int(num_generation*switch_per_generation),))  
            
        # else 1 switch per generation is used
        else:
            split_point_list = torch.randint(num_snps, (int(num_generation*1),))  

        # Perform the simulation    
        for split_point in split_point_list:
            rand_perm = torch.randperm(bsize)
            batch_snps[:,split_point:] = batch_snps[rand_perm,split_point:]
            batch_labels[:,split_point:] = batch_labels[rand_perm,split_point:]
           
        
        if not simulate_in_device:
            batch_vcf_train = batch_vcf_train.to(device)
            batch_map_train = batch_map_train.to(device)

        return batch_snps, batch_labels