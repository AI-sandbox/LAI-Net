import os, sys, time
#import tqdm
from tqdm.auto import tqdm
import allel
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import label_binarize



from .utils.eval import compute_accuracy, AccuracyLogger, AccuracyLogger, complete_sk_eval, print_sk_eval
from .simulation import get_random_simulation_batch, gmap_to_rate_per_snp
from .utils.reader import load_founders_from_vcf_and_map, read_genetic_map, split_train_val
from .models.network_constructor import get_network
from .inference import run_net_in_query





def train_main(config, founders_vcf_file_path, founders_map_file_path, output_folder_prefix, chm=None, genetic_map_path=None):

    # Loading Dataset --------------------------------------------------------------------------------
    train_snps_npy, train_labels_npy, val_snps_npy, val_labels_npy, ancestry_names, info = load_and_process_dataset(config, founders_vcf_file_path, founders_map_file_path)
    print('A total of {} sequences are used for training and {} for validation...'.format(train_snps_npy.shape[0], val_snps_npy.shape[0]))
    print('A total of {} unique categories in train and {} in val'.format(len(np.unique(train_labels_npy)), len(np.unique(val_labels_npy))))
    if genetic_map_path is not None:
        print('Reading genetic map from... {}'.format(genetic_map_path))
        gmap = read_genetic_map(genetic_map_path, chm=chm)
    else:
        gmap = None


    # Save info of VCF and .map and ancestry names  ----------------------------------------------------------------
    output_info_path = output_folder_prefix+'_info.npy'
    output_anc_names_path = output_folder_prefix+'_ancestry_names.txt'
    np.savetxt(output_anc_names_path, ancestry_names, fmt='%s')
    np.save(output_info_path, info)
    print('Information of models saved in {}...'.format(output_info_path))


    # Create Network  --------------------------------------------------------------------------------
    print('Creating Network...')
    net = get_network(config, train_snps_npy.shape[1], len(ancestry_names))


    # Load Training Parameters  --------------------------------------------------------------------------------
    elapsed_time, best_accuracy, success, (val_snps, val_labels) = start_training(config, output_folder_prefix+'_{}_'.format(0), net, train_snps_npy, train_labels_npy, val_snps_npy, val_labels_npy, info, chm=chm, gmap=gmap)
    print('Finished with accuracy {} after {} seconds...'.format(best_accuracy, elapsed_time))

    with torch.no_grad():
        print('Running validation report...')
        # Validation report
        network_path = output_folder_prefix+'_{}_'.format(0)+'_network_model.pth'
        print('Loading... {}'.format(network_path))
        net = torch.load(network_path).eval()
        predicted, prob = run_net_in_query(config, net, val_snps)
        eval_predictions(net, val_labels, prob, predicted, cat_names=ancestry_names, print_results=True)



    


    

    
    
    
# TODO: this function could be merged with load_founders_from_vcf_and_map()
def load_and_process_dataset(config, founders_vcf_file_path, founders_map_file_path):
    random_trainval_split = config['TRAINING']['RANDOM_TRAINVAL_SPLIT']
    train_snps, train_labels, val_snps, val_labels, ancestry_names, info = load_founders_from_vcf_and_map(founders_vcf_file_path, founders_map_file_path, random_split=random_trainval_split)

    # Check if number of samples is even (for diploid based methods) -- if not, move one sequence from train to val
    if train_snps.shape[0] % 2 != 0:
        val_snps = np.concatenate([val_snps, train_snps[0,...][None,...]])
        val_labels = np.concatenate([val_labels, train_labels[0,...][None,...]])
        train_snps = train_snps[1:,...]
        train_labels = train_labels[1:,...]

    train_labels = np.repeat(train_labels[:,None], train_snps.shape[1], axis=1)
    val_labels = np.repeat(val_labels[:,None], train_snps.shape[1], axis=1)
    return train_snps, train_labels, val_snps, val_labels, ancestry_names, info






def start_training(config, output_folder_prefix, net, train_snps, train_labels, val_snps, val_labels, info, gmap=None, chm=None):
    output_network_path = output_folder_prefix+'_network_model.pth'#os.path.join(output_folder,'lainet_orig_best.pth')

    device = config['TRAINING']['DEVICE']
    batch_size = config['TRAINING']['BATCH_SIZE']
    alpha = config['TRAINING']['ALPHA']
    lr = config['TRAINING']['LEARNING_RATE']
    wd = config['TRAINING']['WEIGHT_DECAY']
    num_epochs = config['TRAINING']['NUM_EPOCHS']
    online_simulation_mode = config['TRAINING']['ONLINE_SIMULATION_MODE']
    online_simulation_realistic = config['TRAINING']['ONLINE_SIMULATION_REALISTIC']
    generation_num_list = config['TRAINING']['GENERATION_NUM_LIST']

    
    save_model = config['TRAINING']['SAVE_MODEL']
    num_total_iters = int(train_snps.shape[0]/batch_size * num_epochs)
    



    if config['TRAINING']['BALANCED_TYPE'] == 'Loss':
        unique, counts = np.unique(train_labels[:,0], return_counts=True)
        weights_ancestry = counts.min() / counts
        weights_ancestry = weights_ancestry / weights_ancestry.sum() * len(weights_ancestry)
        weights_ancestry = torch.tensor(weights_ancestry).float().to(device)

    else:
        weights_ancestry = None
        
    criterion = nn.CrossEntropyLoss(weight=weights_ancestry)



    if config['TRAINING']['OPTIM'] == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    elif config['TRAINING']['OPTIM'] == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        print('Optimizer not valid:', config['TRAINING']['OPTIM'])
        assert False
        


    iter_break = config['TRAINING']['ITER_BREAK']
    verbose_train=True
    use_tqdm = False
    
    if gmap is not None:
        cm_max = np.array(gmap['pos_cm'])[-1]
    else:
        cm_max = None
        
    if config['TRAINING']['BALANCED_TYPE'] == 'Batch':
        balanced_simulation = True
    else:
        balanced_simulation = False
        
    if online_simulation_realistic:
        gmap = gmap
    else:
        gmap = None
    
    # Start training
    print('Starting training...')
    elapsed_time, best_accuracy, success, (val_snps, val_labels) = train_eval(net, info, train_snps, train_labels, val_snps, val_labels, 
                                                      optimizer, criterion, device, batch_size, num_total_iters=num_total_iters,
                                                      alpha=alpha, verbose=verbose_train, use_tqdm =use_tqdm, 
                                                      iter_break = iter_break, save_model=save_model, iter_lr_threshold = 5,
                                                      online_simulation_mode = online_simulation_mode, 
                                                      generation_num_list = generation_num_list,                                                      
                                                      output_filename=output_network_path, chm=chm, cm_max =cm_max,
                                                                             balanced_simulation = balanced_simulation,
                                                                             gmap = gmap)
    
    return elapsed_time, best_accuracy, success, (val_snps, val_labels)






#TODO: in a future move it to pytorch lighting and move training logic inside network
def train_eval(net, info, train_snps_npy, train_labels_npy, val_snps_npy, val_labels_npy, optimizer, 
               criterion, device, batch_size, alpha=0.5, alpha_consistency = 1, verbose=True, use_tqdm = False,
               iter_break = 10, freq_iter=10, early_stop=2000000, stop_if_bad=False, num_total_iters=2000000,
               iter_lr_threshold = -1, min_lr=0.00001,
               generation_num_list = None, online_simulation_mode = 'uniform',
               dropout_input=False, optim_step=1, save_model=False, output_filename=None, gmap=None, cm_max=None, chm=None, balanced_simulation = True):
    
    print('Saving network in... {}'.format(output_filename))
    net = net.to(device)
    
    if use_tqdm:
        pbar = tqdm(total=100)
    
    #is_diploid = True
    
    if hasattr(net, 'module'):
        labels2windows = net.module.labels2windows
        haploid2diploid = net.module.haploid2diploid
        is_haploid = net.module.is_haploid
    else:
        labels2windows = net.labels2windows
        haploid2diploid = net.haploid2diploid
        is_haploid = net.is_haploid
    
    
    train_snps, train_labels = torch.tensor(train_snps_npy).float(), torch.tensor(train_labels_npy).long()
    val_snps, val_labels = torch.tensor(val_snps_npy).float(), torch.tensor(val_labels_npy).long()
    
    
    # Simulate Validation Dataset
    if gmap is not None:
        print('Using genetic map for realistic simulation')
        _, snp_recomb_rate = gmap_to_rate_per_snp(info, gmap)
    else:
        snp_recomb_rate = None
        
    with torch.no_grad():
        val_snps_list, val_labels_list = [], []
        for g in range(0,100,2):
            val_snps_batch, val_labels_batch = get_random_simulation_batch(val_snps, val_labels, 'cpu', bsize=int(val_snps.shape[0]/50*1.5), mode=online_simulation_mode, num_generation = g, rate_per_snp = snp_recomb_rate, chm=chm, cM=cm_max, balanced=balanced_simulation, generation_num_list = generation_num_list)
            val_snps_list.append(val_snps_batch)
            val_labels_list.append(val_labels_batch)

        val_snps_batch, val_labels_batch = get_random_simulation_batch(val_snps, val_labels, 'cpu', bsize=-1, mode=online_simulation_mode, num_generation = g, rate_per_snp = snp_recomb_rate, chm=chm, cM=cm_max, generation_num_list = generation_num_list)
        val_snps_list.append(val_snps_batch)
        val_labels_list.append(val_labels_batch)    
        
        val_snps = torch.cat(val_snps_list)
        val_labels = torch.cat(val_labels_list)
        val_labels_win = labels2windows(val_labels)
    print('Validation set of shape {} has been simulated...'.format(val_snps.shape))
    
    running_loss = 0.0
    best_accuracy = 0.0
    loggers = []
    loggers.append(AccuracyLogger(''))
    
    tic = time.time()
    success = True
    optimizer.zero_grad()
       
    for i in range(num_total_iters):
        net.train()
               
        with torch.no_grad():
            inputs, labels = get_random_simulation_batch(train_snps, train_labels, device, mode = online_simulation_mode, bsize=batch_size, generation_num_list = generation_num_list, rate_per_snp = snp_recomb_rate, chm=chm, cM=cm_max, balanced=balanced_simulation)
        labels = labels2windows(labels)
        
        if not is_haploid:
            if inputs.shape[0] % 2 != 0:
                inputs, labels = inputs[1:, ...], labels[1:, ...]
            inputs, labels = haploid2diploid(inputs), haploid2diploid(labels)
                
        outputs_base, outputs_smooth = net(inputs)

        loss_smooth = criterion(outputs_smooth, labels)
        loss_base = criterion(outputs_base, labels)               

           
        
        loss = alpha*loss_smooth + (1-alpha)*loss_base


        loss = loss / optim_step
        
        loss.backward()
        
        
        if i % optim_step == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item()

  
        if i % (2*optim_step) == 0 and verbose:
            print('[ %5d] loss: %.3f' %
                  ( i + 1, running_loss / 2))
            
            if running_loss > 500 and isinstance(criterion, nn.CrossEntropyLoss) and stop_if_bad:
                print('Early stop! - 0')
                success = False
                break
            running_loss = 0.0
            

        if i % (freq_iter*optim_step) == 0 :      
            with torch.no_grad():
                optimizer.zero_grad()
                net.eval()

                num_b = np.max([int(val_snps.shape[0]/batch_size), 1])
                predicted_list = []
                labels_list = []
                for k in range(num_b):
                    s, e = k * batch_size, np.min([(k+ 1) * batch_size, val_snps.shape[0]])
                    inputs, labels = val_snps[s:e, :].to(device).float(), val_labels_win[s:e, :].long().to(device)
                    
                    if not is_haploid:
                        inputs, labels = haploid2diploid(inputs), haploid2diploid(labels)

                    outputs_base, outputs_smooth = net(inputs)
                    _, predicted = torch.max(outputs_smooth, 1)

                    #print('shape labels / predicted')
                    labels_list.append(labels.flatten())
                    predicted_list.append(predicted.flatten())

                predicted = torch.cat(predicted_list)
                labels = torch.cat(labels_list)

                acc = compute_accuracy(predicted.flatten(), labels.flatten(), balanced_accuracy = True)
                is_best = loggers[0].log(acc.cpu().numpy())
                
                if verbose:
                    print(loggers[0])
                if use_tqdm:
                    pbar.update(loggers[0].current_accuracy)
                    
                if is_best and save_model:
                    if hasattr(net, 'module'):
                        torch.save(net.module, output_filename)
                    else:
                        torch.save(net, output_filename)

                iter_since_best_accuracy = loggers[-1].time_since_best
                
                if iter_since_best_accuracy == iter_lr_threshold:
                    for g in optimizer.param_groups:
                        g['lr'] = max(g['lr']*0.1, torch.tensor(min_lr))
                        if verbose:
                            print('Setting learning rate to... ', g['lr'])
                        if (best_accuracy < 0.3) and stop_if_bad:
                            print('Early stop! - 1')
                            success = False
                            return time.time() - tic, best_accuracy, success
                
                if iter_since_best_accuracy >= iter_break:
                    break
                
        best_accuracy =  loggers[-1].best_accuracy
        if ((i >= (early_stop) * freq_iter) and (best_accuracy < 0.1)) or ((i > 8*freq_iter*optim_step) and (best_accuracy < 0.1)) and stop_if_bad:
            print('Early stop! - 2')
            success = False
            return time.time() - tic, best_accuracy, success, (val_snps, val_labels)
                        
    #elapsed_time = time.time() - tic
    
    if use_tqdm:     
        pbar.close()
    
    return time.time() - tic, best_accuracy, success, (val_snps, val_labels)



def eval_predictions(net, labels, probs, predictions, cat_names=None, print_results=True):
    # TODO: clean it up
        
    predicted = predictions
    val_labels_ = labels
    
    # Setup network output and labels for sklearn eval functions
    pred_labels = predicted.cpu().numpy()
    probs = probs.cpu().numpy()
    if len(pred_labels.shape) == 3:
        pred_labels = pred_labels.transpose(1,0,2)
        pred_labels = pred_labels.reshape(pred_labels.shape[0], pred_labels.shape[1]*pred_labels.shape[2]).T
        
        probs = probs.transpose(0,3,1,2)
        probs = probs.reshape(probs.shape[0]*2, probs.shape[2], probs.shape[3])
        
    if hasattr(net, 'module'):
        labels2windows = net.module.labels2windows
        haploid2diploid = net.module.haploid2diploid
        is_haploid = net.module.is_haploid
    else:
        labels2windows = net.labels2windows
        haploid2diploid = net.haploid2diploid
        is_haploid = net.is_haploid

    val_labels = torch.tensor(val_labels_).to('cpu')
    predicted = pred_labels#predicted.to('cpu')

    val_labels_win = labels2windows(val_labels)

    
    n_classes = probs.shape[1]
    y_true = val_labels_win.flatten().cpu().numpy()
    y_one_hot = label_binarize(y_true, classes=range(n_classes))
    
    y_pred = predicted.flatten()#.cpu().numpy()
    y_prob = torch.tensor(probs).permute(0,2,1).flatten(start_dim=0, end_dim=1).cpu().numpy()
    
    if cat_names is None:
        cat_names = [str(i) for i in range(n_classes)]

       
    acc, acc_bal, accuracies, clf_report, cm, cm_norm, jacc_micro, jacc_macro, precision, recall, average_precision, mean_average_precision = complete_sk_eval(n_classes, y_true, y_one_hot, y_pred, y_prob, cat_names=cat_names)

    if print_results:
        print_sk_eval(cat_names, acc, acc_bal, accuracies, clf_report, cm, cm_norm, jacc_micro, jacc_macro, precision, recall, average_precision, mean_average_precision)
    
    return acc, acc_bal, accuracies, clf_report, cm, cm_norm, jacc_micro, jacc_macro, precision, recall, average_precision, mean_average_precision

