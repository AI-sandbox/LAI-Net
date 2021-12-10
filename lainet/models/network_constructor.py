import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .lainet import LAINetOriginal



def get_network(config, input_dimension, number_ancestries):
    network_type = config['MODEL']['NETWORK']
    window_size = config['MODEL']['WINDOW_SIZE']

    INPUT_DIM = input_dimension
    ANC_NUM = number_ancestries

    if network_type == 'lainet':
        net = LAINetOriginal(INPUT_DIM, ANC_NUM, window_size=window_size)
    elif network_type == 'small':
        net = LAINetOriginal(INPUT_DIM, ANC_NUM, window_size=window_size, include_hidden=False)           
        
    else:
        print('Network type not valid! Only - "lainet" or "small"')
        assert False

    return net