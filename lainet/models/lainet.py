import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class LAINetOriginal(nn.Module):

    def __init__(self, input_dimension, ancestry_number, window_size=500, include_hidden = True, hidden_size=30,
                 list_dropout_p=[0.0, 0.0], with_bias=True, is_haploid=False, 
                 norm_layer=nn.BatchNorm1d, activation=nn.ReLU(), smooth_kernel_size = 75,  
                 with_softmax_before_smooth=True, normalize_input_smoother=False):
        
        super(LAINetOriginal, self).__init__()
        
        self.is_haploid = is_haploid

        self.num_ancestries = ancestry_number
        self.windows_size = window_size

        self.input_mode = 'crop'
        self.input_dimension_real = input_dimension 
        self.croppping_dimension = self.input_dimension_real % self.windows_size # Extra SNPs
        self.input_dimension = self.input_dimension_real - self.croppping_dimension
        self.num_windows = int(np.floor(self.input_dimension/self.windows_size))

        #self.input_dimension = input_dimension

        self.include_hidden = include_hidden
        if self.include_hidden:
            self.dim_list = [window_size, hidden_size, ancestry_number]
        else:
            self.dim_list = [window_size, ancestry_number]

        self.with_bias = with_bias
        self.norm_layer = norm_layer
        self.activation = activation
        

        self.smooth_kernel_size = smooth_kernel_size
        self.with_softmax_before_smooth = with_softmax_before_smooth
        

      
        ## Network architecture
        ## Base Windows            
        self.ListMLP = nn.ModuleList()

        for i in range(self.num_windows):
            if i == self.num_windows-1:
                dim_list = np.copy(self.dim_list)
                dim_list[0] += np.remainder(self.input_dimension, self.windows_size)
            else:
                dim_list = np.copy(self.dim_list)

            self.ListMLP.append(self.gen_mlp(dim_list))
           
    
        ## Smoother
        self.smoother = self.get_smoother()


        
    def pad_or_crop_input(self, x, is_labels=False):
        if self.input_mode == 'pad':
            if is_labels:
                x = F.pad(x.float().unsqueeze(1), (0, self.padding_dimension), "reflect").squeeze(1).long()
            else:
                x = F.pad(x, (0, self.padding_dimension), "constant", 0)
        elif self.input_mode == 'crop':
            x = x[:,0:self.input_dimension]
        return x        
        
        
    def input2windows(self, x):
        x = self.pad_or_crop_input(x, False)
        windows = []
        for j in range(len(self.ListMLP)):
            if j == len(self.ListMLP)-1:
                _x = x[:, j*self.windows_size:]
            else:
                _x = x[:, j * self.windows_size : (j + 1) * self.windows_size]
            windows.append(_x)
        return windows
    
    
    def labels2windows(self, x):
        wins = self.input2windows(x)
        vs = []
        for w in wins:
            v, i = torch.mode(w, axis=1)
            vs.append(v)
        vmap = torch.stack(vs, axis=1)
        return vmap

    
    def haploid2diploid(self, x):
        return torch.stack([x[0::2,:],x[1::2,:]], dim=2)
    
    
    def forward(self, x):
        x = x.add_(-0.5).mul_(2)
        if self.is_haploid:
            return self.forward_haploid(x)
        else:
            return self.forward_diploid(x)

               
    
    def forward_base(self, x):
        
        windows = self.input2windows(x)
        outs = []
        for j in range(len(self.ListMLP)):
            o = self.ListMLP[j](windows[j])
            outs.append(o)
        out = torch.stack(outs,dim=2)
        return out
    
    
    def forward_smoother(self, x):
        out = x
        
        if self.with_softmax_before_smooth:
            _out = F.softmax(out, dim=1)
        else:
            _out = out
                
        out_smooth = self.smoother(_out)
            
        if not self.is_haploid:
            out_smooth = out_smooth[:,:,:,0:2]
                
        return out_smooth
    
    
    def forward_haploid(self, x):
        out_base = self.forward_base(x)
        out_smooth = self.forward_smoother(out_base)
        return out_base, out_smooth
    
    
    def forward_diploid(self, x):
        out_0 = self.forward_base(x[:,:,0])
        out_1 = self.forward_base(x[:,:,1])
        out_base = torch.stack([out_0, out_1], dim=3)

        out_smooth = self.forward_smoother(out_base)
        return out_base, out_smooth
    
    
    def gen_mlp(self, list_dim):
        _layer_list = []
        for i in range(len(list_dim)-1):
            _layer_list.append(nn.Linear(list_dim[i], list_dim[i+1], bias=self.with_bias))
            if i < len(list_dim)-2:
                    _layer_list.append(self.activation)
                    _layer_list.append(self.norm_layer(list_dim[i+1], affine=False))

        mlp = nn.Sequential(*_layer_list)
        return mlp



    def get_smoother(self):
        return self._get_conv(self.num_ancestries, self.num_ancestries)
        
        
    def _get_conv(self, in_dim, out_dim):
        if self.is_haploid:
            return nn.Conv1d(in_dim, out_dim, self.smooth_kernel_size, padding=int(np.floor(self.smooth_kernel_size / 2)),
                                 padding_mode='reflect')
        else:
            return nn.Conv2d(in_dim, out_dim, (self.smooth_kernel_size, 2), padding=(int(np.floor(self.smooth_kernel_size / 2)),1),
                             padding_mode='reflect')




