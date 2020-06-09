import torch
from torch.nn import Parameter, ParameterList
import torch.nn as nn
import torch.nn.functional as F
import math


class myRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, recurrent_init=None,
                 hidden_init=None, nonlinearity='tanh'):
        super(myRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_xh = Parameter(torch.Tensor(input_size, hidden_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_xh = Parameter(torch.Tensor(hidden_size))
        self.recurrent_init = recurrent_init
        self.hidden_init = hidden_init
        
        activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()],
            ['tanh', nn.Tanh()]
        ])
        self.activation = activations[nonlinearity]
        self.reset_parameters()
        
    def reset_parameters(self):
        for name, weight in self.named_parameters():
            if "bias" in name:
                weight.data.zero_()
            elif "weight_hh" in name:
                if self.recurrent_init is None:
                    #nn.init.constant_(weight, 1)
                    nn.init.normal_(weight, 0, 0.01)
                else:
                    self.recurrent_init(weight)
            elif "weight_xh" in name:
                if self.hidden_init is None:
                    nn.init.normal_(weight, 0, 0.01)
                else:
                    self.hidden_init(weight)
            else:
                weight.data.normal_(0, 0.01)
    def forward(self, input, h):
        h = self.activation(torch.matmul(input, self.weight_xh) + torch.matmul(h, self.weight_hh) + self.bias_xh)
        
        return h

class myRNN(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes=[32, 32], batch_first = True, recurrent_inits=None,
                 hidden_inits=None, **kwargs):
        super(myRNN, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_first = batch_first
        
        if batch_first:
            self.time_index = 1
            self.batch_index = 0
        else:
            self.time_index = 0
            self.batch_index = 1
        
        rnn_cells = []
        in_size = input_size
        i=0
        for i, hidden_size in enumerate(hidden_layer_sizes):
            if recurrent_inits is not None:
                kwargs["recurrent_init"] = recurrent_inits[i]
            if hidden_inits is not None:
                kwargs["hidden_init"] = hidden_inits[i]
            rnn_cells.append(myRNNCell(in_size, hidden_size, **kwargs))
            in_size = hidden_size
        
        self.rnncells = nn.ModuleList(rnn_cells)
        
        #h0 = torch.zeros(hidden_size * num_directions, requires_grad=False)
        #self.register_buffer('h0', h0)
        
    def forward(self, x, hidden=None):
        time_index = self.time_index
        batch_index = self.batch_index
        hiddens = []
        
        i=0
        for cell in self.rnncells:
            #hx = self.h0.unsqueeze(0).expand(
            #    x.size(batch_index),
            #    self.hidden_size * num_directions).contiguous()
            self.device = x.device
            hx = torch.zeros(x.size(batch_index), self.hidden_layer_sizes[i]).to(self.device)
            x_n = []
            outputs = []
            x_time = torch.unbind(x, time_index)
            seqlen = len(x_time)
            for t in range(seqlen):
                hx = cell(x_time[t], hx)
                outputs.append(hx)
            x = torch.stack(outputs, time_index)
            #x=torch.cat(outputs, -1)
            hiddens.append(hx)
            i=i+1
        
        return x, torch.cat(hiddens, -1)
            
                 