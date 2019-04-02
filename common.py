
# coding: utf-8

# In[ ]:


'''
some common code for ctr model
'''


# In[ ]:


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


# In[ ]:


class FirstOrder(nn.Module):
    def __init__(self, params):
        super(FirstOrder, self).__init__()
        # parse params
        self.device = params['device']
        self.feature_size = params['feature_size']
        
        weights_first_order = torch.empty(self.feature_size, 1, 
                                          dtype=torch.float32, device=self.device,
                                          requires_grad=True)
        nn.init.normal_(weights_first_order)
        self.weights_first_order = nn.Parameter(weights_first_order)
        
    def forward(self, feature_values, feature_idx):  
        weights_first_order = self.weights_first_order[feature_idx, :]
        first_order = torch.mul(feature_values, weights_first_order.squeeze())
        return first_order


# In[ ]:


class SecondOrder(nn.Module):
    def __init__(self, params, get_embeddings=False):
        super(SecondOrder, self).__init__()
        # parse params
        self.device = params['device']
        self.feature_size = params['feature_size']
        self.embedding_size = params['embedding_size']
        self.get_embeddings = get_embeddings
        
        feature_embeddings = torch.empty(self.feature_size, self.embedding_size, 
                              dtype=torch.float32, device=self.device, 
                              requires_grad=True)
        nn.init.normal_(feature_embeddings)
        self.feature_embeddings = nn.Parameter(feature_embeddings)
        
    def forward(self, feature_values, feature_idx):  
        embeddings = self.feature_embeddings[feature_idx, :]
        ## second order
        temp1 = torch.pow(torch.einsum('bf,bfk->bk', (feature_values, embeddings)), 2)
        temp2 = torch.einsum('bf,bfk->bk', (torch.pow(feature_values, 2), torch.pow(embeddings, 2)))
        second_order = temp1-temp2
        if self.get_embeddings:
            return second_order, embeddings
        else:
            return second_order


# In[ ]:


class FirstOrderMutiHot(nn.Module):
    '''support muti-hot feature for fm
    
    '''
    def __init__(self, params):
        super(FirstOrderMutiHot, self).__init__()
        # parse params
        self.device = params['device']
        self.feature_size = params['feature_size']
        self.field_size = params['field_size']
        self.fea_name = params['fea_name']
        self.max_len = params['max_len'] 
    
        weights_first_order = torch.empty(self.feature_size+2, 1, 
                                          dtype=torch.float32, device=self.device,
                                          requires_grad=True)
        nn.init.normal_(weights_first_order)
        self.weights_first_order = nn.Parameter(weights_first_order)
        
    def forward(self, feature_values, feature_idx):
        batch_size = feature_values.shape[0]
        
        # feature index and value padding
        feature_idx_concat, feature_values_concat = [], []
        for t in self.fea_name:
            feature_idx_concat = feature_idx_concat + list(feature_idx[t])
            feature_values_concat = feature_values_concat + list(feature_values[t])

        seqLen = torch.tensor(list(map(len, feature_idx_concat)), dtype=torch.float32, device=self.device)
        seqLen = torch.transpose(seqLen.reshape(self.field_size, batch_size), 0, 1)
        feature_idx_padded = pad_sequence(feature_idx_concat, batch_first=True, padding_value=self.feature_size)[:, 0:self.max_len].to(self.device)
        feature_values_padded = pad_sequence(feature_values_concat, batch_first=True, padding_value=0)[:, 0:self.max_len].to(self.device)
        
        # first_order
        weights_first_order = self.weights_first_order[feature_idx_padded]
        first_order = torch.mul(feature_values_padded, weights_first_order.squeeze())
        first_order = first_order.reshape(self.field_size, batch_size, -1)
        first_order = torch.transpose(first_order, 0, 1)
        first_order = first_order.sum(dim=2)
        first_order = first_order / seqLen
           
        return first_order


# In[ ]:


class SecondOrderMutiHot(nn.Module):
    '''support muti-hot feature for fm
    
    '''
    def __init__(self, params, get_embeddings=False):
        super(SecondOrderMutiHot, self).__init__()
        # parse params
        self.device = params['device']
        self.feature_size = params['feature_size']
        self.field_size = params['field_size']
        self.embedding_size = params['embedding_size']
        self.get_embeddings = get_embeddings
        self.fea_name = params['fea_name']
        self.max_len = params['max_len'] 

        feature_embeddings = torch.empty(self.feature_size+2, self.embedding_size, 
                              dtype=torch.float32, device=self.device, 
                              requires_grad=True)
        nn.init.normal_(feature_embeddings)
        self.feature_embeddings = nn.Parameter(feature_embeddings)
        
    def forward(self, feature_values, feature_idx):
        batch_size = feature_values.shape[0]
        
        # feature index padding and mask
        feature_idx_concat = []
        for t in self.fea_name:
            feature_idx_concat = feature_idx_concat + list(feature_idx[t])
        seqLen = torch.tensor(list(map(len, feature_idx_concat)), dtype=torch.float32, device=self.device)
        feature_idx_padded = pad_sequence(feature_idx_concat, batch_first=True, padding_value=self.feature_size)[:, 0:self.max_len].to(self.device)
        mask = feature_idx_padded != self.feature_size
        feature_weight = torch.ones_like(feature_idx_padded, dtype=torch.float32, device=self.device)
        feature_weight.masked_fill_(mask == 0, 0)
        
        # get embeddings and average
        embeddings = self.feature_embeddings[feature_idx_padded, :]
        embeddings = torch.einsum('ble,bl->ble', embeddings, feature_weight)
        embeddings = embeddings.sum(dim=1)
        embeddings = embeddings / seqLen.reshape(embeddings.shape[0], 1)
        embeddings = embeddings.reshape(self.field_size, batch_size, -1)
        embeddings = torch.transpose(embeddings, 0, 1)

        # feature values padding and average
        feature_values_concat = []
        for t in self.fea_name:
            feature_values_concat = feature_values_concat + list(feature_values[t])
        feature_values_padded = pad_sequence(feature_values_concat, batch_first=True, padding_value=0)[:, 0:self.max_len].to(self.device)
        feature_values_padded = feature_values_padded.reshape(self.field_size, batch_size, -1)
        feature_values_padded = torch.transpose(feature_values_padded, 0, 1)
        feature_values_padded = feature_values_padded.sum(dim=2)
        seqLen = torch.transpose(seqLen.reshape(self.field_size, batch_size), 0, 1)
        feature_values = feature_values_padded / seqLen

        # second order
        temp1 = torch.pow(torch.einsum('bf,bfk->bk', (feature_values, embeddings)), 2)
        temp2 = torch.einsum('bf,bfk->bk', (torch.pow(feature_values, 2), torch.pow(embeddings, 2)))
        second_order = temp1-temp2
        if self.get_embeddings:
            return second_order, embeddings
        else:
            return second_order


# In[ ]:


class MLP(nn.Module):
    def __init__(self, params, use_batchnorm=True, use_dropout=True):
        super(MLP, self).__init__()
        
        self.embedding_size = params['embedding_size']
        self.field_size = params['field_size']
        self.hidden_dims = params['hidden_dims']
        self.device = params['device']
        self.p = params['p']
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout


        
        self.input_dim = self.field_size * self.embedding_size
        self.num_layers = len(self.hidden_dims)

        
        ## deep weights
        self.deep_layers = nn.Sequential()

        net_dims = [self.input_dim]+self.hidden_dims
        for i in range(self.num_layers):
            self.deep_layers.add_module('fc%d' % (i+1), nn.Linear(net_dims[i], net_dims[i+1]).to(self.device))
            if self.use_batchnorm:
                self.deep_layers.add_module('bn%d' % (i+1), nn.BatchNorm1d(net_dims[i+1]).to(self.device))
            self.deep_layers.add_module('relu%d' % (i+1), nn.ReLU().to(self.device)) 
            if self.use_dropout:
                self.deep_layers.add_module('dropout%d' % (i+1), nn.Dropout(self.p).to(self.device))
    
    def forward(self, embeddings):
        deepInput = embeddings.reshape(embeddings.shape[0], self.input_dim)
        deepOut = self.deep_layers(deepInput)
        return deepOut


# In[ ]:


class CIN(nn.Module):
    '''xDeepFM CIN Module
    '''
    def __init__(self, params):
        super(CIN, self).__init__()
        # parse params
        self.split_half = params['split_half']
        self.field_size = params['field_size']
        self.hidden_dims = params['cin_hidden_dims']
        self.num_layers = len(self.hidden_dims)
        
        self.net_dims = [self.field_size]+self.hidden_dims
        self.hidden_dims_split_half = [self.field_size]
        self.conv1ds = nn.ModuleList()
        for i in range(self.num_layers):
#             h_weights['h_weight%d' % (i+1)] = torch.empty(net_dims[i], self.field_size)
#             nn.init.normal_(h_weights['h_weight%d' % (i+1)])
            self.conv1ds.append(nn.Conv1d(self.net_dims[0]*self.hidden_dims_split_half[-1], self.net_dims[i+1], 1))
            if self.split_half:
                self.hidden_dims_split_half.append(self.net_dims[i+1] // 2)
            else:
                self.hidden_dims_split_half.append(self.net_dims[i+1])
        
    def forward(self, inputs):
        res = []
        h = [inputs]
        for i in range(self.num_layers):
            temp = torch.einsum('bhd,bmd->bhmd', h[-1], h[0])
            temp = temp.reshape(inputs.shape[0], h[-1].shape[1]*inputs.shape[1], inputs.shape[2])
            # b * hi * d
            temp = self.conv1ds[i](temp)
            if self.split_half:
                next_hidden, hi = torch.split(temp, 2*[temp.shape[1]//2], 1)
            else:
                next_hidden, hi = temp, temp
            h.append(next_hidden)
            res.append(hi)
        res = torch.cat(res, dim=1)
        # b * (h1 + h2 + ... + hn)
        res = torch.sum(res, dim=2)
        return res

