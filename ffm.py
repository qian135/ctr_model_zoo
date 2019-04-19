
# coding: utf-8

# In[ ]:


'''
Author:
    Shenxin Zhan,zhanshenxin135@163.com
    
Reference:
    https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf
    
    Field-aware Factorization Machines for CTR Prediction
'''


# In[ ]:


import torch
import torch.nn as nn


# In[ ]:


class FFM(nn.Module):
    def __init__(self, params):
        super(FFM, self).__init__()
        
        # parse params
        self.embedding_size = params['embedding_size']
        self.field_size = params['field_size']
        self.feature_size = params['feature_size']
        self.device = params['device']
                    
        feature_embeddings = torch.empty(self.feature_size, self.field_size, self.embedding_size, 
                                      dtype=torch.float32, device=self.device, 
                                      requires_grad=True)
        nn.init.normal_(feature_embeddings)
        self.feature_embeddings = nn.Parameter(feature_embeddings)
    
        weights_first_order = torch.empty(self.feature_size, 1, 
                                          dtype=torch.float32, device=self.device,
                                          requires_grad=True)
        nn.init.normal_(weights_first_order)
        self.weights_first_order = nn.Parameter(weights_first_order)
        
        fm_bias = torch.empty(1, dtype=torch.float32, device=self.device, 
                              requires_grad=True)
        nn.init.constant_(fm_bias, 0)
        self.fm_bias = nn.Parameter(fm_bias)
        
    def forward(self, features):
        # parse features
        feature_idx = features["feature_idx"]
        feature_values = features["feature_values"]
        
        bias = self.fm_bias

        weights_first_order = self.weights_first_order[feature_idx].squeeze()
        first_order = torch.mul(feature_values, weights_first_order)
        first_order = torch.sum(first_order, dim=1, keepdim=True)
        
        second_order = torch.tensor([[0]]*feature_idx.shape[0], dtype=torch.float32, device=self.device)
        for i in range(self.field_size):
            for j in range(i+1, self.field_size):
                vifj = self.feature_embeddings[feature_idx[:, i], torch.tensor([j], device=self.device), :]
                vjfi = self.feature_embeddings[feature_idx[:, j], torch.tensor([i], device=self.device), :]
                second_order += torch.sum(torch.mul(vifj, vjfi), dim=1, keepdim=True) *                             feature_values[:, i][:, np.newaxis] *                             feature_values[:, j][:, np.newaxis]
        
        logits = second_order + first_order + bias
                    
        return logits

