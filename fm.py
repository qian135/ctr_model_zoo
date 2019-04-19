
# coding: utf-8

# In[ ]:


'''
Author:
    Shenxin Zhan,zhanshenxin135@163.com
    
Reference:
    https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
    
    Factorization Machines
'''


# In[10]:


import torch
import torch.nn as nn

from common import FirstOrder, SecondOrder


# In[11]:


class FM(nn.Module):
    '''implement Factorization Machine by PyTorch
    
    '''
    def __init__(self, params):
        super(FMModel, self).__init__()
        
        # parse params
        self.embedding_size = params['embedding_size']
        self.feature_size = params['feature_size']
        self.device = params['device']

        self.first_order = FirstOrder({'device': self.device, 
                                       'feature_size': self.feature_size})
        self.second_order = SecondOrder({'device': self.device, 
                                         'feature_size': self.feature_size,
                                         'embedding_size': self.embedding_size})
        
        fm_bias = torch.empty(1, dtype=torch.float32, device=self.device, 
                              requires_grad=True)
        nn.init.constant_(fm_bias, 0)
        self.fm_bias = nn.Parameter(fm_bias)
        
    def forward(self, features):
        feature_idx = features["feature_idx"]
        feature_values = features["feature_values"]
        
        bias = self.fm_bias

        first_order = self.first_order(feature_values, feature_idx)
        first_order = torch.sum(first_order, dim=1)

        second_order = self.second_order(feature_values, feature_idx)
        second_order = torch.sum(second_order, dim=1)
                
        logits = second_order + first_order + bias

        return logits

