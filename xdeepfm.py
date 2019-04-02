
# coding: utf-8

# In[ ]:


'''
Author:
    Shenxin Zhan,zhanshenxin135@163.com
    
Reference:
    https://arxiv.org/abs/1803.05170
    xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems
'''


# In[ ]:


import torch
import torch.nn as nn

from common import MLP, CIN


# In[ ]:


class xDeepFM(nn.Module):
    
    def __init__(self, params, get_embeddings=True, use_batchnorm=True, 
                 use_dropout=True, use_fm_second_order=False):
        super(xDeepFM, self).__init__()
        self.device = params['device']
        self.mlp_input_dim = params['field_size'] * params['embedding_size']
        self.use_fm_second_order = use_fm_second_order
        
        self.first_order = FirstOrder(params)
        self.second_order = SecondOrder(params, get_embeddings=get_embeddings)
        self.mlp = MLP(params, use_batchnorm=use_batchnorm, use_dropout=use_dropout)
        self.cin = CIN(params)
        if params['split_half']:
            cinOutputSize = reduce(lambda x, y: x//2 + y//2, params['cin_hidden_dims'])
        else:
            cinOutputSize = reduce(lambda x, y: x + y, params['cin_hidden_dims'])
        if self.use_fm_second_order:
            concat_size = params['field_size'] + params['embedding_size'] +                           params['hidden_dims'][-1] + cinOutputSize
        else:
            concat_size = params['field_size'] +  params['hidden_dims'][-1] +                           cinOutputSize

        self.concat_layer = nn.Linear(concat_size, 1).to(self.device)


    def forward(self, features):
        # parse features
        feature_idx = features["feature_idx"]
        feature_values = features["feature_values"]
                     
        ## first order
        first_order = self.first_order(feature_values, feature_idx)
        
        ## second order
        second_order, embeddings = self.second_order(feature_values, feature_idx)
                
        # deep
        mlpInput = embeddings.reshape(embeddings.shape[0], self.mlp_input_dim)
        mlpOut = self.mlp(mlpInput)
        
        # cin
        cinOut = self.cin(embeddings)
        
        # concat layer
        if self.use_fm_second_order:
            concat = torch.cat([first_order, second_order, mlpOut, cinOut], dim=1) 
        else:
            concat = torch.cat([first_order, mlpOut, cinOut], dim=1) 
        logits = self.concat_layer(concat)
        
        return logits

