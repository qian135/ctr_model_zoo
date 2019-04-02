
# coding: utf-8

# In[ ]:


'''
Author:
    Shenxin Zhan,zhanshenxin135@163.com
    
Reference:
    https://arxiv.org/abs/1703.04247
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
'''


# In[1]:


import torch
import torch.nn as nn

from common import MLP


# In[2]:


class DeepFM(nn.Module):
    def __init__(self, params, get_embeddings=True, use_batchnorm=True, 
                 use_dropout=True, use_fm=True, use_deep=True):
        super(DeepFM, self).__init__()
        self.device = params['device']
        self.mlp_input_dim = params['field_size'] * params['embedding_size']
        self.use_fm = use_fm
        self.use_deep = use_deep

        self.first_order = FirstOrder(params)
        self.second_order = SecondOrder(params, get_embeddings=get_embeddings)   
        self.mlp = MLP(params, use_batchnorm=use_batchnorm, use_dropout=use_dropout)
                
        ## final concat layer
        if self.use_fm and self.use_deep:
            concat_size = params['field_size'] + params['embedding_size'] +                           params['hidden_dims'][-1]
        elif self.use_deep:
            concat_size = params['hidden_dims'][-1]
        elif self.use_fm:
            concat_size = params['field_size'] + params['embedding_size']
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
        deepInput = embeddings.reshape(embeddings.shape[0], self.mlp_input_dim)
        deepOut = self.mlp(deepInput)
        
        # concat layer
        if self.use_deep and self.use_fm:
            concat = torch.cat([first_order, second_order, deepOut], dim=1)
        elif self.use_deep:
            concat = deepOut
        elif self.use_fm:
            concat = torch.cat([first_order, second_order], dim=1)
                    
        logits = self.concat_layer(concat)
        
        return logits

