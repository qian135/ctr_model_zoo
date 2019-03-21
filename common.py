
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn


# In[4]:


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


# In[5]:


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

