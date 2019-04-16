
# coding: utf-8

# In[ ]:


'''
Author:
    Shenxin Zhan,zhanshenxin135@163.com
    
Reference:
    https://arxiv.org/abs/1706.06978
    Deep Interest Network for Click-Through Rate Prediction
'''


# In[ ]:


import torch
import torch.nn as nn

from common import MLP


# In[ ]:


class DIN(nn.Module):
    def __init__(self, params, use_batchnorm=True, use_dropout=True):
        super(DIN, self).__init__()
        
        self.device = params['device']
        self.feature_size = params['feature_size']
        self.embedding_size = params['embedding_size']
        self.userItemDict = params['userItemDict']
        self.hidden_dims = params['hidden_dims']
        self.userItemMaxLen = params['userItemMaxLen']
        
        feature_embeddings = torch.empty(self.feature_size+1, self.embedding_size, 
                              dtype=torch.float32, device=self.device, 
                              requires_grad=True)
        nn.init.normal_(feature_embeddings)
        self.feature_embeddings = nn.Parameter(feature_embeddings)
        
        self.mlp = MLP(params, use_batchnorm=use_batchnorm, use_dropout=use_dropout)
        
        self.output_layer = nn.Linear(self.hidden_dims[-1], 1).to(self.device)

 
    def forward(self, features):
        feature_idx = features["feature_idx"]
        
        uid = feature_idx[:, 0]
        feaName_ix = [('item_id', 2), ('author_id', 3), ('music_id', 6)]
        feaName_maxlen = [('item_id', 350), ('author_id', 250), ('music_id', 100)]
        feaName = ['item_id', 'author_id', 'music_id']
        ad_idx = {}
        for t in feaName_ix:
            ad_idx[t[0]] = feature_idx[:, t[1]]
            

        hist_idx = self.userItemDict.loc[uid.cpu().numpy()][feaName]
    
        hist_idx_padded = {}
        for temp in feaName_maxlen:
            hist_idx_padded[temp[0]] = pad_sequence(list(hist_idx[temp[0]]), batch_first=True, 
                                                 padding_value=self.feature_size)[:, 0:temp[1]].to(self.device)
        user_beha_embeddings = []
        for temp in feaName:
            hist_embeddings = self.feature_embeddings[hist_idx_padded[temp], :] 
            ad_embeddings = self.feature_embeddings[ad_idx[temp], :]
            
            hist_weight = torch.einsum('blk,bk->bl', (hist_embeddings, ad_embeddings))
            mask = hist_idx_padded[temp] != self.feature_size
            hist_weight.masked_fill_(mask == 0, -1e9)
            hist_weight = torch.softmax(hist_weight, dim=1)
            user_beha_embeddings.append(torch.einsum('blk,bl->bk', (hist_embeddings, hist_weight)))
        
        user_beha_embeddings = torch.cat(user_beha_embeddings, dim=1)
        
        ad_embeddings = self.feature_embeddings[feature_idx, :].reshape(feature_idx.shape[0], -1)
        
        embeddings = torch.cat((user_beha_embeddings, ad_embeddings), dim=1)
        
        # deep
#         deepInput = embeddings.reshape(embeddings.shape[0], self.mlp_input_dim)
        deepOut = self.mlp(embeddings)
        
        logits = self.output_layer(deepOut)
        
        return logits

