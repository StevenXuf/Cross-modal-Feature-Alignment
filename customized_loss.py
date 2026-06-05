import torch
import torch.nn as nn

def feature_loss(ref_feature,in_feature):
    len_ref=torch.linalg.norm(ref_feature,dim=-1,ord=2)
    len_in=torch.linalg.norm(in_feature,dim=-1,ord=2)
    cosine=nn.CosineSimilarity(dim=-1)
    cos_val=cosine(ref_feature,in_feature)
    sin_val=torch.sqrt(1-cos_val**2)
    loss=len_in*sin_val+torch.abs(len_in*cos_val-len_ref)
    prob=1/torch.exp(loss)
    return prob

