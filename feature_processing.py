import torch
import torch.nn as nn

from tqdm import tqdm
from NN import MLP

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchmetrics.functional.pairwise import pairwise_cosine_similarity

def process_features(model_name,dataset,img_retrieval,device='cpu'):
    path='/data/data_fxu/alignment_features/'+model_name.upper()
    caption_embed=torch.load(path+f'/{model_name.lower()}_caption_embed_{dataset}.pt',mmap=True,map_location=torch.device(device),weights_only=True)
    img_embed=torch.load(path+f'/{model_name.lower()}_img_embed_{dataset}.pt',mmap=True,map_location=torch.device(device),weights_only=True)

    if img_retrieval!='yes':
        caption_embed,img_embed=img_embed,caption_embed

    if dataset.lower()=='imdb':
        idx=torch.arange(0,caption_embed.size(0))
    else:
        idx=torch.randperm(caption_embed.size(0))[:20000]

    caption_embed,img_embed=caption_embed[idx],img_embed[idx]

    '''
    if caption_embed.size(1)<768:
        caption_embed=torch.cat((caption_embed,torch.zeros(caption_embed.size(0),768-caption_embed.size(1)).to(device)),dim=1)
        img_embed=torch.cat((img_embed,torch.zeros(img_embed.size(0),768-img_embed.size(1)).to(device)),dim=1)'''

    caption_embed_train,caption_embed_test,img_embed_train,img_embed_test=train_test_split(caption_embed,img_embed,shuffle=False,test_size=.2)
    caption_embed_test,caption_embed_val,img_embed_test,img_embed_val=train_test_split(caption_embed_test,img_embed_test,shuffle=False,test_size=.5)

    return caption_embed_train,caption_embed_test,caption_embed_val,img_embed_train,img_embed_test,img_embed_val

def input_transform(batch_size,idx,mat_a,mat_b,device):
    mat_a=mat_a[batch_size*idx:batch_size*(idx+1),:].to(device)
    mat_b=mat_b.to(device)

    in_a=torch.repeat_interleave(
            mat_a,
            torch.tensor([mat_b.size(0)]*mat_a.size(0)).to(device),
            dim=0
            )
    mat_b=torch.tile(mat_b,(mat_a.size(0),1))
    return torch.cat((in_a,mat_b),dim=1)

def compute_batch_similarity(idx,batch_size,mat_a,mat_b,device):
    return pairwise_cosine_similarity(mat_a[batch_size*idx:batch_size*(idx+1),:].to(device),mat_b.to(device))

def dataloader(inputs):
    loader=DataLoader(
            inputs,
            batch_size=inputs.size(0),
            shuffle=False,
            #sampler=DistributedSampler(inputs,shuffle=False)
            )
    return loader

def contrastive_loss_for_cosine(outputs,indicator):
    loss=torch.sum(indicator*(1-outputs)**2+(1-indicator)*(1+outputs)**2,dim=-1)/(2*len(indicator))
    return loss

def compute_loss(
        i,
        batch_size,
        cap_embed,
        img_embed,
        device,
        loss_func,
        model
        ):
    inputs=input_transform(batch_size,i,cap_embed,img_embed,device)

    if loss_func=='contrastive':
        nrow=cap_embed[batch_size*i:batch_size*(i+1)].size(0)
        indicator=torch.zeros(nrow,img_embed.size(0)).to(device)
        for j in range(nrow):
            indicator[j,batch_size*i+j]=1
        indicator=indicator.flatten().reshape(-1,1)
        loader=dataloader(torch.cat((inputs,indicator),dim=1))
    else:
        sim=compute_batch_similarity(i,batch_size,cap_embed,img_embed,device)
        sim=sim.flatten().reshape(-1,1)
        loader=dataloader(torch.cat((inputs,sim),dim=1))

    for _,batch in enumerate(loader):
        batch=batch.to(device)
        outputs=model(batch[:,:-1])

        if loss_func=='contrastive':
            loss=contrastive_loss_for_cosine(outputs.flatten(),batch[:,-1])
        else:
            criterion=nn.MSELoss()
            loss=criterion(outputs.flatten(),batch[:,-1])

    return loss

if __name__=='__main__':
    torch.manual_seed(0)
    model_name='clip'
    dataset='imdb'
    img_retrieval='yes'
    batch_size=64
    
    cap_train,cap_test,cap_val,img_train,img_test,img_val=process_features(model_name,dataset,img_retrieval)
    
    assert cap_train.size()==img_train.size()

    n_batches=cap_train.size(0)//batch_size if cap_train.size(0)%batch_size==0 else cap_train.size(0)//batch_size+1

    for i in tqdm(range(n_batches)):
        if i>=10:
            break
        else:
            loss=compute_loss(i,batch_size,cap_train,img_train,'cpu','mse',MLP())
