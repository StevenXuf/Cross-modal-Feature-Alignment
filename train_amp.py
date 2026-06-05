import torch
import argparse
import os
import time

import matplotlib.pyplot as plt
from tqdm import tqdm
from NN import MLP,BranchMLP

import torch.nn as nn
import torch.distributed as dist

from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader,random_split
from torch.utils.data.distributed import DistributedSampler

from sklearn.model_selection import train_test_split

class Train():
    def __init__(self,
            model_name,
            dataset,
            path='/project/home/p200248/alignment',
            loss_func='mse',
            lr=5e-5,
            batch_size=16,
            n_epochs=200,
            network='mlp',
            patience=5,
            use_amp=True
            ):
        self.model_name=model_name
        self.path=path+'/'+self.model_name.upper()
        self.loss_func=loss_func
        self.lr=lr
        self.n_epochs=n_epochs
        self.batch_size=batch_size
        self.network=network
        self.checkpoint_path=self.path+f'/checkpoints'
        self.patience=patience

        if self.network=='mlp':
            self.model=MLP()
        else:
            self.model=BranchMLP()
        
        self.dataset=dataset
        
        self.caption_embed=torch.load(self.path+f'/{self.model_name.lower()}_caption_embed_{self.dataset}.pt',mmap=True,map_location=torch.device('cuda')).half()
        self.img_embed=torch.load(self.path+f'/{self.model_name.lower()}_img_embed_{self.dataset}.pt',mmap=True,map_location=torch.device('cuda')).half()
        
        if self.dataset.lower()=='imdb':
            self.idx=torch.arange(0,self.caption_embed.size(0))
        else:
            self.idx=torch.randint(0,self.caption_embed.size(0),(20000,),generator=torch.Generator())
        
        self.caption_embed,self.img_embed=self.caption_embed[self.idx],self.img_embed[self.idx]

        if self.caption_embed.size(1)<768:
            self.caption_embed=torch.cat((self.caption_embed,torch.zeros(self.caption_embed.size(0),768-self.caption_embed.size(1),dtype=torch.float16).to('cuda')),dim=1)
            self.img_embed=torch.cat((self.img_embed,torch.zeros(self.img_embed.size(0),768-self.img_embed.size(1),dtype=torch.float16).to('cuda')),dim=1)


        '''if  self.dataset=='coco_train':
            self.caption_embed_train,self.caption_embed_test,self.img_embed_train,self.img_embed_test=train_test_split(self.caption_embed,self.img_embed,shuffle=False,test_size=.1)
            self.caption_embed_val=torch.load(self.path+f'/{self.model_name.lower()}_caption_embed_coco_val.pt',mmap=True,map_location=torch.device('cuda')).half()[torch.arange(0,2000)]
            self.img_embed_val=torch.load(self.path+f'/{self.model_name.lower()}_img_embed_coco_val.pt',mmap=True,map_location=torch.device('cuda')).half()[torch.range(0,2000)]
        else:'''
        
        self.caption_embed_train,self.caption_embed_test,self.img_embed_train,self.img_embed_test=train_test_split(self.caption_embed,self.img_embed,shuffle=False,test_size=.2)
        self.caption_embed_test,self.caption_embed_val,self.img_embed_test,self.img_embed_val=train_test_split(self.caption_embed_test,self.img_embed_test,shuffle=False,test_size=.5)

        #self.train_sim,self.val_sim=self.fetch_distance()
        
        if self.loss_func=='l1':
            self.criterion=nn.L1Loss()
        elif self.loss_func=='mse':
            self.criterion=nn.MSELoss()
        
        self.use_amp=use_amp

        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.lr)
        self.scaler=torch.cuda.amp.GradScaler(enabled=self.use_amp)

    @torch.no_grad
    def _pairwise_cosine_distance(self,mat_a,mat_b,device_id):
        mat_a,mat_b=mat_a.to(device_id),mat_b.to(device_id)
        res=torch.zeros(mat_a.size(0),mat_b.size(0),dtype=torch.float16).to(device_id)
        for i in range(mat_a.size(0)):
            sim=torch.sum(mat_a[i,:]*mat_b,dim=1)/torch.sqrt(torch.sum(mat_a[i,:]**2)*torch.sum(mat_b**2,dim=1))
            res[i,:]=sim.detach()
        return res

    def compute_batch_similarity(self,idx,batch_size,mat_a,mat_b,device_id):
        return self._pairwise_cosine_distance(mat_a[batch_size*idx:batch_size*(idx+1),:],mat_b,device_id)

    '''
    def fetch_distance(self):
        cosine_sim=torch.load(self.path+f'/{self.model_name.lower()}_{self.dataset}_distance.json')['cosine'].half()
        train_size=self.caption_embed_train.size(0)
        if self.dataset=='coco_train':
            return cosine_sim[:train_size,:train_size],torch.load(self.path+f'/{self.model_name.lower()}_coco_val_distance.json')['cosine'].half()
        else:
            val_size=self.caption_embed_val.size(0)
            return cosine_sim[:train_size,:train_size],cosine_sim[-val_size:,-val_size:]
    
    def fetch_batch_similarity(self,idx,batch_size,sim):
        return sim[batch_size*idx:batch_size*(idx+1),:]
'''
    
    def input_transform(self,batch_size,idx,mat_a,mat_b,device_id):
        mat_a=mat_a[batch_size*idx:batch_size*(idx+1),:].to(device_id)
        mat_b=mat_b.to(device_id)

        in_a=torch.repeat_interleave(
                mat_a,
                torch.tensor([mat_b.size(0)]*mat_a.size(0)).to(device_id),
                dim=0
                )
        mat_b=torch.tile(mat_b,(mat_a.size(0),1))
        return torch.cat((in_a,mat_b),dim=1)

    def dataloader(self,inputs):
        loader=DataLoader(
                inputs,
                batch_size=inputs.size(0),
                shuffle=False,
                sampler=DistributedSampler(inputs,shuffle=False)
                )
        return loader

    def compute_loss(self,
            i,
            batch_size,
            cap_embed,
            img_embed,
            #sim,
            device_id
            ):
        with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=self.use_amp):
            sim=self.compute_batch_similarity(i,batch_size,cap_embed,img_embed,device_id).flatten().reshape(-1,1)
            #sim=self.fetch_batch_similarity(i,batch_size,sim).flatten().reshape(-1,1)
            inputs=self.input_transform(batch_size,i,cap_embed,img_embed,device_id)
            loader=self.dataloader(torch.cat((inputs,sim),dim=1))
            for _,batch in enumerate(loader):
                targets=batch[:,-1]
                if self.network=='mlp':
                    outputs=self.model(batch[:,:-1].to(device_id))
                else:
                    outputs=self.model(batch[:,:768].to(device_id),batch[:,768:1536].to(device_id))
                #del batch
        #torch.cuda.empty_cache()
        loss=self.criterion(targets.to(device_id),outputs.flatten())
        return loss

    def save_model(self,epoch):
        torch.save(self.model.module.state_dict(),self.checkpoint_path+f'/{self.network}_{self.model_name}_{self.dataset}_epoch{epoch+1}.pt')

    def resume(self):
        pass

    @record
    def train_model(self):
        train_losses=[]
        val_losses=[]
        
        n_train_batch=torch.ceil(torch.tensor([self.caption_embed_train.size(0)/self.batch_size])).item()
        n_val_batch=torch.ceil(torch.tensor([self.caption_embed_val.size(0)/self.batch_size])).item()
        
        '''prof=torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                )
        prof.start()'''
        
        dist.init_process_group('nccl')
        rank=dist.get_rank()
        print(f'Running DDP on rank: {rank}.')

        device_id=rank%torch.cuda.device_count()
        self.model=self.model.to(device_id)
        self.model=DDP(self.model,device_ids=[device_id])
        
        last_loss=1e5
        patience=5
        trigger_count=0
        threshold=.1

        if device_id==0:
            print(f'Training {self.network} on {self.dataset} features extracted by {self.model_name}.')

        for epoch in range(self.n_epochs):
            running_loss=0.0
            
            self.model.train()
            
            for i in tqdm(range(int(n_train_batch))):
                self.optimizer.zero_grad()
                loss=self.compute_loss(i,self.batch_size,self.caption_embed_train,self.img_embed_train,device_id)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                running_loss+=loss.item()
            epoch_loss=running_loss/n_train_batch
            train_losses.append(epoch_loss)
            
            if os.path.exists(self.checkpoint_path)==False:
                os.makedirs(self.checkpoint_path)

            if (epoch+1)%1==0 and device_id==0:
                self.save_model(epoch)

            val_running_loss=0.0
            self.model.eval()
            with torch.no_grad():
                for j in range(int(n_val_batch)):
                    val_loss=self.compute_loss(j,self.batch_size,self.caption_embed_val,self.img_embed_val,device_id) 
                    #dist.all_reduce(val_loss,op=dist.ReduceOp.SUM)
                    val_running_loss+=val_loss.item()
                
                val_epoch_loss=val_running_loss/n_val_batch
                print(f'Epoch: {epoch+1}, training loss: {epoch_loss:.9f}, validation loss: {val_epoch_loss:.9f}.')
                val_losses.append(val_epoch_loss)
            
            if device_id==0 and (last_loss-val_epoch_loss)/last_loss<threshold:
                trigger_count+=1
                print(f'Trigger count: {trigger_count}')
                if trigger_count>=patience:
                    print('*'*5+'Early stopped at epoch {epoch+1}'+'*'*5)
                    self.save_model(epoch)
                    break
            else:
                trigger_count=0

            last_loss=val_epoch_loss

            #torch.cuda.empty_cache()
        
            #prof.step()
        #prof.stop()

        #torch.save(prof.key_averages(),self.path+'/new_profiler')
        
        if device_id==0:
            torch.save(train_losses,self.checkpoint_path+f'/{self.network}_{self.model_name}_{self.dataset}_training_loss.pt')
            torch.save(val_losses,self.checkpoint_path+f'/{self.network}_{self.model_name}_{self.dataset}_val_loss.pt')
        
        dist.destroy_process_group()

        return train_losses,val_losses

    def plot(self,train_losses,val_losses):
        plt.figure(figsize=(4,3))
        plt.plot(range(1,len(train_losses)+1),train_losses,c='b',label='training loss')
        plt.plot(range(1,len(val_losses)+1),val_losses,c='r',label='validation loss')
        minposs = val_losses.index(min(val_losses))+1
        plt.axvline(minposs, linestyle='--', color='r',label='early stopping checkpoint')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        plt.savefig(self.checkpoint_path+f'/{self.network}_{self.model_name}_{self.dataset}_losses.pdf',bbox_inches='tight')
        print(f'{self.model_name} {self.dataset} figure plotted!')
        
def set_config():
    parser=argparse.ArgumentParser()
    parser.add_argument('model_name',type=str,help='Multimodal name')
    parser.add_argument('dataset',type=str,help='which dataset')
    parser.add_argument('--network',type=str,default='mlp',help='which model structure')
    parser.add_argument('--n_epochs',type=int,default=200,help='number of epochs')
    parser.add_argument('--batch_size',type=int,default=16,help='batch size')
    parser.add_argument('--use_amp',type=bool,default=True,help='use mixed precision or not')
    parser.add_argument('--patience',type=int,default=5,help='patience for early stopping')
    args=parser.parse_args()
    return args

if __name__=='__main__':
    args=set_config()
    
    train=Train(args.model_name,args.dataset,
            network=args.network,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            patience=args.patience
            )
    train_loss,val_loss=train.train_model()
    train.plot(train_loss,val_loss)
