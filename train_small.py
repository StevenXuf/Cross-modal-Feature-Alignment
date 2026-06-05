import torch
import argparse
import os

import matplotlib.pyplot as plt
from tqdm import tqdm
from NN import MLP

import torch.nn as nn
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import torch.distributed as dist

from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from sklearn.model_selection import train_test_split

class Train():
    def __init__(self,
            model_name,
            dataset,
            path='/data/data_fxu/alignment_features/',
            loss_func='mse',
            lr=5e-5,
            batch_size=128,
            n_epochs=100,
            network='mlp',
            hidden_sizes=[500,500],
            patience=5,
            tolerance=.01,
            early_stopping=True,
            img_retrieval='yes'
            ):
        self.model_name=model_name
        self.path=path+'/'+self.model_name.upper()
        self.loss_func=loss_func
        self.lr=lr
        self.n_epochs=n_epochs
        self.batch_size=batch_size
        self.network=network
        self.checkpoint_path=self.path+f'/model_checkpoints'
        self.patience=patience
        self.tolerance=tolerance
        self.early_stopping=early_stopping
        self.img_retrieval=img_retrieval
        
        self.dataset=dataset
        
        self.caption_embed=torch.load(self.path+f'/{self.model_name.lower()}_caption_embed_{self.dataset}.pt',mmap=True,map_location=torch.device('cuda'))
        self.img_embed=torch.load(self.path+f'/{self.model_name.lower()}_img_embed_{self.dataset}.pt',mmap=True,map_location=torch.device('cuda'))
        
        if self.img_retrieval!='yes':
            self.caption_embed,self.img_embed=self.img_embed,self.caption_embed
        
        subsize=1000
        self.idx=torch.randperm(self.caption_embed.size(0))[:subsize]
        
        self.caption_embed,self.img_embed=self.caption_embed[self.idx],self.img_embed[self.idx]
        '''
        if self.caption_embed.size(1)<768:
            self.caption_embed=torch.cat((self.caption_embed,torch.zeros(self.caption_embed.size(0),768-self.caption_embed.size(1)).to('cuda')),dim=1)
            self.img_embed=torch.cat((self.img_embed,torch.zeros(self.img_embed.size(0),768-self.img_embed.size(1)).to('cuda')),dim=1)
        '''
        self.caption_embed_train,self.caption_embed_test,self.img_embed_train,self.img_embed_test=train_test_split(self.caption_embed,self.img_embed,shuffle=False,test_size=.2)
        self.caption_embed_test,self.caption_embed_val,self.img_embed_test,self.img_embed_val=train_test_split(self.caption_embed_test,self.img_embed_test,shuffle=False,test_size=.5)
        
        self.model=MLP(in_size=self.caption_embed_train.size(1)+self.img_embed_train.size(1),hidden_sizes=hidden_sizes)

        #self.train_sim,self.val_sim=self.fetch_distance()
        
        if self.loss_func=='l1':
            self.criterion=nn.L1Loss()
        elif self.loss_func=='mse':
            self.criterion=nn.MSELoss()
        
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.lr)
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda=lambda epoch: 1 / (2**(.5*epoch) + 1))    

    def compute_batch_similarity(self,idx,batch_size,mat_a,mat_b,device_id):
        return pairwise_cosine_similarity(mat_a[batch_size*idx:batch_size*(idx+1),:].to(device_id),mat_b.to(device_id))

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
        sim=self.compute_batch_similarity(i,batch_size,cap_embed,img_embed,device_id)
        sim=sim.flatten().reshape(-1,1)

        if self.loss_func=='contrastive':
            nrow=cap_embed[batch_size*i:batch_size*(i+1)].size(0)
            indicator=torch.zeros(nrow,img_embed.size(0)).to(device_id)
            for j in range(nrow):
                indicator[j,batch_size*i+j]=1
            indicator=indicator.flatten().reshape(-1,1)

        inputs=self.input_transform(batch_size,i,cap_embed,img_embed,device_id)
        
        if self.loss_func=='contrastive':
            loader=self.dataloader(torch.cat((inputs,indicator),dim=1))
        else:
            loader=self.dataloader(torch.cat((inputs,sim),dim=1))

        for _,batch in enumerate(loader):
            batch=batch.to(device_id)
            outputs=self.model(batch[:,:-1])
            
            if self.loss_func=='contrastive':
                loss=self.contrastive_loss_for_cosine(outputs.flatten(),batch[:,-1])
            else:
                loss=self.criterion(outputs.flatten(),batch[:,-1])

        return loss
        
    def contrastive_loss_for_cosine(self,outputs,indicator):
        loss=torch.sum(indicator*(1-outputs)**2+(1-indicator)*(1+outputs)**2,dim=-1)/(2*len(indicator))
        return loss
    
    def contrastive_loss_for_entropy(self,sim,temperature):
        rows,cols=sim.size()



    def save_model(self,epoch):
        torch.save(self.model.module.state_dict(),self.checkpoint_path+f'/{self.network}_{self.model_name}_{self.dataset}_{self.loss_func}_epoch{epoch+1}_{self.img_retrieval}_small.pt')
    
    def save_loss(self,train_loss,val_loss,epoch):
        torch.save(train_loss,self.checkpoint_path+f'/{self.network}_{self.model_name}_{self.dataset}_{self.loss_func}_epoch{epoch+1}_{self.img_retrieval}_training_loss_small.pt')
        torch.save(val_loss,self.checkpoint_path+f'/{self.network}_{self.model_name}_{self.dataset}_{self.loss_func}_epoch{epoch+1}_{self.img_retrieval}_val_loss_small.pt')

    def resume(self):
        pass

    def plot(self,train_losses,val_losses):
        fig=plt.figure(figsize=(4,3))
        ax=fig.add_subplot(111)
        plt.plot(range(1,len(train_losses)+1),train_losses,c='b',label='training loss')
        plt.plot(range(1,len(val_losses)+1),val_losses,c='r',label='validation loss')
        minposs=len(val_losses)
        plt.axvline(minposs, linestyle='--', color='r',label='early stopping checkpoint')
        plt.text(0.8,0.15,f'{minposs}',c='r',transform = ax.transAxes)
        #plt.xticks(ticks=range(1,len(train_losses)+1))
        plt.xlabel('epochs')
        plt.ylabel('loss')

        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        
        plt.savefig(self.checkpoint_path+f'/{self.network}_{self.model_name}_{self.dataset}_{self.loss_func}_{self.img_retrieval}_loss_curves_small.pdf',bbox_inches='tight')
        print(f'{self.model_name} {self.dataset} figure plotted!')
    
    @record
    def train_model(self):
        dist.init_process_group('nccl')

        train_losses=[]
        val_losses=[]
        
        n_train_batch=torch.ceil(torch.tensor([self.caption_embed_train.size(0)/self.batch_size])).item()
        n_val_batch=torch.ceil(torch.tensor([self.caption_embed_val.size(0)/self.batch_size])).item()
        
        
        rank=dist.get_rank()
        print(f'Running DDP on rank: {rank}.')
        
        if os.path.exists(self.checkpoint_path)==False and rank==0:
            os.makedirs(self.checkpoint_path)

        device_id=rank%torch.cuda.device_count()
        self.model=self.model.to(device_id)
        self.model=DDP(self.model,device_ids=[device_id])
        
        last_loss=1e5
        trigger_count=0

        if rank==0:
            print(f'Training {self.network} on {self.dataset} features extracted by {self.model_name}.')
        
        break_flag=torch.tensor([0]).to(device_id) ###
        for epoch in range(self.n_epochs):
            running_loss=0.0
            
            self.model.train()
            
            for i in tqdm(range(int(n_train_batch))):
                self.optimizer.zero_grad()
                loss=self.compute_loss(i,self.batch_size,self.caption_embed_train,self.img_embed_train,device_id)
                loss.backward()
                self.optimizer.step()

                running_loss+=loss.item()
            epoch_loss=running_loss/n_train_batch
            
            train_losses.append(epoch_loss)

            val_running_loss=0.0
            self.model.eval()
            with torch.no_grad():
                for j in range(int(n_val_batch)):
                    val_loss=self.compute_loss(j,self.batch_size,self.caption_embed_val,self.img_embed_val,device_id) 
                    val_running_loss+=val_loss.item()
                
                val_epoch_loss=val_running_loss/n_val_batch
                print(f'Epoch: {epoch+1}, training loss: {epoch_loss:.9f}, validation loss: {val_epoch_loss:.9f}.')
                
                val_losses.append(val_epoch_loss)

            self.scheduler.step()

            if (epoch+1)%5==0 and rank==0:
                self.save_model(epoch)
                self.save_loss(train_losses,val_losses,epoch)
            
            #early epoch for improvement less than 1% for 5 consecutive epochs
            if self.early_stopping==True:
                if (last_loss-val_epoch_loss)/last_loss<self.tolerance and rank==0:
                    trigger_count+=1
                    print(f'Trigger count: {trigger_count}')
                    if trigger_count>=self.patience:
                        print('*'*5+f'Early stopped at epoch {epoch+1}'+'*'*5)
                        self.save_model(epoch)
                        self.save_loss(train_losses,val_losses,epoch)
                        break_flag=torch.tensor([1]).to(device_id)
                else:
                    trigger_count=0

                last_loss=val_epoch_loss
            
            dist.broadcast(break_flag,0) #dist functions should be execuated bu all processes.
            if break_flag.item()!=0:
                break

        if rank==0:
            self.plot(train_losses,val_losses)
        
        dist.destroy_process_group()

def set_config():
    parser=argparse.ArgumentParser()
    parser.add_argument('model_name',type=str,help='Multimodal name')
    parser.add_argument('dataset',type=str,help='which dataset')
    parser.add_argument('hidden_sizes',type=int,nargs='+',help='number of hidden nuerons')
    parser.add_argument('--network',type=str,default='mlp',help='which model structure')
    parser.add_argument('--n_epochs',type=int,default=100,help='number of epochs')
    parser.add_argument('--batch_size',type=int,default=128,help='batch size')
    parser.add_argument('--patience',type=int,default=5,help='patience for early stopping')
    parser.add_argument('--tolerance',type=float,default=.01,help='improvement for each epoch')
    parser.add_argument('--early_stopping',type=bool,default=True)
    parser.add_argument('--img_retrieval',type=str,default='yes')
    parser.add_argument('--loss_func',type=str,default='mse')
    args=parser.parse_args()
    return args

if __name__=='__main__':
    torch.manual_seed(0)

    args=set_config()
    
    print(args.hidden_sizes)

    train=Train(args.model_name,args.dataset,
            network=args.network,
            hidden_sizes=args.hidden_sizes,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            patience=args.patience,
            tolerance=args.tolerance,
            early_stopping=args.early_stopping,
            img_retrieval=args.img_retrieval,
            loss_func=args.loss_func
            )
    train.train_model()
