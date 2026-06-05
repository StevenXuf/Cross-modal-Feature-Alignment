import torch
import argparse
import os
import sys

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from torch.utils.data import DataLoader,random_split
from accelerate import Accelerator
from accelerate.utils import broadcast

from feature_extractor import FeatureExtraction
from NN import MLP
from loss_module import CustomContrastiveLoss,InfoNCE
from load_data import LoadCOCO,LoadFlickr30K,LoadIMDB

class Train():
    def __init__(self,
            extractor_name,
            dataset,
            in_size,
            loss_func='mse',
            lr=5e-5,
            batch_size=256,
            n_epochs=100,
            patience=5,
            tolerance=.01,
            img_retrieval='yes',
            early_stopping=True,
            path='/data/data_fxu/alignment_features_journal_paper/'
            ):
        self.accelerator=Accelerator()
        self.extractor_name=extractor_name
        self.path=path+'/'+self.extractor_name.upper()
        self.loss_func=loss_func
        self.lr=lr
        self.n_epochs=n_epochs
        self.batch_size=batch_size
        self.in_size=in_size
        self.checkpoint_path=self.path+f'/model_checkpoints'
        self.patience=patience
        self.tolerance=tolerance
        self.early_stopping=early_stopping
        self.img_retrieval=img_retrieval
        self.device=self.accelerator.device

        if dataset.lower()=='imdb':
            self.dataset=LoadIMDB().get_dataset()
        elif dataset.lower()=='flickr30k':
            self.dataset=LoadFlickr30K().get_dataset()
        elif dataset.lower()=='coco':
            self.dataset,_=LoadCOCO().get_dataset()

        self.train_loader,self.val_loader,self.test_loader=list(map(lambda x:DataLoader(x,batch_size=self.batch_size,shuffle=True),random_split(self.dataset,[.8,.1,.1],generator=torch.Generator().manual_seed(42))))
        
        self.model=MLP(in_size=in_size).to(self.device)
        
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.lr)
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda=lambda epoch: 1 / (2**(.5*epoch) + 1))    
        
        self.model,self.optimizer,self.train_loader,self.val_loader=self.accelerator.prepare(self.model,self.optimizer,self.train_loader,self.val_loader)

    def save_model(self,epoch,model_ckpt):
        torch.save(model_ckpt.state_dict(),self.checkpoint_path+f'/{self.extractor_name}_{self.dataset.name}_{self.loss_func}_epoch{epoch+1}_{self.img_retrieval}.pt')
    
    def save_loss(self,train_loss,val_loss,epoch):
        torch.save(train_loss,self.checkpoint_path+f'/{self.extractor_name}_{self.dataset.name}_{self.loss_func}_epoch{epoch+1}_{self.img_retrieval}_training_loss.pt')
        torch.save(val_loss,self.checkpoint_path+f'/{self.extractor_name}_{self.dataset.name}_{self.loss_func}_epoch{epoch+1}_{self.img_retrieval}_val_loss.pt')
    
    def load_init_info(self,model):
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
        
        plt.savefig(self.checkpoint_path+f'/{self.extractor_name}_{self.dataset.name}_{self.loss_func}_{self.img_retrieval}_loss_curves.pdf',bbox_inches='tight')
        print(f'{self.extractor_name} {self.dataset.name} figure plotted!')
    
    def train_model(self):
        train_losses=[]
        val_losses=[]
        
        if os.path.exists(self.checkpoint_path)==False:
            os.makedirs(self.checkpoint_path)

        last_loss=1e5
        trigger_count=0
        
        for epoch in range(self.n_epochs):
            running_loss=0.0
            
            self.model.train()
            
            for i,batch in enumerate(tqdm(self.train_loader)):
                images=batch['image']
                texts=batch['text']
                
                self.optimizer.zero_grad()
                img_feats,txt_feats=FeatureExtraction(self.extractor_name,self.device).get_features(images,texts)
                img_feats=F.normalize(img_feats)
                txt_feats=F.normalize(txt_feats)

                if self.img_retrieval=='yes':
                    img_feats,txt_feats=txt_feats,img_feats
                
                img_feat_repeat=img_feats.repeat_interleave(txt_feats.size(0),dim=0)
                txt_feat_repeat=txt_feats.repeat(img_feats.size(0),1)
                inputs=torch.cat((img_feat_repeat,txt_feat_repeat),dim=1).to(self.device)
                preds=self.model(inputs)
                
                if self.loss_func=='contrastive': 
                    loss=CustomContrastiveLoss(temperature=1)(preds.reshape(img_feats.size(0),txt_feats.size(0)))
                else:
                    tars=pairwise_cosine_similarity(img_feats,txt_feats).flatten()
                    loss=nn.MSELoss()(preds,tars)

                self.accelerator.backward(loss)
                self.optimizer.step()

                running_loss+=loss.item()
            
            epoch_loss=torch.tensor(running_loss/(i+1),device=self.accelerator.device)
            gathered_e_train_loss=self.accelerator.gather(epoch_loss)
            
            train_losses.append(gathered_e_train_loss.mean().item())

            val_running_loss=0.0
            self.model.eval()

            with torch.no_grad():
                for j,batch_v in enumerate(self.val_loader):
                    img_feats,txt_feats=FeatureExtraction(self.extractor_name,self.device).get_features(batch_v['image'],batch_v['text'])
                    img_feats=F.normalize(img_feats)
                    txt_feats=F.normalize(txt_feats)

                    if self.img_retrieval=='yes':
                        img_feats,txt_feats=txt_feats,img_feats

                    img_feat_repeat=img_feats.repeat_interleave(txt_feats.size(0),dim=0)
                    txt_feat_repeat=txt_feats.repeat(img_feats.size(0),1)
                    inputs=torch.cat((img_feat_repeat,txt_feat_repeat),dim=1).to(self.device)
                    preds=self.model(inputs)

                    if self.loss_func=='contrastive':
                        val_loss=CustomContrastiveLoss(temperature=1)(preds.reshape(img_feats.size(0),txt_feats.size(0)))
                    else:
                        tars=pairwise_cosine_similarity(img_feats,txt_feats).flatten()
                        val_loss=nn.MSELoss()(preds,tars) 

                    val_running_loss+=val_loss.item()
                
                val_epoch_loss=torch.tensor(val_running_loss/(j+1),device=self.accelerator.device)
                gathered_val_e_loss=self.accelerator.gather(val_epoch_loss)
                
                val_losses.append(gathered_val_e_loss.mean().item())

            self.scheduler.step()
            self.accelerator.print(f'Epoch: {epoch+1}, training loss: {epoch_loss.item():.9f}, validation loss: {val_epoch_loss.item():.9f}.')
            
            if (epoch+1)%5==0 and self.accelerator.is_main_process:
                model_ckpt=self.accelerator.unwrap_model(self.model)
                self.save_model(epoch,model_ckpt)
                self.save_loss(train_losses,val_losses,epoch)
            
            #early epoch for improvement less than 1% for 5 consecutive epochs
            flag=torch.tensor(0) 
            if self.early_stopping==True:
                if (last_loss-val_epoch_loss)/last_loss<self.tolerance:
                    trigger_count+=1
                    print(f'Trigger count: {trigger_count}')
                    if trigger_count>=self.patience:
                        print('*'*5+f'Early stopped at epoch {epoch+1}'+'*'*5)
                        flag=torch.tensor(1)
                        broadcast(flag,from_process=self.accelerator.device)
                else:
                    trigger_count=0
                last_loss=val_epoch_loss

            self.accelerator.wait_for_everyone() 
            if flag.item==1:
                break

        if self.accelerator.is_main_process:
            model_ckpt=self.accelerator.unwrap_model(self.model)
            self.save_model(epoch,model_ckpt)
            self.save_loss(train_losses,val_losses,epoch)
            self.plot(train_losses,val_losses)
        
        self.accelerator.print('Training complete.')

def set_config():
    parser=argparse.ArgumentParser()
    parser.add_argument('extractor_name',type=str,help='Multimodal name')
    parser.add_argument('dataset',type=str,help='which dataset')
    parser.add_argument('in_size',type=int,help='which model structure')
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
    
    train=Train(args.extractor_name,args.dataset,
            in_size=args.in_size,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            patience=args.patience,
            tolerance=args.tolerance,
            early_stopping=args.early_stopping,
            img_retrieval=args.img_retrieval,
            loss_func=args.loss_func
            )
    train.train_model()
