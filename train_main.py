import torch
import argparse
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchmetrics.retrieval import RetrievalRecall

from NN import MLP, NewMLP, CrossAttentionSimilarity
from contrastive import CustomContrastiveLossBalanced as CustomContrastiveLoss

class Train():
    def __init__(self,
            model_name,
            dataset,
            path='/data/data_fxu/alignment_features/',
            loss_func='mse',
            lr=5e-5,
            batch_size=64,
            n_epochs=100,
            network='mlp',
            patience=10,
            tolerance=.01,
            early_stopping=False,
            task='txt2img',
            gpu_id=0
            ):
        self.model_name=model_name
        self.path=path+'/'+self.model_name.upper()
        self.loss_func=loss_func
        self.lr=lr
        self.n_epochs=n_epochs
        self.batch_size=batch_size
        self.network=network
        self.checkpoint_path=self.path+'/model_checkpoints'
        self.patience=patience
        self.tolerance=tolerance
        self.early_stopping=early_stopping
        self.task=task
        
        self.dataset=dataset
        self.device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
        
        self.caption_embed=torch.load(self.path+f'/{self.model_name.lower()}_caption_embed_{self.dataset}.pt',mmap=True,map_location=torch.device('cpu'),weights_only=True)
        self.img_embed=torch.load(self.path+f'/{self.model_name.lower()}_img_embed_{self.dataset}.pt',mmap=True,map_location=torch.device('cpu'),weights_only=True)
        print(f"Emebedding shapes: {self.caption_embed.shape}.")

        if self.task=='img2txt':
            self.caption_embed,self.img_embed=self.img_embed,self.caption_embed

        if self.dataset.lower()=='imdb':
            self.idx=torch.arange(0,self.caption_embed.size(0))
        else:
            self.idx=torch.randperm(self.caption_embed.size(0))[:20000]
        
        self.caption_embed,self.img_embed = self.caption_embed[self.idx],self.img_embed[self.idx]

        self.caption_embed_train,self.caption_embed_test,self.img_embed_train,self.img_embed_test=train_test_split(self.caption_embed,self.img_embed,shuffle=False,test_size=.2)
        self.caption_embed_test,self.caption_embed_val,self.img_embed_test,self.img_embed_val=train_test_split(self.caption_embed_test,self.img_embed_test,shuffle=False,test_size=.5)
        self.train_loader=self.dataloader(torch.cat((self.caption_embed_train,self.img_embed_train),dim=1))
        self.val_loader=self.dataloader(torch.cat((self.caption_embed_val,self.img_embed_val),dim=1))
        self.test_loader=self.dataloader(torch.cat((self.caption_embed_test,self.img_embed_test),dim=1))
        
        if self.network=='mlp':
            self.model=MLP(in_size=self.caption_embed_train.size(1),hidden_sizes=[500,500])
        elif self.network=='new_mlp':
            self.model=NewMLP(in_size=self.caption_embed_train.size(1),hidden_sizes=[512,256])
        elif self.network=='transformer':
            self.model=CrossAttentionSimilarity(in_size=self.caption_embed_train.size(1))
        else:
            raise ValueError('Unsupported network type. Choose from "mlp", "new_mlp" or "cross_attention".')

        if self.loss_func=='contrastive':
            self.criterion=CustomContrastiveLoss()
        elif self.loss_func=='mse':
            self.criterion=nn.MSELoss()
        
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.lr)
        self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=self.n_epochs,eta_min=1e-7)   

    def dataloader(self,inputs):
        loader=DataLoader(
                inputs,
                batch_size=self.batch_size,
                shuffle=True,
                )
        return loader
    
    def save_model(self,epoch, store_path):
        torch.save(self.model.state_dict(),store_path+f'/{self.network}_{self.model_name}_{self.dataset}_{self.loss_func}_epoch{epoch+1}_{self.task}.pt')
    
    def save_loss(self,train_loss,val_loss,epoch, store_path):
        torch.save(train_loss,store_path+f'/{self.network}_{self.model_name}_{self.dataset}_{self.loss_func}_epoch{epoch+1}_{self.task}_training_loss.pt')
        torch.save(val_loss,store_path+f'/{self.network}_{self.model_name}_{self.dataset}_{self.loss_func}_epoch{epoch+1}_{self.task}_val_loss.pt')

    def plot(self,train_losses,val_losses, store_path):
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
        
        plt.savefig(store_path+f'/{self.network}_{self.model_name}_{self.dataset}_{self.loss_func}_{self.task}_loss_curves.pdf',bbox_inches='tight')
        print(f'{self.model_name} {self.dataset} figure plotted!')
    
    def train_model(self):
        train_losses=[]
        val_losses=[]
        
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.model.to(self.device)
        
        last_loss=1e5
        trigger_count=0

        print(f'Training {self.network} on {self.dataset} features extracted by {self.model_name}.')

        store_path = os.path.join(self.checkpoint_path, f"{self.network}_{self.dataset}_{self.loss_func}_{self.task}")
        if os.path.exists(store_path) and os.path.isdir(store_path):
            for filename in os.listdir(store_path):
                filepath = os.path.join(store_path, filename)

                if os.path.isfile(filepath):
                    os.remove(filepath)
        else:
            os.makedirs(store_path)
        
        for epoch in range(self.n_epochs):
            running_loss=0.0
            
            self.model.train()
            
            for i, batch in enumerate(tqdm(self.train_loader,desc=f'Training epoch {epoch+1}')):
                self.optimizer.zero_grad()
                text_feat = batch[:,:self.caption_embed_train.size(1)].to(self.device)
                img_feat = batch[:,self.caption_embed_train.size(1):].to(self.device)
                B = text_feat.size(0)
                # Ground-truth cosine matrix
                text_norm = F.normalize(text_feat, dim=-1)
                img_norm = F.normalize(img_feat, dim=-1)
                targets = text_norm @ img_norm.T      # [B, B]
                # Generate all image-text pairs
                text_expand = text_feat[:, None, :].expand(B, B, -1)
                img_expand = img_feat[None, :, :].expand(B, B, -1)
                # Evaluate all pairs
                outputs = self.model(
                    text_expand.reshape(B * B, -1),
                    img_expand.reshape(B * B, -1)
                )
                outputs = outputs.view(B, B)

                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            epoch_loss=running_loss/len(self.train_loader)
            train_losses.append(epoch_loss)

            val_running_loss=0.0
            self.model.eval()
            with torch.no_grad():
                for j, val_batch in enumerate(tqdm(self.val_loader, desc=f'Validating epoch {epoch+1}')):
                    text_feat = val_batch[:, :self.caption_embed_val.size(1)].to(self.device)
                    img_feat = val_batch[:, self.caption_embed_val.size(1):].to(self.device)

                    B = text_feat.size(0)

                    # Target cosine matrix: [B, B]
                    text_norm = F.normalize(text_feat, dim=-1)
                    img_norm = F.normalize(img_feat, dim=-1)
                    targets = text_norm @ img_norm.T              # [B, B]

                    # Build all text-image pairs: [B, B, D]
                    text_expand = text_feat[:, None, :].expand(B, B, -1)
                    img_expand = img_feat[None, :, :].expand(B, B, -1)

                    # Predict similarity for all pairs
                    outputs = self.model(
                        text_expand.reshape(B * B, -1),
                        img_expand.reshape(B * B, -1)
                    ).view(B, B)

                    # Full matrix MSE: diagonal + off-diagonal
                    val_loss = self.criterion(outputs, targets)
                    val_running_loss += val_loss.item()
                
                val_epoch_loss=val_running_loss/len(self.val_loader)
                val_losses.append(val_epoch_loss)

                print(f'Epoch: {epoch+1}, training loss: {epoch_loss:.5f}, validation loss: {val_epoch_loss:.5f}.')
                
            self.scheduler.step()

            if (epoch+1)%5==0:
                self.save_model(epoch, store_path)
                self.save_loss(train_losses,val_losses,epoch, store_path)
            
            # early epoch for improvement less than 1% for 5 consecutive epochs
            if self.early_stopping:
                if (last_loss-val_epoch_loss)/last_loss<self.tolerance:
                    trigger_count+=1
                    print(f'Trigger count: {trigger_count}')
                    if trigger_count>=self.patience:
                        print('*'*5+f'Early stopped at epoch {epoch+1}'+'*'*5)
                        self.save_model(epoch, store_path)
                        self.save_loss(train_losses,val_losses,epoch, store_path)
                        break
                else:
                    trigger_count=0

                last_loss=val_epoch_loss
            print('-'*50)
            
        self.plot(train_losses,val_losses, store_path)

    def evaluate_model(self, query_chunk_size=128, cand_chunk_size=512):
        ks = [1, 5, 10]

        self.model.eval()

        N = self.caption_embed_test.size(0)

        all_preds = []
        with torch.no_grad():
            for q_start in tqdm(range(0, N, query_chunk_size), desc="Testing"):
                q_end = min(q_start + query_chunk_size, N)

                text_feat = self.caption_embed_test[q_start:q_end].to(self.device)
                Q = text_feat.size(0)

                preds_chunks = []

                for c_start in range(0, N, cand_chunk_size):
                    c_end = min(c_start + cand_chunk_size, N)

                    img_feat = self.img_embed_test[c_start:c_end].to(self.device)
                    C = img_feat.size(0)

                    text_expand = text_feat[:, None, :].expand(Q, C, -1)
                    img_expand = img_feat[None, :, :].expand(Q, C, -1)

                    preds = self.model(
                        text_expand.reshape(Q * C, -1),
                        img_expand.reshape(Q * C, -1)
                    ).view(Q, C)

                    preds_chunks.append(preds)

                all_preds.append(torch.cat(preds_chunks, dim=1))

        all_preds = torch.cat(all_preds, dim=0)
        print(f"Diagonal mean: {all_preds.diag().mean()}")
        print(f"Off-diagonal mean: {(all_preds.diagonal(offset=1).mean() + all_preds.diagonal(offset=-1).mean()) / 2}")

        targets = torch.ones(N, device=self.device, dtype=torch.long).diag()
        index = torch.arange(N, device=self.device)[:, None].expand(N, N)
        print(f"*{'*'*10} Evaluation Results {'*'*10}")
        for k in ks:
            recall = RetrievalRecall(top_k=k).to(self.device)
            recall_val = recall(all_preds, targets, index) * 100
            print(f"Test Recall@{k}: {recall_val:.1f}%.")

def set_config():
    parser=argparse.ArgumentParser()
    parser.add_argument('model_name',type=str,help='Multimodal name')
    parser.add_argument('dataset',type=str,help='which dataset')
    parser.add_argument('--path',type=str,default='/data/data_fxu/alignment_features/',help='Root directory containing per-model feature folders')
    parser.add_argument('--network',type=str,default='mlp',help='which model structure')
    parser.add_argument('--lr',type=float,default=5e-5,help='learning rate')
    parser.add_argument('--n_epochs',type=int,default=100,help='number of epochs')
    parser.add_argument('--batch_size',type=int,default=64,help='batch size')
    parser.add_argument('--patience',type=int,default=10,help='patience for early stopping')
    parser.add_argument('--tolerance',type=float,default=1e-6,help='improvement for each epoch')
    parser.add_argument('--early_stopping',type=bool,default=False)
    parser.add_argument('--task',type=str,default='txt2img')
    parser.add_argument('--loss_func',type=str,default='mse')
    parser.add_argument('--gpu_id',type=int,default=0)
    args=parser.parse_args()

    return args


if __name__=='__main__':
    torch.manual_seed(0)

    args=set_config()

    train=Train(args.model_name,args.dataset,
            path=args.path,
            network=args.network,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            patience=args.patience,
            tolerance=args.tolerance,
            lr=args.lr,
            early_stopping=args.early_stopping,
            task=args.task,
            loss_func=args.loss_func,
            gpu_id=args.gpu_id
            )
    train.train_model()
    train.evaluate_model()
