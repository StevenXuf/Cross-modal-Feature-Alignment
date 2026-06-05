import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

import matplotlib.pyplot as plt

class ComputeDistance():
    def __init__(self,model_name,which_dataset,img_retrieval,gpu_id=0,only_cosine='no'):
        self.gpu_id=gpu_id
        self.model_name=model_name
        self.which_dataset=which_dataset
        self.img_retrieval=img_retrieval

        self.path='/data/data_fxu/alignment_features/'+self.model_name.upper()

        self.device=torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        if only_cosine=='yes':
            self.metrics=['cosine']
        else:
            self.metrics=['euclidean','cosine','manhattan','chi-square']

    def load_embeddings(self):
        title_embed=torch.load(self.path+f'/{self.model_name.lower()}_caption_embed_{self.which_dataset}.pt').to(self.device)
        img_embed=torch.load(self.path+f'/{self.model_name.lower()}_img_embed_{self.which_dataset}.pt').to(self.device)
        return title_embed,img_embed 


    def get_distance(self,txt_embed,img_embed):
        res={}
        for metric in self.metrics:
            print(f'Computing {metric} distance')
            with torch.no_grad():
                if metric=='euclidean':
                    sim=self._pairwise_euclidean_distance(txt_embed,img_embed)
                elif metric=='cosine':
                    sim=self._pairwise_cosine_distance(txt_embed,img_embed)
                elif metric=='manhattan':
                    sim=self._pairwise_manhattan_distance(txt_embed,img_embed)
                elif metric=='canberra':
                    sim=self.pairwise_canberra_distance(txt_embed,img_embed)
                elif metric=='bray-curtis':
                    sim=self.pairwise_bray_curtis(txt_embed,img_embed)
                elif metric=='chi-square':
                    sim=self.pairwise_chi_square(txt_embed,img_embed)
                elif metric=='customized':
                    sim=self.customized_probs(txt_embed,img_embed)
                print(f'{metric.capitalize()} distance done!')
            #self.visualize(sim,metric)
            res[f'{metric}']=sim
        return res
    
    def top_k_accuracy(self):
        cnt=0
        title_embed,img_embed=self.load_embeddings()
        ids=torch.randperm(title_embed.size(0))[:1000]
        if self.img_retrieval=='yes':
            print('txt2img retrieval')
            all_distance=self.get_distance(title_embed[ids,:],img_embed[ids,:])
        else:
            print('img2txt retrieval')
            all_distance=self.get_distance(img_embed[ids,:],title_embed[ids,:])	
        keys=all_distance.keys()
        res={}
        for key in keys:
            distance=all_distance[key][:1000,:1000]
            for k in [1,5,10]:
                if key in ['euclidean','manhattan','canberra','chi-square','bray-curtis']:
                    vals,ids=torch.topk(-1*distance,k=k,dim=1)
                elif key in ['cosine','customized']:
                    vals,ids=torch.topk(distance,k=k,dim=1)
                else:
                    raise Exception('No such metric')
                
                for i in range(ids.size(0)):
                    if i in ids[i]:
                        cnt+=1

                acc=round(cnt/ids.size(0)*100,2)

                res[key+f'_top{k}_acc']=acc
                print(f'Top {k} accuracy: {acc} using {key}')
                cnt=0
            #torch.save(res,self.path+f'/{self.model_name.lower()}_{self.which_dataset}_{key}_acc.json')

    def customized_probs(self,mat_a,mat_b):
        nrow=mat_a.size(0)
        ncol=mat_b.size(0)
        res=torch.zeros(nrow,ncol)
        for i in tqdm(range(nrow)):
            len_ref=torch.linalg.norm(mat_a[i,:],dim=-1,ord=2)
            len_in=torch.linalg.norm(mat_b,dim=-1,ord=2)
            cosine=nn.CosineSimilarity(dim=-1)
            cos_val=cosine(mat_a[i,:]+torch.zeros(mat_b.size()).to(self.device),mat_b)
            sin_val=torch.sqrt(1-cos_val**2)
            loss=len_in*sin_val+torch.abs(len_in*cos_val-len_ref)
            probs=1/torch.exp(loss)
            res[i,:]=probs
        return res

    @torch.no_grad
    def _pairwise_euclidean_distance(self,mat_a,mat_b):
        nrow,ncol=mat_a.size()
        res=torch.zeros(nrow,nrow)
        for i in tqdm(range(nrow)):
            sim=torch.sqrt(torch.sum((mat_a[i,:]-mat_b)**2,dim=1))
            res[i,:]=sim.detach().cpu()
        return res
    
    @torch.no_grad
    def _pairwise_cosine_distance(self,mat_a,mat_b):
        nrow=mat_a.size(0)
        ncol=mat_b.size(0)
        res=torch.zeros(nrow,ncol)
        for i in tqdm(range(nrow)):
            sim=torch.sum(mat_a[i,:]*mat_b,dim=1)/torch.sqrt(torch.sum(mat_a[i,:]**2)*torch.sum(mat_b**2,dim=1))
            res[i,:]=sim.detach().cpu()
        return res
    
    @torch.no_grad
    def pairwise_bray_curtis(self,mat_a,mat_b):
        nrow,ncol=mat_a.size()
        res=torch.zeros(nrow,nrow)
        for i in tqdm(range(nrow)):
            sim=torch.sum(abs(mat_a[i,:]-mat_b),dim=1)/torch.sum((mat_a[i,:]+mat_b),dim=1)
            res[i,:]=sim.detach().cpu()
        return res

    @torch.no_grad
    def pairwise_chi_square(self,mat_a,mat_b):
        nrow,ncol=mat_a.size()
        res=torch.zeros(nrow,nrow)
        for i in tqdm(range(nrow)):
            sim=torch.sum((mat_a[i,:]-mat_b)**2/(mat_a[i,:]+mat_b),dim=1)/2
            res[i,:]=sim.detach().cpu()
        return res
    
    @torch.no_grad
    def pairwise_canberra_distance(self,mat_a,mat_b):
        nrow,ncol=mat_a.size()
        res=torch.zeros(nrow,nrow)
        for i in tqdm(range(nrow)):
            sim=torch.sum(abs(mat_a[i,:]-mat_b)/(abs(mat_a[i,:])+abs(mat_b)),dim=1)
            res[i,:]=sim.detach().cpu()
        return res
    
    @torch.no_grad
    def _pairwise_manhattan_distance(self,mat_a,mat_b):
        nrow,ncol=mat_a.size()
        res=torch.zeros(nrow,nrow)
        for i in tqdm(range(nrow)):
            sim=torch.sum(abs(mat_a[i,:]-mat_b),dim=1)
            res[i,:]=sim.detach().cpu()
        return res

    def visualize(self,sim_mat,metric):
        plt.figure(figsize=(4,3))
        plt.imshow(sim_mat[:20,:20],cmap='viridis',interpolation='nearest')
        plt.axis('off')
        plt.colorbar()
        plt.tight_layout()

        plt.savefig(self.path+f'/{self.model_name}_{self.which_dataset}_{metric}_plot.pdf')


def set_config():
        parser=argparse.ArgumentParser()
        parser.add_argument('model_name',help='the folder containing text and image features')
        parser.add_argument('which_dataset',help='which dataset')
        parser.add_argument('img_retrieval',help='retrieve img or text')
        parser.add_argument('--GPU_id',type=int,default=0,help='Specify a GPU')
        args=parser.parse_args()

        return args

if __name__=='__main__':
    torch.manual_seed(0)
    args=set_config()
    compute_distance=ComputeDistance(args.model_name,args.which_dataset,args.img_retrieval,gpu_id=args.GPU_id)
    compute_distance.top_k_accuracy()

