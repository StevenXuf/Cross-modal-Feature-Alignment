import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

import matplotlib.pyplot as plt

class ComputeDistance():
    def __init__(self,gpu_id=0):
        self.gpu_id=gpu_id
        self.model_name=['clip','blip_official_itm','metatransformer','resnet/bert','convit/roberta','convnext/xlnet']
        self.datasets=['imdb','flickr30k','coco_train']
        
        self.device=torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    def load_embeddings(self,model_name,dataset):
        title_embed=torch.load('/data/data_fxu/alignment_features/'+model_name.upper()+f'/{model_name.lower()}_caption_embed_{dataset}.pt').to(self.device)
        img_embed=torch.load('/data/data_fxu/alignment_features/'+model_name.upper()+f'/{model_name.lower()}_img_embed_{dataset}.pt').to(self.device)
        return title_embed,img_embed 


    @torch.no_grad
    def _pairwise_cosine_distance(self,mat_a,mat_b):
        nrow,ncol=mat_a.size()
        res=torch.zeros(nrow,nrow)
        for i in tqdm(range(nrow)):
            sim=torch.sum(mat_a[i,:]*mat_b,dim=1)/torch.sqrt(torch.sum(mat_a[i,:]**2)*torch.sum(mat_b**2,dim=1))
            res[i,:]=sim.detach().cpu()
        return res
    
    def visualize(self):
        torch.manual_seed(0)
        idx=torch.randperm(3000).tolist()[:20]
        fig,axes=plt.subplots(3,6,figsize=(18,9),layout='constrained')
        margin_config={
                    'left':0.02,
                    'right':.96,
                    'bottom':0,
                    'top':1
                    }
        #plt.subplots_adjust(wspace=0.18, hspace=0,**margin_config)
        for j in range(len(self.datasets)):
            for i in range(6):
                model_name=''.join(self.model_name[i].split('/'))
                dataset=self.datasets[j]
                txt_embed,img_embed=self.load_embeddings(model_name,dataset)

                sim=self._pairwise_cosine_distance(txt_embed[idx,:],img_embed[idx,:])
                
                ax=axes[j,i] 
                im=ax.imshow(sim,cmap='inferno',interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                cbar=plt.colorbar(im,ax=ax,fraction=0.05,pad=0.02)
                cbar.ax.tick_params(rotation=55)
                if i==0 and j==2:
                    ax.set_ylabel('MS-COCO',fontsize=12)
                elif i==0:
                    ax.set_ylabel(f'{dataset.upper()}',fontsize=12)
                
                if j==0:
                    if model_name=='blip_official_itm':
                        ax.title.set_text('BLIP')
                    else:
                        ax.title.set_text(self.model_name[i].upper())
        plt.savefig(f'./heatmap_all_datasets.pdf')
        print('Heatmap plotted.')

if __name__=='__main__':
    compute_distance=ComputeDistance()
    compute_distance.visualize()
