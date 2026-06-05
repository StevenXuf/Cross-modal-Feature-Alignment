import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import umap

from sklearn.manifold import TSNE
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay

from torchmetrics.functional.pairwise import pairwise_cosine_similarity

from kl_divergence import compute_kl_divergence
class FeaturePlot():
    def __init__(self,method='tsne',store_path='/data/data_fxu/alignment_features',sample_size=2000):
        self.model_names=['clip','blip_official_itm','metatransformer','resnet/bert','convit/roberta','convnext/xlnet']
        self.store_path=store_path
        self.method=method
        self.datasets=['imdb','flickr30k','coco_train']
        self.sample_size=sample_size

        self.img_embeds=[]
        self.caption_embeds=[]

    def load_embeddings(self,model_name,dataset):
        model_name=''.join(model_name.split('/'))
        img_embed=torch.load(self.store_path+f'/{model_name.upper()}/{model_name.lower()}_img_embed_{dataset}.pt')
        txt_embed=torch.load(self.store_path+f'/{model_name.upper()}/{model_name.lower()}_caption_embed_{dataset}.pt')
        return img_embed,txt_embed

    def get_reducer(self):
        random_state=0
        if self.method.lower()=='tsne':
            reducer=TSNE(n_components=2,random_state=random_state)
        elif self.method.lower()=='umap':
            reducer=umap.UMAP(n_components=n_components)
        else:
            raise Exception('No such method.')
        return reducer

    def get_reduced_data(self,model_name,dataset):
        path=f'/data/data_fxu/reduced_features/{dataset.upper()}/'
        data_path=path+f"{''.join(model_name.split('/'))}.pt"
        if os.path.exists(data_path)==False:
            os.makedirs(path,exist_ok=True)
            reducer=self.get_reducer()

            img_embed,txt_embed=self.load_embeddings(model_name,dataset)
            all_mat=torch.cat((txt_embed.cpu(),img_embed.cpu()),0)
            features=torch.from_numpy(reducer.fit_transform(all_mat))
        
            torch.save(features,data_path)
            return features
        else:
            return torch.load(data_path)
    
    def get_rand_idx(self):
        rand_idx=torch.randperm(3000)[:self.sample_size]
        return rand_idx
    
    def get_all_reduced_data(self):
        for dataset in self.datasets:
            for model_name in self.model_names:
                print(dataset,model_name)
                self.get_reduced_data(model_name,dataset)
    
    def compute_cosine(self,reduced='yes'):
        diag_all=[]
        avg_all=[]
        for dataset in self.datasets:
            diag_per_dataset=[]
            avg_per_dataset=[]
            for model_name in self.model_names:
                if reduced=='yes':
                    print('Reduced features',dataset,model_name)
                    feat=self.get_reduced_data(model_name,dataset)
                    row=feat.size(0)
                    txt_feat=feat[:row//2,:]
                    img_feat=feat[row//2:,:]
                else:
                    print('Unreduced features',dataset,model_name)
                    img_feat,txt_feat=self.load_embeddings(model_name,dataset)

                cosine=pairwise_cosine_similarity(txt_feat,img_feat)
                diag_mean=torch.mean(torch.diag(cosine))
                all_mean=torch.mean(cosine.flatten())

                diag_per_dataset.append(diag_mean)
                avg_per_dataset.append(all_mean)
                print(f'Diagnal cosine mean: {round(diag_mean.item(),2)}')
                print(f'Average of all cosine: {round(all_mean.item(),2)}')
            diag_all.append(diag_per_dataset)
            avg_all.append(avg_per_dataset)
        
        return diag_all,avg_all
    
    def plot_cosine(self):
        reduced_diag_all,reduced_avg_all=self.compute_cosine('yes')
        unreduced_diag_all,unreduced_avg_all=self.compute_cosine('no')
        
        nrow=1
        ncol=2
        fig,axes=plt.subplots(nrow,ncol,figsize=(ncol*5,nrow*3),layout='constrained',sharex=True,sharey=True)
        datasets=['imdb','flickr30k','ms-coco']

        for i,data in enumerate(zip(self.compute_cosine('yes'),self.compute_cosine('no'))):
            ax=axes[i]
            ax.set_xticks(range(1,7),list(map(lambda string: string.upper(),['clip','blip','metatransformer','resnet/bert','convit/roberta','convnext/xlnet'])),fontsize=5)
            if i==0:
                ax.title.set_text(r'curves of $\frac{\sum_{i=1}^{n}cos(ft_{i},fi_{i})}{n}$')
            elif i==1:
                ax.title.set_text(r'curves of $\frac{\sum_{i=0}^{N}\sum_{j=0}^{N}cos(ft_{i},fi_{j})}{N^2}$')
            for j in range(3):
                ax.plot(range(1,7),data[0][j],label=f'2D {datasets[j].upper()}')
                ax.plot(range(1,7),data[1][j],label=f'full-dim {datasets[j]}')
            ax.legend(fontsize=8)
        plt.savefig('./cosine_curves.pdf')

    def plot_features(self):
        ncol=6
        nrow=3
        fig,axes=plt.subplots(nrow,ncol,sharex=True,sharey=True,figsize=(3*ncol,3*nrow),layout='constrained')
        
        rand_idx=self.get_rand_idx()

        for i in range(nrow):
            for j in range(ncol):
                clf=svm.SVC(kernel='rbf')
                
                features=self.get_reduced_data(self.model_names[j],self.datasets[i])
                half_size=features.size(0)//2
                txt=features[:half_size,:]
                txt_centroid=torch.mean(txt,dim=0)
                img=features[half_size:,:]
                img_centroid=torch.mean(img,dim=0)
                
                print(txt_centroid)
                print(img_centroid)
                kl_div=compute_kl_divergence(txt.numpy(),img.numpy())

                reduced_samples=torch.cat((txt[rand_idx,:],img[rand_idx,:]),dim=0)
                reduced_labels=torch.cat((torch.ones(self.sample_size,),torch.zeros(self.sample_size,)),dim=-1)
                clf.fit(reduced_samples,reduced_labels)
                
                decision_function=clf.decision_function(reduced_samples)
                margin=abs(decision_function)
                print(margin.shape)

                axes[i,j].scatter(txt[rand_idx,0],txt[rand_idx,1],c='#ef8a62',alpha=.3,marker='.',label='text features')
                axes[i,j].scatter(img[rand_idx,0],img[rand_idx,1],c='#67a9cf',alpha=.3,marker='+',label='image features')
                
                DecisionBoundaryDisplay.from_estimator(clf,reduced_samples,plot_method='contour',colors='k',levels=[-1,1],alpha=.5,linestyles=['--','--'],ax=axes[i,j])

                axes[i,j].scatter(txt_centroid[0],txt_centroid[1],c='yellow')
                axes[i,j].scatter(img_centroid[0],img_centroid[1],c='yellow')
                axes[i,j].plot(*zip(txt_centroid,img_centroid),linestyle='--',c='yellow')
                
                #center_cosine=pairwise_cosine_similarity(img_centroid.unsqueeze(0),txt_centroid.unsqueeze(0))
                center_dist=torch.sqrt(torch.sum(torch.pow(torch.subtract(txt_centroid,img_centroid), 2), dim=0))
                axes[i,j].text(.65,.9,f'euclidean: {round(center_dist.item(),2)}',horizontalalignment='center', verticalalignment='center', transform=axes[i,j].transAxes)
                #axes[i,j].text(.05,.9,f'margin: {np.round(np.mean(margin,-1),2)}',horizontalalignment='center', verticalalignment='center', transform=axes[i,j].transAxes)
                axes[i,j].text(.05,.9,f'kl: {kl_div}',horizontalalignment='center', verticalalignment='center', transform=axes[i,j].transAxes)
                if i==0 and self.model_names[j]=='blip_official_itm':
                    axes[i,j].title.set_text(f'BLIP')
                elif i==0:
                    axes[i,j].title.set_text(f'{self.model_names[j].upper()}')
                
                if j==0:
                    if self.datasets[i]=='coco_train':
                        axes[i,j].set_ylabel('MS-COCO',fontsize=12)
                    else:
                        axes[i,j].set_ylabel(f'{self.datasets[i].upper()}',fontsize=12)

        fig.legend(*fig.axes[0].get_legend_handles_labels(),loc='outside lower center',ncols=2)
        
        fig.savefig(f"./{self.method}_feature.pdf")
        
        print('Figure saved!')

def main():
    torch.manual_seed(0)
    feature_plot=FeaturePlot()
    feature_plot.plot_features()
    #feature_plot.plot_cosine()

if __name__=='__main__':
    main()
