import math
import torch 
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from compute_distance import ComputeDistance
from captum.attr import IntegratedGradients
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma

def load_embeddings(model,dataset,device):
    path='/data/data_fxu/alignment_features/'+model.upper()
    title_embed=torch.load(path+f'/{model.lower()}_caption_embed_{dataset}.pt',weights_only=True).to(device)
    img_embed=torch.load(path+f'/{model.lower()}_img_embed_{dataset}.pt',weights_only=True).to(device)
    return title_embed,img_embed

def plot_metric_dist(model,dataset):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    txt_embed,img_embed=load_embeddings(model,dataset,device)
    cd=ComputeDistance(model,dataset,'img2txt',gpu_id=0,only_cosine='no')

    txt_norms=txt_embed.norm(p=2,dim=-1)
    img_norms=img_embed.norm(p=2,dim=-1)
    norm_diff=txt_norms-img_norms
    
    euclidean=torch.norm(txt_embed-img_embed,p=2,dim=-1).cpu()
    manhattan=torch.norm(txt_embed-img_embed,p=1,dim=-1).cpu()
    cosine=F.cosine_similarity(txt_embed,img_embed,dim=-1).cpu()
    
    pairwise_euclidean=cd._pairwise_euclidean_distance(txt_embed,img_embed)
    pairwise_manhattan=cd._pairwise_manhattan_distance(txt_embed,img_embed)
    pairwise_chi_square=cd.pairwise_chi_square(txt_embed,img_embed)
    pairwise_cosine=cd._pairwise_cosine_distance(txt_embed,img_embed)

    fig,axes=plt.subplots(nrows=1,ncols=4,figsize=(16,3))
    axes[0].hist(norm_diff.cpu(),bins=100,alpha=.5,label='norm diff')
    axes[0].hist(txt_norms.cpu(),bins=100,alpha=.5,label='text norm')
    axes[0].hist(img_norms.cpu(),bins=100,alpha=.5,label='image norm')

    axes[1].hist(euclidean,bins=100,alpha=.5,label='positive')
    axes[1].hist(pairwise_euclidean.flatten().tolist(),bins=100,alpha=.5,label='all')

    axes[2].hist(manhattan,bins=100,alpha=.5,label='positive')
    axes[2].hist(pairwise_manhattan.flatten().tolist(),bins=100,alpha=.5,label='all')

    axes[3].hist(cosine,bins=100,alpha=.5,label='positive')
    axes[3].hist(pairwise_cosine.flatten().tolist(),bins=100,alpha=.5,label='all')
    labels=['','euclidean','manhattan','cosine']
    
    for i in range(4):
        axes[i].grid(True, linestyle='--', alpha=0.5)
        axes[i].set_title(labels[i])

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./hist_{model}_{dataset}.pdf')

def sim_score(metric, txt_embed, img_embed):
    if metric == 'cosine':
        return torch.cosine_similarity(txt_embed, img_embed, dim=-1)
    elif metric == 'manhattan':
        return -torch.norm(txt_embed - img_embed, p=1, dim=-1)
    elif metric == 'euclidean':
        return -torch.norm(txt_embed - img_embed, p=2, dim=-1)
    elif metric=='chi-squre':
        return -.5*torch.sum(torch.pow(txt_embed-img_embed,2)/(txt_embed+img_embed),dim=-1)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def measure_importance(model, dataset, device,task='img2txt'):
    print(f'\nModel: {model}, Dataset: {dataset}')
    txt_embed, img_embed = load_embeddings(model, dataset, device)
    
    if task=='txt2img':
        txt_embed,img_embed=img_embed,txt_embed

    metrics = ['cosine','euclidean', 'manhattan','chi-squre']
    
    nrows,ncols=img_embed.size()
    importance=torch.zeros(1,ncols)
    info={}
    shared_dim=set(range(ncols))
    for metric in metrics:
        print(f"\nMetric: {metric}")
        for idx in range(1000):
            txt_sample = txt_embed[idx:idx+1, :].to(device).requires_grad_(True)
            img_sample = img_embed[idx:idx+1, :].to(device)
            baseline = torch.zeros_like(img_sample)

            def wrapped(img_input, text_input=txt_sample, m=metric):
                return sim_score(m, text_input, img_input)

            ig = IntegratedGradients(wrapped)
            attrs, delta = ig.attribute(
                inputs=img_sample,
                baselines=baseline,
                return_convergence_delta=True
            )

            importance+=attrs.detach().cpu().view(1,-1).abs()
        info[metric]=importance/1000
        
        vals,ids=torch.topk(info[metric],k=10)
        print(f'Top-10: {ids.flatten().tolist()}')
        
        if metric=='cosine':
            cosine_dim=ids.flatten().tolist()
        shared_dim=set(cosine_dim)&set(ids.flatten().tolist())
        print(f'Number of shared dims: {len(shared_dim)}')

        shared_importance=torch.sum(info[metric][:,list(shared_dim)],dim=-1)
        total_importance=torch.sum(vals,dim=-1)
        print(f'Percentage of importance of shared dims: {shared_importance.item()/total_importance.item()*100:.2f}%')
    '''
    plt.figure(figsize=(4,3))
    for j in range(len(metrics)):
        plt.bar(range(ncols),info[metrics[j]],alpha=.5,label=metrics[j])
    plt.legend()          
    plt.savefig('./importance_{model}_{dataset}.pdf')
    '''

def knn_entropy(embeddings: np.ndarray, k: int = 5) -> float:
    """
    Non‐parametric Kozachenko–Leonenko kNN entropy estimator.
    
    embeddings: array of shape (n_samples, d)
    k:          which neighbor to use (default: 5)
    returns:    estimated differential entropy (in nats)
    """
    n, d = embeddings.shape

    # Find distances to k-th neighbor
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    eps = distances[:, -1]  # k-th neighbor distance for each point

    # Compute average log-distance
    avg_log_eps = np.mean(np.log(eps + 1e-12))

    # Numerically‐stable log volume of unit ball in d dims
    log_cd = (d / 2) * math.log(math.pi) - math.lgamma(d / 2 + 1)

    # Kozachenko–Leonenko estimator:
    return digamma(n) - digamma(k) + log_cd + d * avg_log_eps

def measure_info_richness(model,dataset):
    print(f'Model: {model}, Dataset: {dataset}')
    txt_embed, img_embed = load_embeddings(model, dataset,'cpu')
    txt_embed=(txt_embed-txt_embed.mean(0))/txt_embed.std(0)
    img_embed=(img_embed-img_embed.mean(0))/img_embed.std(0)

    H_img_knn = knn_entropy(img_embed, k=5)
    H_txt_knn = knn_entropy(txt_embed,  k=5)
    print(f"kNN Entropy (images): {H_img_knn:.2f} nats")
    print(f"kNN Entropy (text  ): {H_txt_knn:.2f} nats")
    
    text_norms  = np.linalg.norm(txt_embed,  axis=1)
    image_norms = np.linalg.norm(img_embed, axis=1)
    
    print("Image norms: mean %.3f, std %.3f" %
          (image_norms.mean(), image_norms.std()))
    print("Text norms:  mean %.3f, std %.3f" %
          (text_norms.mean(), text_norms.std()))

if __name__=='__main__':
    models=['clip','blip'] #'metatransformer','resnetbert','convitroberta','convnextxlnet']
    datasets=['imdb','flickr30k','coco_train']
    for dataset in datasets:
        for model in models:
            #plot_metric_dist(model,dataset)
            measure_importance(model,dataset,0,'txt2img')
            #measure_info_richness(model,dataset)
