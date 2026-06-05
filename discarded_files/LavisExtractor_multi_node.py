import torch
import os
import argparse
import torch.distributed as dist
import torch.nn as nn

from tqdm import tqdm
from PIL import Image
from lavis.models import load_model_and_preprocess

from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Subset

from load_data import LoadCOCO,LoadFlickr30K,LoadIMDB,LoadCelebA

class FeatureExtractor():
    def __init__(self,checkpoint="blip2_feature_extractor",n_dim_features=768):
        self.checkpoint=checkpoint
        self.model_name=self.checkpoint.split('_')[0]
        self.n_dim_features=n_dim_features

        self.load_coco=LoadCOCO()
        self.load_flickr=LoadFlickr30K()
        self.load_celebA=LoadCelebA()
        self.load_imdb=LoadIMDB()

        dist.init_process_group('nccl')
        self.rank=dist.get_rank()
        self.n_total_device=dist.get_world_size()
        print(f'Running DDP on rank: {self.rank}. World size: {self.n_total_device}. GPUs: {torch.cuda.device_count()}.')

        self.device_id=self.rank%torch.cuda.device_count()
        
        if self.model_name=='blip':
            self.model_type='base'
        elif self.model_name=='blip2':
            self.model_type='pretrain'
        elif self.model_name=='clip':
            self.model_type='ViT-B-32'
            self.n_dim_features=512
        
        self.model,self.img_processor,self.txt_processor=load_model_and_preprocess(
                name=self.checkpoint,
                model_type=self.model_type,
                is_eval=True
                )
        self.model.to(self.device_id)
        self.model=DDP(self.model,device_ids=[self.device_id])
    
    @record
    @torch.no_grad
    def get_features(self,which_dataset='coco_train',batch_size=1,feature_choice='1st',store_path='/project/home/p200248/alignment'):
        if which_dataset=='coco_train':
            dataset,_=self.load_coco.get_dataset()
            path=store_path+'/MS-COCO/train'
        elif which_dataset=='coco_val':
            _,dataset=self.load_coco.get_dataset()
            path=store_path+'/MS-COCO/validation'
        elif which_dataset=='flickr30k':
            dataset=self.load_flickr.get_dataset()
            path=store_path+'/Flickr30k/flickr30k_images'
        elif which_dataset=='imdb':
            dataset=self.load_imdb.get_dataset()
            path=store_path+'/IMDB/images'
        elif which_dataset=='celebA':
            dataset=self.load_celebA.get_dataset()
            path=store_path+'/CelebA/img_align_celeba'

        n_samples=len(dataset)
        
        caption_embed=torch.zeros(n_samples,self.n_dim_features).to(self.device_id)
        img_embed=torch.zeros(n_samples,self.n_dim_features).to(self.device_id)

        if batch_size==1:
            n_batch=n_samples//batch_size
        else:
            n_batch=n_samples//batch_size+1

        batch_partition=n_batch//self.n_total_device
        
        if self.rank==self.n_total_device-1:
            batch_range=range(self.rank*batch_partition,n_batch)
        else:
            batch_range=range(self.rank*batch_partition,(self.rank+1)*batch_partition)
        
        subset=Subset(dataset,list(batch_range)*batch_size)
        dataloader=DataLoader(subset,batch_size=batch_size)

        for i,batch in enumerate(tqdm(dataloader)):
            imgs=[Image.open(path+f"/data/{'0'*(12-len(img_id))+img_id}.jpg").convert('RGB')  if 'coco' in which_dataset else Image.open(path+f"/{img_id}").convert('RGB') for img_id in batch['img_id']]
            caption_txts=batch['text']
            
            caption_inputs=self.txt_processor['eval'](caption_txts[0])
            img_inputs=self.img_processor['eval'](imgs[0]).unsqueeze(0).to(self.device_id)
            
            samples={'image':img_inputs,'text_input':[caption_inputs]}
            
            if self.model_name=='clip':
                caption_features=self.model.module.extract_features(samples).text_embeds.detach()
                img_features=self.model.module.extract_features(samples).image_embeds.detach()
                caption_embed[(batch_range[0]+i)*batch_size:(batch_range[0]+i+1)*batch_size,:]=caption_features
                img_embed[(batch_range[0]+i)*batch_size:(batch_range[0]+i+1)*batch_size,:]=img_features
            else:
                caption_features=self.model.module.extract_features(samples,mode='text').text_embeds.detach()
                img_features=self.model.module.extract_features(samples,mode='image').image_embeds.detach()
            
                if feature_choice=='1st':
                    caption_embed[(batch_range[0]+i)*batch_size:(batch_range[0]+i+1)*batch_size,:]=caption_features[:,0,:]
                    img_embed[(batch_range[0]+i)*batch_size:(batch_range[0]+i+1)*batch_size,:]=img_features[:,0,:]
                elif feature_choice=='avg':
                    caption_embed[(batch_range[0]+i)*batch_size:(batch_range[0]+i+1)*batch_size,:]=torch.mean(caption_features[0],dim=0)
                    img_embed[(batch_range[0]+i)*batch_size:(batch_range[0]+i+1)*batch_size,:]=torch.mean(img_features[0],dim=0)
                elif feature_choice=='max':
                    caption_embed[(batch_range[0]+i)*batch_size:(batch_range[0]+i+1)*batch_size,:]=torch.max(caption_features[0],dim=0).values
                    img_embed[(batch_range[0]+i)*batch_size:(batch_range[0]+i+1)*batch_size,:]=torch.max(img_features[0],dim=0).values

        print('Feature extraction done!')
        
        dist.all_reduce(caption_embed,op=dist.ReduceOp.SUM)
        dist.all_reduce(img_embed,op=dist.ReduceOp.SUM)

        if os.path.exists(store_path+f'/{self.model_name.upper()}')==False:
            os.makedirs(store_path+f'/{self.model_name.upper()}')
        torch.save(caption_embed,store_path+f'/{self.model_name.upper()}/{self.model_name.lower()}_caption_embed_{which_dataset}.pt')
        torch.save(img_embed,store_path+f'/{self.model_name.upper()}/{self.model_name.lower()}_img_embed_{which_dataset}.pt')

        dist.destroy_process_group()

    def load_image_features(self,which_feature='train',store_path='/project/home/p200248/alignment'):
        torch.load(store_path+f'/{self.model_name.upper()}/{self.model_name.lower()}_img_embed_{which_feature}.pt')

    def load_captionfeatures(self,which_feature='train',store_path='/project/home/p200248/alignment'):
        torch.load(store_path+f'/{self.model_name.upper()}/{self.model_name.lower()}_caption_embed_{which_feature}.pt')


def set_config():
    parser=argparse.ArgumentParser()
    parser.add_argument('--feature_choice',type=str,default='1st',const='1st',nargs='?',choices=['1st','avg','max'],help='Use the 1st vector as feature if True')
    parser.add_argument('--batch_size',type=int,default=1,help='The batch size of images and texts')
    parser.add_argument('--n_dim_features',type=int,default=768,help='The number of feature dimensions')
    parser.add_argument('--checkpoint',type=str,default="blip2_feature_extractor",help='Specify which feature extractor')
    parser.add_argument('--which_dataset',type=str,default='coco_train',help='Choose a dataset')
    parser.add_argument('--store_path',type=str,default='/project/home/p200248/alignment',help='Store path of features')

    args=parser.parse_args()

    return args

if __name__=='__main__':
    args=set_config()

    feature_extractor=FeatureExtractor(
            checkpoint=args.checkpoint,
            n_dim_features=args.n_dim_features
            )
    feature_extractor.get_features(
            which_dataset=args.which_dataset,
            feature_choice=args.feature_choice,
            batch_size=args.batch_size,
            store_path=args.store_path
            )
