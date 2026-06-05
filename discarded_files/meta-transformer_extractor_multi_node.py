import torch
import os
import sys
import argparse
import torch.distributed as dist
import torch.nn as nn
import torchvision.io as io

from tqdm import tqdm

import torchvision.transforms as transforms

from torch.utils.data import DataLoader,Subset
from torch.utils.data.distributed import DistributedSampler

from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP

from load_data import LoadCOCO,LoadFlickr30K,LoadIMDB,LoadCelebA
from timm.models.vision_transformer import Block

from sampler import DistributedEvalSampler

sys.path.append('/home/users/u101139/alignment/MetaTransformer_changed/Data2Seq')
from Data2Seq import Data2Seq

class MetaTransformerExtractor():    
    def __init__(self,batch_size=512,n_dim_features=768):
        dist.init_process_group('nccl')
        self.rank=dist.get_rank()
        self.n_total_device=dist.get_world_size()
        
        print(f'Running DDP on rank: {self.rank}. World size: {self.n_total_device}. GPUs: {torch.cuda.device_count()}.')

        self.device_id=self.rank%torch.cuda.device_count()
        torch.cuda.set_device(self.device_id)

        self.model_name='MetaTransformer'
        self.batch_size=batch_size
        self.n_dim_features=n_dim_features

        self.load_coco=LoadCOCO()
        self.load_flickr30k=LoadFlickr30K()
        self.load_imdb=LoadIMDB()
        self.load_celebA=LoadCelebA()

        self.model=nn.Sequential(*[
            Block(
                dim=self.n_dim_features,
                num_heads=12,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(12)])
        ckpt=torch.load("/project/home/p200248/alignment/Meta-Transformer-checkpoint/Meta-Transformer_base_patch16_encoder.pth")
        self.model.load_state_dict(ckpt,strict=True)

        self.text_processor=Data2Seq(modality='text',dim=self.n_dim_features,device_id=self.device_id)
        self.img_processor=Data2Seq(modality='image',dim=self.n_dim_features,device_id=self.device_id)
        
        self.model.to(self.device_id)
        self.model=DDP(self.model,device_ids=[self.device_id])

    @torch.no_grad
    def get_features(self,which_dataset='coco_train',store_path='/project/home/p200248/alignment'):
        if which_dataset=='coco_train':
            dataset,_=self.load_coco.get_dataset()
            path=store_path+'/MS-COCO/train'
        elif which_dataset=='coco_val':
            _,dataset=self.load_coco.get_dataset()
            path=store_path+'/MS-COCO/validation'
        elif which_dataset=='flickr30k':
            dataset=self.load_flickr30k.get_dataset()
            path=store_path+'/Flickr30k/flickr30k_images'
        elif which_dataset=='imdb':
            dataset=self.load_imdb.get_dataset()
            path=store_path+'/IMDB/images'
        elif which_dataset=='celebA':
            dataset=self.load_celebA.get_dataset()
            path=store_path+'/CelebA/img_align_celeba'

        n_samples=len(dataset)
        print(f'Total number of data points: {n_samples}')
        
        img_embed=[]
        caption_embed=[]
        
        sampler=DistributedEvalSampler(dataset)
        dataloader=DataLoader(dataset,batch_size=self.batch_size,shuffle=False,sampler=sampler)
        
        transform=transforms.Compose([
            lambda img: torch.cat((img,img,img),dim=0) if img.size(0)==1 else img,
            transforms.Resize((224,224),antialias=True),
            lambda img: img/255,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        for i,batch in enumerate(tqdm(dataloader)):
            img_inputs=[]            
            for img_id in batch['img_id']:
                if 'coco' in which_dataset:
                    img=transform(io.read_image(path+f"/data/{'0'*(12-len(img_id))+img_id}.jpg")).unsqueeze(0)
                else:
                    img=transform(io.read_image(path+f'/{img_id}')).unsqueeze(0)
                img_inputs.append(img)
            
            img_inputs=self.img_processor(torch.cat(img_inputs,dim=0).to(self.device_id))

            caption_txts=batch['text']
            current_batch_size=len(caption_txts)
            
            print(f"batch size on machine {self.rank}: {current_batch_size}")

            caption_inputs=self.text_processor(caption_txts).to(self.device_id)

            caption_features=self.model(caption_inputs).detach()
            img_features=self.model(img_inputs).detach()
            caption_features=caption_features.squeeze(1)
            img_features=torch.mean(img_features,dim=1)
            print(f'feature size on machine {self.rank}: {caption_features.size()}')
            batch_size_list=[None for _ in range(self.n_total_device)]
            dist.all_gather_object(batch_size_list,current_batch_size)
            max_size=max(batch_size_list)
            print(f'batch size list: {batch_size_list}')

            if current_batch_size<max_size:
                caption_features=torch.cat((caption_features,torch.zeros(max_size-current_batch_size,self.n_dim_features).to(self.device_id)),dim=0)
                img_features=torch.cat((img_features,torch.zeros(max_size-current_batch_size,self.n_dim_features).to(self.device_id)),dim=0)

            gather_list_cap=[torch.zeros_like(caption_features).to(self.device_id) for _ in range(self.n_total_device)]
            gather_list_img=[torch.zeros_like(img_features).to(self.device_id) for _ in range(self.n_total_device)]
            print(f'gather list length: {list(map(len,gather_list_cap))}')
            
            dist.barrier()
            dist.all_gather(gather_list_cap,caption_features)
            dist.all_gather(gather_list_img,img_features)

            if self.rank==0:
                caption_embed.append(torch.cat(gather_list_cap,dim=0))
                img_embed.append(torch.cat(gather_list_img,dim=0))

            dist.barrier()

        print('Feature extraction done!')
        
        #bring all results together via in-place operations.
        #dist.all_reduce(caption_embed,op=dist.ReduceOp.SUM)
        #dist.all_reduce(img_embed,op=dist.ReduceOp.SUM)
        
        if os.path.exists(store_path+f'/{self.model_name.upper()}')==False:
            os.makedirs(store_path+f'/{self.model_name.upper()}')
        
        if self.rank==0:
            caption_embed=self.remove_zeros(torch.cat(caption_embed,dim=0))
            img_embed=self.remove_zeros(torch.cat(img_embed,dim=0))
            print(f'embedding size: {caption_embed.size()}')

            torch.save(caption_embed,store_path+f'/{self.model_name.upper()}/{self.model_name.lower()}_caption_embed_{which_dataset}.pt')
            torch.save(img_embed,store_path+f'/{self.model_name.upper()}/{self.model_name.lower()}_img_embed_{which_dataset}.pt')
        
        dist.destroy_process_group()

    def remove_zeros(self,tensor):
        zeros=torch.zeros(self.n_dim_features).to(self.device_id)
        idx=[]
        for i in range(tensor.size(0)):
            if not torch.equal(tensor[i,:],zeros):
                idx.append(i)
            else:
                print(f'padding row: {i}')
        return tensor[idx,:]

    def load_image_features(self,which_dataset='coco_train',store_path='/project/home/p200248/alignment'):
        torch.load(store_path+f'/{self.model_name.upper()}/{self.model_name.lower()}_img_embed_{which_dataset}.pt')

    def load_captionfeatures(self,which_dataset='coco_train',store_path='/project/home/p200248/alignment'):
        torch.load(store_path+f'/{self.model_name.upper()}/{self.model_name.lower()}_caption_embed_{which_dataset}.pt')

def set_config():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=512,help='The batch size of images and texts')
    parser.add_argument('--n_dim_features',type=int,default=768,help='The number of feature dimensions')
    parser.add_argument('--which_dataset',type=str,default='coco_train',help='Choose a image or caption features')
    parser.add_argument('--store_path',type=str,default='/project/home/p200248/alignment',help='Store path of features')

    args=parser.parse_args()

    return args

if __name__=='__main__':
    args=set_config()

    feature_extractor=MetaTransformerExtractor(
            batch_size=args.batch_size,
            n_dim_features=args.n_dim_features,
            )
    feature_extractor.get_features(which_dataset=args.which_dataset,
            store_path=args.store_path)
