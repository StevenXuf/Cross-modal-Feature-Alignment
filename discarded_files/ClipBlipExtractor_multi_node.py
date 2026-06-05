import torch
import os
import argparse
import torch.distributed as dist

from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor,AutoModel
from lavis.models import load_model_and_preprocess

from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP

from load_data import LoadCOCO
    
class ClipBlipFeatureExtractor():    
    def __init__(self,checkpoint,batch_size=256,n_dim_features=512):
        self.checkpoint=checkpoint

        dist.init_process_group('nccl')
        self.rank=dist.get_rank()
        self.n_total_device=dist.get_world_size()
        print(f'Running DDP on rank: {self.rank}. World size: {self.n_total_device}. GPUs: {torch.cuda.device_count()}.')

        self.device_id=self.rank%torch.cuda.device_count()

        self.model_name=self.checkpoint.split('/')[1].split('-')[0]
        self.batch_size=batch_size
        self.n_dim_features=n_dim_features

        self.load_coco=LoadCOCO()

        self.model=AutoModel.from_pretrained(self.checkpoint)
        self.model.to(self.device_id)
        self.model=DDP(self.model,device_ids=[self.device_id])
        self.processor=AutoProcessor.from_pretrained(self.checkpoint)

    @record
    def get_features(self,which_features='train',cap_idx=0,store_path='/project/home/p200248'):
        if which_features=='train':
            info=self.load_coco.get_imgID_captions_train()
            path='/project/home/p200248/train'
            flag='train'
        else:
            info=self.load_coco.get_imgID_captions_val()
            path='/project/home/p200248/validation'
            flag='val'
        
        img_ids=list(info.keys())

        n_samples=len(img_ids)

        img_embed=torch.zeros(n_samples,self.n_dim_features).to(self.device_id)
        caption_embed=torch.zeros(n_samples,self.n_dim_features).to(self.device_id)

        if self.batch_size==1:
            n_batch=n_samples//self.batch_size
        else:
            n_batch=n_samples//self.batch_size+1

        batch_partition=n_batch//self.n_total_device
        if self.rank==self.n_total_device-1:
            batch_range=range(self.rank*batch_partition,n_batch)
        else:
            batch_range=range(self.rank*batch_partition,(self.rank+1)*batch_partition)
        
        for i in tqdm(batch_range):
            imgs=[Image.open(path+f"/data/{'0'*(12-len(img_id))+img_id}.jpg") for img_id in img_ids[self.batch_size*i:self.batch_size*(i+1)]]
            caption_txts=[info[img_id][cap_idx] for img_id in img_ids[self.batch_size*i:self.batch_size*(i+1)]]

            caption_inputs=self.processor(text=caption_txts,return_tensors='pt',padding=True,truncation=True).to(self.device_id)
            img_inputs=self.processor(images=imgs,return_tensors='pt').to(self.device_id)

            with torch.no_grad():
                caption_features=self.model.module.get_text_features(**caption_inputs).detach()
                img_features=self.model.module.get_image_features(**img_inputs).detach()

            caption_embed[self.batch_size*i:self.batch_size*(i+1),:]=caption_features
            img_embed[self.batch_size*i:self.batch_size*(i+1),:]=img_features
        print('Feature extraction done!')
        
        #bring all results together via in-place operations.
        dist.barrier()
        dist.all_reduce(caption_embed,op=dist.ReduceOp.SUM)
        dist.all_reduce(img_embed,op=dist.ReduceOp.SUM)
        
        if os.path.exists(store_path+f'/{self.model_name.upper()}')==False:
            os.makedirs(store_path+f'/{self.model_name.upper()}')
        torch.save(caption_embed,store_path+f'/{self.model_name.upper()}/{self.model_name.lower()}_caption_{cap_idx}_embed_{flag}.pt')
        torch.save(img_embed,store_path+f'/{self.model_name.upper()}/{self.model_name.lower()}_img_embed_{flag}.pt')
        
        dist.destroy_process_group()

    def load_image_features(self,which_feature='train',store_path='/project/home/p200248'):
        torch.load(store_path+f'/{self.model_name.upper()}/{self.model_name.lower()}_img_embed_{which_feature}.pt')

    def load_captionfeatures(self,which_feature='train',store_path='/project/home/p200248',cap_idx=0):
        torch.load(store_path+f'/{self.model_name.upper()}/{self.model_name.lower()}_caption_{cap_idx}_embed_{which_feature}.pt')

def set_config():
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint',help='The model checkpoint')
    parser.add_argument('--batch_size',type=int,default=256,help='The batch size of images and texts')
    parser.add_argument('--n_dim_features',type=int,default=512,help='The number of feature dimensions')
    parser.add_argument('--which_features',type=str,default='train',help='Choose a image or caption features')
    parser.add_argument('--cap_idx',type=int,default=0,help='Choose one caption index')
    parser.add_argument('--store_path',type=str,default='/project/home/p200248',help='Store path of features')

    args=parser.parse_args()

    return args

if __name__=='__main__':
    args=set_config()

    feature_extractor=ClipBlipFeatureExtractor(checkpoint=args.checkpoint,
            batch_size=args.batch_size,
            n_dim_features=args.n_dim_features,
            )
    feature_extractor.get_features(which_features=args.which_features,
            cap_idx=args.cap_idx,
            store_path=args.store_path)
