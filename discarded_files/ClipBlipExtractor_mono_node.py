import torch
import os
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor,AutoModel
from lavis.models import load_model_and_preprocess

import argparse
from load_data import LoadCOCO
    
class ClipBlipFeatureExtractor():    
    def __init__(self,checkpoint,batch_size=256,n_dim_features=512,n_gpus=4):
        self.checkpoint=checkpoint
        self.model_name=self.checkpoint.split('/')[1].split('-')[0]
        self.batch_size=batch_size
        self.n_dim_features=n_dim_features
        self.n_gpus=n_gpus

        self.load_coco=LoadCOCO()

        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model=AutoModel.from_pretrained(self.checkpoint)
        self.model=nn.DataParallel(self.model,device_ids=list(range(self.n_gpus)))
        self.model.to(self.device)
        self.processor=AutoProcessor.from_pretrained(self.checkpoint)

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

        img_embed=torch.zeros(n_samples,self.n_dim_features)
        caption_embed=torch.zeros(n_samples,self.n_dim_features)

        if self.batch_size==1:
            n_batch=n_samples//self.batch_size
        else:
            n_batch=n_samples//self.batch_size+1

        for i in tqdm(range(n_batch)):
            imgs=[Image.open(path+f"/data/{'0'*(12-len(img_id))+img_id}.jpg") for img_id in img_ids[self.batch_size*i:self.batch_size*(i+1)]]
            caption_txts=[info[img_id][cap_idx] for img_id in img_ids[self.batch_size*i:self.batch_size*(i+1)]]

            caption_inputs=self.processor(text=caption_txts,return_tensors='pt',padding=True,truncation=True).to(self.device)
            img_inputs=self.processor(images=imgs,return_tensors='pt').to(self.device)

            with torch.no_grad():
                caption_features=self.model.module.get_text_features(**caption_inputs).detach().cpu()
                img_features=self.model.module.get_image_features(**img_inputs).detach().cpu()

            caption_embed[self.batch_size*i:self.batch_size*(i+1),:]=caption_features
            img_embed[self.batch_size*i:self.batch_size*(i+1),:]=img_features
        print('Feature extraction done!')

        if os.path.exists(store_path+f'/{self.model_name.upper()}_mono_node')==False:
            os.mkdirs(store_path+f'/{self.model_name.upper()}_mono_node')
        torch.save(caption_embed,store_path+f'/{self.model_name.upper()}_mono_node/{self.model_name.lower()}_caption_embed_{flag}.pt')
        torch.save(img_embed,store_path+f'/{self.model_name.upper()}_mono_node/{self.model_name.lower()}_img_embed_{flag}.pt')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint',help='The model checkpoint')
    parser.add_argument('--batch_size',type=int,default=256,help='The batch size of images and texts')
    parser.add_argument('--n_dim_features',type=int,default=512,help='The number of feature dimensions')
    parser.add_argument('--n_gpus',type=int,default=4,help='Specify number of GPUs')
    parser.add_argument('--which_features',type=str,default='train',help='Choose a image or caption features')
    parser.add_argument('--cap_idx',type=int,default=0,help='Choose one caption index')
    parser.add_argument('--store_path',type=str,default='/project/home/p200248',help='Store path of features')

    args=parser.parse_args()
    
    feature_extractor=ClipBlipFeatureExtractor(checkpoint=args.checkpoint,
            batch_size=args.batch_size,
            n_dim_features=args.n_dim_features,
            n_gpus=args.n_gpus
            )
    feature_extractor.get_features(which_features=args.which_features,
            cap_idx=args.cap_idx,
            store_path=args.store_path)
