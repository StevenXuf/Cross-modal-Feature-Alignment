import torch
import os
import sys
import argparse
import torch.distributed as dist
import torch.nn as nn
import torchvision.io as io
import torchextractor as tx

from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor,AutoModel
from transformers import set_seed
from timm.models.vision_transformer import Block
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss

import torchvision.transforms as transforms
from torchvision.models.feature_extraction import get_graph_node_names,create_feature_extractor
from torchvision import models

from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP

from load_data import LoadCOCO,LoadFlickr30K,LoadIMDB,LoadCelebA
from sampler import DistributedEvalSampler
sys.path.append('/home/fxu/alignment/MetaTransformer_changed/Data2Seq')
#from Data2Seq import Data2Seq
from convit_models import convit_base
from ConvNeXt import convnext_small

sys.path.append('/home/fxu/alignment/BLIP_Official')
from models.blip import blip_feature_extractor
from models.blip_itm import blip_itm

from metatransformer_arch import metatransformer

class FeatureExtractor():    
    def __init__(self,model_name,batch_size=512,n_dim_features=768):
        torch.manual_seed(0)
        dist.init_process_group('nccl')

        self.model_name=model_name.lower()
        self.batch_size=batch_size
        self.n_dim_features=n_dim_features
        
        self.rank=dist.get_rank()
        self.n_total_device=dist.get_world_size()
        
        print(f'Running DDP on rank: {self.rank}. World size: {self.n_total_device}. local GPU number: {torch.cuda.device_count()}.')

        self.device_id=self.rank%torch.cuda.device_count()
        torch.cuda.set_device(self.device_id) 
        
        self.model_list=[]
        if self.model_name=='metatransformer':
            self.model=metatransformer(self.n_dim_features)
            self.model_list.append(self.model)

            self.text_processor=Data2Seq(modality='text',dim=self.n_dim_features,device_id=self.device_id)
            self.img_processor=Data2Seq(modality='image',dim=self.n_dim_features,device_id=self.device_id)
            
        elif self.model_name=='clip':
            self.n_dim_features=512
            self.model=AutoModel.from_pretrained('openai/clip-vit-base-patch32')
            self.model_list.append(self.model)

            self.processor=AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
        elif self.model_name=='blip':
            self.n_dim_features=512
            self.model=AutoModel.from_pretrained('Salesforce/blip-image-captioning-base')
            self.model_list.append(self.model)

            self.processor=AutoProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
        
        elif self.model_name=='blip_official':
            model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth'
            self.model = blip_feature_extractor(pretrained=model_url, image_size=224, vit='base').eval()
            self.model_list.append(self.model)

        elif self.model_name=='blip_official_itm':
            self.n_dim_features=256
            model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
            self.model = blip_itm(pretrained=model_url, image_size=224, vit='base').eval()
            self.model_list.append(self.model)

        elif self.model_name=='resnetbert':
            set_seed(42)
            self.n_dim_features=512
            self.resnet34=models.resnet34(weights='ResNet34_Weights.DEFAULT')
            self.nodes,_=get_graph_node_names(self.resnet34,suppress_diff_warning=True)
            self.resnet_extractor=create_feature_extractor(self.resnet34,return_nodes=[self.nodes[-2]])
            self.model_list.append(self.resnet_extractor)

            self.processor=AutoProcessor.from_pretrained('prajjwal1/bert-small')
            self.bert=AutoModel.from_pretrained('prajjwal1/bert-small')
            self.model_list.append(self.bert)

        elif self.model_name=='convitroberta':
            set_seed(42)
            self.convit_base=convit_base(pretrained=True)
            self.convit_extractor=tx.Extractor(self.convit_base,['norm'])
            self.model_list.append(self.convit_extractor)

            self.processor=AutoProcessor.from_pretrained('roberta-base')
            self.roberta=AutoModel.from_pretrained('roberta-base')
            self.model_list.append(self.roberta)

        elif self.model_name=='convnextxlnet':
            set_seed(42)
            self.convnext_small=convnext_small(pretrained=True,in_22k=False)
            self.small_nodes,_=get_graph_node_names(self.convnext_small,suppress_diff_warning=True)
            self.feature_extractor_small=create_feature_extractor(self.convnext_small,return_nodes=[self.small_nodes[-2]])
            self.model_list.append(self.feature_extractor_small)

            self.processor=AutoProcessor.from_pretrained('xlnet-base-cased')
            self.xlnet=AutoModel.from_pretrained('xlnet-base-cased')
            self.model_list.append(self.xlnet)
        
        else:
            raise Exception('No such a model available.')
        
        for model in self.model_list:
            model.to(self.device_id)
            model=DDP(model,device_ids=[self.device_id])

    def load_dataset(self,which_dataset,store_path):
        if which_dataset=='coco_train':
            dataset,_=LoadCOCO().get_dataset()
            path=store_path+'/MS-COCO/train'
        elif which_dataset=='coco_val':
            _,dataset=LoadCOCO().get_dataset()
            path=store_path+'/MS-COCO/validation'
        elif which_dataset=='flickr30k':
            dataset=LoadFlickr30K().get_dataset()
            path=store_path+'/Flickr30k/flickr30k_images'
        elif which_dataset=='imdb':
            dataset=LoadIMDB().get_dataset()
            path=store_path+'/IMDB/images'
        elif which_dataset=='celebA':
            dataset=LoadCelebA().get_dataset()
            path=store_path+'/CelebA/img_align_celeba'
        else:
            raise Exception('No such a dataset.')
        return dataset,path
    
    def remove_zeros(self,tensor):
        zeros=torch.zeros(self.n_dim_features)
        idx=[]
        for i in range(tensor.size(0)):
            if not torch.equal(tensor[i,:],zeros):
                idx.append(i)
            else:
                print(f'row with all zero entries: {i}')
        return tensor[idx,:]

    def img_transform(self,path,which_dataset,img_ids):
        transform=transforms.Compose([
            lambda img: torch.cat((img,img,img),dim=0) if img.size(0)==1 else img,
            transforms.Resize((224,224),antialias=True),
            lambda img: img/255,
            ])
        img_inputs=[transform(io.read_image(path+f"/data/{'0'*(12-len(img_id))+img_id}.jpg")).unsqueeze(0) if 'coco' in which_dataset else transform(io.read_image(path+f'/{img_id}')).unsqueeze(0) for img_id in img_ids]
        
        return img_inputs

    def clip_and_blip_feature_extraction(self,
            path,
            which_dataset,
            img_ids,
            caption_txts
            ):
        imgs=[Image.open(path+f"/data/{'0'*(12-len(img_id))+img_id}.jpg") if 'coco' in which_dataset else Image.open(path+f"/{img_id}") for img_id in img_ids]
        caption_inputs=self.processor(text=caption_txts,return_tensors='pt',padding=True,truncation=True).to(self.device_id)
        img_inputs=self.processor(images=imgs,return_tensors='pt').to(self.device_id)
        caption_features=self.model.get_text_features(**caption_inputs).detach()
        img_features=self.model.get_image_features(**img_inputs).detach()
        
        return img_features,caption_features

    def blip_official_extraction(self,
            path,
            which_dataset,
            img_ids,
            caption_txts
            ):
        img_features=torch.zeros(len(img_ids),self.n_dim_features).to(self.device_id)
        txt_features=torch.zeros(len(img_ids),self.n_dim_features).to(self.device_id)
        img_inputs=self.img_transform(path,which_dataset,img_ids)
        
        for i,pair in enumerate(zip(img_inputs,caption_txts)):
            image=pair[0].to(self.device_id)
            caption=pair[1]
            image_feature = self.model(image, caption, mode='image')[0,0]
            img_features[i,:]=image_feature
            text_feature = self.model(image, caption, mode='text')[0,0]
            txt_features[i,:]=text_feature
        
        return img_features,txt_features

    def blip_official_itm_extraction(self,
            path,
            which_dataset,
            img_ids,
            caption_txts
            ):
        img_features=torch.zeros(len(img_ids),self.n_dim_features).to(self.device_id)
        txt_features=torch.zeros(len(img_ids),self.n_dim_features).to(self.device_id)
        img_inputs=self.img_transform(path,which_dataset,img_ids)

        for i,pair in enumerate(zip(img_inputs,caption_txts)):
            image=pair[0].to(self.device_id)
            caption=pair[1]
            image_feature,text_feature = self.model(image, caption, match_head='itc')
            img_features[i,:]=image_feature
            txt_features[i,:]=text_feature

        return img_features,txt_features
    def metatransformer_feature_extraction(self,
            path,
            which_dataset,
            img_ids,
            caption_txts
            ):
        img_inputs=self.img_transform(path,which_dataset,img_ids)
        img_inputs=self.img_processor(torch.cat(img_inputs,dim=0).to(self.device_id))
        caption_inputs=self.text_processor(caption_txts).to(self.device_id)
        caption_features=self.model(caption_inputs).detach().squeeze(1)
        img_features=torch.mean(self.model(img_inputs).detach(),dim=1)
        
        return img_features,caption_features

    def resnetbert_feature_extraction(
            self,
            path,
            which_dataset,
            img_ids,
            caption_txts
            ):
        img_inputs=self.img_transform(path,which_dataset,img_ids)
        img_inputs=torch.cat(img_inputs,dim=0).to(self.device_id)

        caption_inputs=self.processor(caption_txts,return_tensors='pt',truncation=True,padding=True).to(self.device_id)
        caption_features=self.bert(**caption_inputs).last_hidden_state.detach()[:,0,:]
        img_features=self.resnet_extractor(img_inputs)[self.nodes[-2]].detach()
        #img_features=torch.cat((img_features,torch.zeros(img_features.size(0),self.n_dim_features-img_features.size(1)).to(self.device_id)),dim=1)

        return img_features,caption_features

    def convitroberta_feature_extraction(self,
            path,
            which_dataset,
            img_ids,
            caption_txts
            ):
        img_inputs=self.img_transform(path,which_dataset,img_ids)
        img_inputs=torch.cat(img_inputs,dim=0).to(self.device_id)
        caption_inputs=self.processor(caption_txts,return_tensors='pt',truncation=True,padding=True).to(self.device_id)
        caption_features=self.roberta(**caption_inputs).last_hidden_state.detach()
        model_output,img_features=self.convit_extractor(img_inputs)

        caption_features=torch.mean(caption_features,dim=1)
        img_features=torch.mean(img_features['norm'].detach(),dim=1)
        
        return img_features,caption_features

    def convnextxlnet_feature_extraction(self,
            path,
            which_dataset,
            img_ids,
            caption_txts
            ):
        img_inputs=self.img_transform(path,which_dataset,img_ids)
        img_inputs=torch.cat(img_inputs,dim=0).to(self.device_id)
        caption_inputs=self.processor(caption_txts,return_tensors='pt',truncation=True,padding=True,max_length=512).to(self.device_id)
        caption_features=self.xlnet(**caption_inputs).last_hidden_state.detach()
        img_features=self.feature_extractor_small(img_inputs)[self.small_nodes[-2]].detach()
        caption_features=torch.mean(caption_features,dim=1)
        
        return img_features,caption_features

    @torch.no_grad
    def get_features(self,which_dataset,store_path='/data/data_fxu'):
        if self.rank==0:
            print(f'Extracting features from {which_dataset} using {self.model_name}.')
            
        dataset,path=self.load_dataset(which_dataset,store_path)

        n_samples=len(dataset)
        print(f'Total number of data points: {n_samples}')
        
        img_embed=[]
        caption_embed=[]
        
        sampler=DistributedEvalSampler(dataset)
        dataloader=DataLoader(dataset,batch_size=self.batch_size,shuffle=False,sampler=sampler)
        
        start_idx=0
        cnt=0
        for i,batch in enumerate(tqdm(dataloader)):
            caption_txts=batch['text']
            img_ids=batch['img_id']
            current_batch_size=len(caption_txts)
            print(f"batch size on machine {self.rank}: {current_batch_size}")
            
            cnt+=current_batch_size

            if self.model_name=='metatransformer':
                pass
                #img_features,caption_features=self.metatransformer_feature_extraction(path,which_dataset,img_ids,caption_txts)
            elif self.model_name=='clip' or self.model_name=='blip':
                img_features,caption_features=self.clip_and_blip_feature_extraction(path,which_dataset,img_ids,caption_txts)              
            elif self.model_name=='blip_official':
                img_features,caption_features=self.blip_official_extraction(path,which_dataset,img_ids,caption_txts)
            elif self.model_name=='blip_official_itm':
                img_features,caption_features=self.blip_official_itm_extraction(path,which_dataset,img_ids,caption_txts)
            elif self.model_name=='resnetbert':
                img_features,caption_features=self.resnetbert_feature_extraction(path,which_dataset,img_ids,caption_txts)
            elif self.model_name=='convitroberta':
                img_features,caption_features=self.convitroberta_feature_extraction(path,which_dataset,img_ids,caption_txts)
            elif self.model_name=='convnextxlnet':
                img_features,caption_features=self.convnextxlnet_feature_extraction(path,which_dataset,img_ids,caption_txts) 
            print(f'feature size on machine {self.rank}: {caption_features.size()}')
            
            img_embed.append(img_features)
            caption_embed.append(caption_features)
            
        print(f'length of img_embed on local machine: {len(img_embed)}/{len(caption_embed)}')

        img_embed=torch.cat(img_embed,dim=0)
        caption_embed=torch.cat(caption_embed,dim=0)

        print(f'local image embed size: {img_embed.size()}')
        print(f'local caption embed size: {caption_embed.size()}')

        batch_size_list=[None for _ in range(self.n_total_device)]
        dist.all_gather_object(batch_size_list,cnt)
        dist.barrier()
        print(f'total batch size on each machine: {batch_size_list}')
        
        #use torch.empty to initialize each tensor in tensor list!!! DO NOT use torch.zeros!
        gather_list_cap=[torch.empty(nrow,self.n_dim_features).to(self.device_id) for nrow in batch_size_list]
        gather_list_img=[torch.empty(nrow,self.n_dim_features).to(self.device_id) for nrow in batch_size_list]
        print(f'gather list length: {list(map(len,gather_list_cap))}')
        
        dist.all_gather(gather_list_cap,caption_embed)
        dist.all_gather(gather_list_img,img_embed)
        dist.barrier()
        print('Feature extraction done!')
        
        if self.rank==0 and os.path.exists(store_path+f'/alignment_features/{self.model_name.upper()}')==False:
            os.makedirs(store_path+f'/alignment_features/{self.model_name.upper()}')
        
        if self.rank==0:
            caption_embed_final=torch.cat(gather_list_cap,dim=0).cpu()
            img_embed_final=torch.cat(gather_list_img,dim=0).cpu()
            print(f'embedding size: {caption_embed_final.size()}/{img_embed_final.size()}')

            torch.save(caption_embed_final,store_path+f'/alignment_features/{self.model_name.upper()}/{self.model_name.lower()}_caption_embed_{which_dataset}.pt')
            torch.save(img_embed_final,store_path+f'/alignment_features/{self.model_name.upper()}/{self.model_name.lower()}_img_embed_{which_dataset}.pt')
            print('Features Saved!!!')

            self.remove_zeros(caption_embed_final)
            self.remove_zeros(img_embed_final)
        dist.destroy_process_group()

    def load_image_features(self,which_dataset,store_path='/data/data_fxu/alignment_features'):
        return torch.load(store_path+f'/{self.model_name.upper()}/{self.model_name.lower()}_img_embed_{which_dataset}.pt')

    def load_caption_features(self,which_dataset,store_path='/data/data_fxu/alignment_features'):
        return torch.load(store_path+f'/{self.model_name.upper()}/{self.model_name.lower()}_caption_embed_{which_dataset}.pt')
    
    def find_correspondence(self,
            img_embed,
            cap_embed,
            img_features,
            cap_features
            ):
        img_loss=torch.empty(img_features.size(0),img_embed.size(0))
        cap_loss=torch.empty(cap_features.size(0),cap_embed.size(0))
        min_img_idx=[]
        min_cap_idx=[]
        for i in range(img_features.size(0)):
            img_loss[i,:]=torch.mean((img_features[i,:]-img_embed)**2,dim=1)
            cap_loss[i,:]=torch.mean((cap_features[i,:]-cap_embed)**2,dim=1)
            min_img_idx.append(img_loss[i,:].tolist().index(min(img_loss[i,:].tolist())))
            min_cap_idx.append(cap_loss[i,:].tolist().index(min(cap_loss[i,:].tolist())))
        print(f'img idx: {min_img_idx}')
        print(f'cap idx: {min_cap_idx}')
        print(f'sampled img embed: {img_embed[min_img_idx,:5]}')
        print(f'calculated img embed: {img_features[:,:5]}')
        print(f'sampled cap embed: {cap_embed[min_cap_idx,:5]}')
        print(f'calculated cap embed: {cap_features[:,:5]}')

    def check_correctness(self,which_dataset,store_path='/data/data_fxu/alignment_features'):
        if self.rank==0: 
            img_embed=self.load_image_features(which_dataset,store_path)
            cap_embed=self.load_caption_features(which_dataset,store_path)
            dataset,path=self.load_dataset(which_dataset,store_path)
            print(f'loaded embeding size: {img_embed.size()}/{cap_embed.size()}')

            rand_idx=torch.randperm(len(dataset))[:10]
            txts=[dataset[idx]['text'] for idx in rand_idx]
            img_ids=[dataset[idx]['img_id'] for idx in rand_idx]
            
            print(f'text and img_ids length: {len(txts)}/{len(img_ids)}')
            if self.model_name=='metatransformer':
                img_features,cap_features=self.metatransformer_feature_extraction(path,which_dataset,img_ids,txts)
                print(f'img_features and cap_features size: {img_features.size()}/{cap_features.size()}')
                self.find_correspondence(img_embed,cap_embed,img_features.cpu(),cap_features.cpu())

            elif self.model_name=='clip' or self.model_name=='blip':
                img_features,cap_features=self.clip_and_blip_feature_extraction(path,which_dataset,img_ids,txts)
                print(f'img_features and cap_features size: {img_features.size()}/{cap_features.size()}')
                self.find_correspondence(img_embed,cap_embed,img_features.cpu(),cap_features.cpu())

            elif self.model_name=='resnetbert':
                img_features,cap_features=self.resnetbert_feature_extraction(path,which_dataset,img_ids,txts)
                print(f'img_features and cap_features size: {img_features.size()}/{cap_features.size()}')
                self.find_correspondence(img_embed,cap_embed,img_features.cpu(),cap_features.cpu())
            elif self.model_name=='convitroberta':
                img_features,cap_features=self.convitroberta_feature_extraction(path,which_dataset,img_ids,txts)
                print(f'img_features and cap_features size: {img_features.size()}/{cap_features.size()}')
                self.find_correspondence(img_embed,cap_embed,img_features.cpu(),cap_features.cpu())

            elif self.model_name=='convnextxlnet':
                img_features,cap_features=self.convnextxlnet_feature_extraction(path,which_dataset,img_ids,txts)
                print(f'img_features and cap_features size: {img_features.size()}/{cap_features.size()}')
                self.find_correspondence(img_embed,cap_embed,img_features.cpu(),cap_features.cpu())

            else:
                raise Exception('No such model.')

def set_config():
    parser=argparse.ArgumentParser()
    parser.add_argument('model_name',type=str)
    parser.add_argument('which_dataset',type=str,default='coco_train',help='Choose a image or caption features')
    parser.add_argument('--batch_size',type=int,default=512,help='The batch size of images and texts')
    parser.add_argument('--n_dim_features',type=int,default=768,help='The number of feature dimensions')
    parser.add_argument('--store_path',type=str,default='/data/data_fxu',help='Store path of features')

    args=parser.parse_args()

    return args

if __name__=='__main__':
    args=set_config()

    feature_extractor=FeatureExtractor(
            args.model_name,
            batch_size=args.batch_size,
            n_dim_features=args.n_dim_features,
            )
    #feature_extractor.get_features(args.which_dataset,store_path=args.store_path)
    data_name='flickr30k'
    img_feats=feature_extractor.load_image_features(data_name)
    txt_feats=feature_extractor.load_caption_features(data_name)
    print(img_feats.size())
    print(txt_feats.size())
