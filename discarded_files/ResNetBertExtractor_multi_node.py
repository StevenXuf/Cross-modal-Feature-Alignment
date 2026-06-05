import torch
from tqdm import tqdm
import argparse
import os
import torch.distributed as dist
import torchvision.transforms as transforms

import torchvision.io as io
from torchvision.models.feature_extraction import get_graph_node_names,create_feature_extractor
from torchvision import models

from transformers import AutoTokenizer,AutoModel

from torch.utils.data import DataLoader,Subset

from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP

from load_data import LoadFlickr30K,LoadIMDB,LoadCelebA,LoadCOCO

class SepImageTextFeatureExtractor():
    def __init__(self,n_dim_features=768,batch_size=512,model_name='ResNetBert'):
        self.batch_size=batch_size
        self.n_dim_features=n_dim_features

        dist.init_process_group('nccl')
        self.rank=dist.get_rank()
        self.n_total_device=dist.get_world_size()
        print(f'Running DDP on rank: {self.rank}.')

        self.device_id=self.rank%torch.cuda.device_count()

        self.model_name=model_name

        self.resnet34=models.resnet34(weights='ResNet34_Weights.DEFAULT')
        self.nodes,_=get_graph_node_names(self.resnet34,suppress_diff_warning=True)
        self.resnet_extractor=create_feature_extractor(self.resnet34,return_nodes=[self.nodes[-2]]).to(self.device_id)
        self.feature_extractor=DDP(self.resnet_extractor,device_ids=[self.device_id])
        
        self.text_processor=AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert=AutoModel.from_pretrained('bert-base-uncased').to(self.device_id)
        self.bert=DDP(self.bert,device_ids=[self.device_id])
    
    @record
    @torch.no_grad
    def get_features(self,which_dataset='coco_train',store_path='/project/home/p200248/alignment'):
        if which_dataset=='coco_train':
            dataset,_=LoadCOCO().get_dataset()
            path=store_path+'/MS-COCO/train/data'
        elif which_dataset=='coco_val':
            _,dataset=LoadCOCO().get_dataset()
            path=store_path+'/MS-COCO/validation/data'
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
            raise Exception('No such a dataset!')

        n_samples=len(dataset)

        img_embed=torch.zeros(n_samples,self.n_dim_features).to(self.device_id)
        caption_embed=torch.zeros(n_samples,self.n_dim_features).to(self.device_id)

        if self.batch_size==1:
            n_batch=n_samples//self.batch_size
        else:
            n_batch=n_samples//self.batch_size+1

        batch_partition_size=n_batch//self.n_total_device

        if self.rank==self.n_total_device-1:
            batch_range=range(self.rank*batch_partition_size*self.batch_size,n_samples)
        else:
            batch_range=range(self.rank*batch_partition_size*self.batch_size,(self.rank+1)*batch_partition_size*self.batch_size)

        subset=Subset(dataset,list(batch_range))
        dataloader=DataLoader(subset,batch_size=self.batch_size)

        transform=transforms.Compose([
            transforms.Resize((224,224),antialias=True),
            lambda img: torch.cat((img,img,img),dim=0) if img.size(0)==1 else img,
            lambda img: img/255,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        for i,batch in enumerate(tqdm(dataloader)):

            img_inputs=[]
            for img_id in batch['img_id']:
                if 'coco' in which_dataset:
                    img=transform(io.read_image(path+f"/{'0'*(12-len(img_id))+img_id}.jpg")).unsqueeze(0)
                else:
                    img=transform(io.read_image(path+f'/{img_id}')).unsqueeze(0)
                img_inputs.append(img)

            img_inputs=torch.cat(img_inputs,dim=0).to(self.device_id)

            caption_txts=batch['text']
            current_batch_size=len(caption_txts)

            caption_inputs=self.text_processor(caption_txts,return_tensors='pt',truncation=True,padding=True).to(self.device_id)

            caption_features=self.bert(**caption_inputs).last_hidden_state.detach()
            img_features=self.resnet_extractor(img_inputs)[self.nodes[-2]].detach()

            caption_embed[(batch_range[0]+i*current_batch_size):(batch_range[0]+(i+1)*current_batch_size),:]=torch.mean(caption_features,dim=1)
            img_embed[(batch_range[0]+i*current_batch_size):(batch_range[0]+(i+1)*current_batch_size),:]=torch.cat((img_features,torch.zeros(img_features.size(0),self.n_dim_features-img_features.size(1)).to(self.device_id)),dim=1)
        print('Feature extraction done!')

        dist.all_reduce(caption_embed,op=dist.ReduceOp.SUM)
        dist.all_reduce(img_embed,op=dist.ReduceOp.SUM)

        if os.path.exists(store_path+f'/{self.model_name.upper()}')==False:
            os.makedirs(store_path+f'/{self.model_name.upper()}')
        torch.save(caption_embed,store_path+f'/{self.model_name.upper()}/{self.model_name.lower()}_caption_embed_{which_dataset}.pt')
        torch.save(img_embed,store_path+f'/{self.model_name.upper()}/{self.model_name.lower()}_img_embed_{which_dataset}.pt')

        dist.destroy_process_group()

def set_config():
    parser=argparse.ArgumentParser()
    parser.add_argument('--n_dim_features',default=768,type=int,help='feature dimensions')
    parser.add_argument('--batch_size',default=512,type=int,help='batch size')
    parser.add_argument('--which_dataset',type=str,default='coco_train',help='Choose a dataset')

    args=parser.parse_args()

    return args

def main():
    args=set_config()

    feature_extractor=SepImageTextFeatureExtractor(
            n_dim_features=args.n_dim_features,
            batch_size=args.batch_size
            )
    feature_extractor.get_features(args.which_dataset)

if __name__=='__main__':
    main()
