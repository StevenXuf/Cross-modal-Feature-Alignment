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
from timm.models.vision_transformer import Block
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.models.feature_extraction import get_graph_node_names,create_feature_extractor
from torchvision import models

from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP

from load_data import LoadCOCO,LoadFlickr30K,LoadIMDB,LoadCelebA
sys.path.append('/home/users/u101139/alignment/MetaTransformer_changed/Data2Seq')
from Data2Seq import Data2Seq

resnet34=models.resnet34(weights='ResNet34_Weights.DEFAULT')
nodes,_=get_graph_node_names(resnet34,suppress_diff_warning=True)
resnet_extractor=create_feature_extractor(resnet34,return_nodes=[nodes[-2]])

processor=AutoProcessor.from_pretrained('bert-base-uncased')
bert=AutoModel.from_pretrained('bert-base-uncased')

dataset=LoadIMDB().get_dataset()
print(dataset[1025])

img_id=dataset[1025]['img_id']
txt=dataset[1025]['text']

transform=transforms.Compose([
            lambda img: torch.cat((img,img,img),dim=0) if img.size(0)==1 else img,
            transforms.Resize((224,224),antialias=True),
            lambda img: img/255,
            ])
path='/project/home/p200248/alignment/IMDB/images'
img_inputs=transform(io.read_image(path+f"/{img_id}")).unsqueeze(0)

caption_inputs=processor(txt,return_tensors='pt',truncation=True,padding=True).to(device_id)
caption_features=bert(**caption_inputs).last_hidden_state.detach()
caption_features=torch.mean(caption_features,dim=1)
img_features=resnet_extractor(img_inputs)[nodes[-2]].detach()
img_features=torch.cat((img_features,torch.zeros(img_features.size(0),n_dim_features-img_features.size(1)).to(device_id)),dim=1)


