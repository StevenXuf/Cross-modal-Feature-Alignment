import torch
import timm
import numpy as np
import transformers
import matplotlib.pyplot as plt

from tqdm import tqdm
#from tqdm.notebook import tqdm #if using notebook

from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torchmetrics.functional.pairwise import pairwise_cosine_similarity

from transformers import AutoProcessor,AutoModel
import sys
sys.path.append('/home/fxu/alignment/BLIP_Official')
from models.blip_itm import blip_itm

from datasets import load_dataset
from huggingface_hub import list_datasets

from convit_models import convit_base
from ConvNeXt import convnext_small

def preprocess(examples):
    transform=transforms.Compose([
            transforms.Resize((224,224),antialias=True),
            transforms.ToTensor(),
            lambda img:torch.cat((img,img,img)) if img.size(0)==1 else img
            ])
    examples['image'] = [transform(img) for img in examples['image']]
    examples['label'] = [lbl for lbl in examples['label']]
    return examples

def string_label(idx):
    return imagenet1k_val.features['label'].names[idx]

def plot_images(images,labels):
    size=len(labels)
    fig,axes=plt.subplots(1,size,figsize=(size*3,3))
    for i in range(size):
        img=images[i].permute(1,2,0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(string_label(labels[i]),fontsize=6)
        axes[i].axis('off')

def compute_vision_model_acc(model,dataloader,device,eps=.0):
    model=model.to(device)
    total=0
    correct=0
    with torch.no_grad():
        for i,batch in tqdm(enumerate(dataloader)):
            images = (1-eps)*batch['image'].to(device)+eps*torch.rand(*batch['image'].size()).to(device)
            labels = batch['label'].to(device)
            model.eval()
            outputs=model(images)
            
            _,preds=torch.max(outputs.data,dim=1)
            correct+=(preds==labels).sum().item()
            total+=len(labels)
        acc=100*correct/total
        print(f'{model.__class__.__name__} accuracy with noise {eps}: {acc}')

def get_dataloader(split='validation',batch_size=64):
    dataset=load_dataset('ILSVRC/imagenet-1k',split=split,streaming=True,trust_remote_code=True)
    classes=dataset.features['label'].names
    dataset=dataset.map(preprocess, batched=True)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return classes,dataloader

def compute_multimodal_acc(model,dataloader,device,eps=0):
    model=model.to(device)
    classes,_=get_dataloader()
    text=[f"a picture of {name.strip().split(',')[0]}" for name in classes]
    with torch.no_grad():
        if model.__class__.__name__=='BLIP_ITM':
            _,caption_features=model(image=torch.zeros(1,3,224,224).to(device),caption=text,match_head='itc')
        elif model.__class__.__name__=='CLIPModel':
            processor=AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
            caption_inputs=processor(text=text,return_tensors='pt',padding=True,truncation=True).to(device)
            caption_features=model.get_text_features(**caption_inputs).detach()
        else:
            raise Exception('No such a model.')

        total=0
        correct=0
        for i,batch in tqdm(enumerate(dataloader)):
            images=(1-eps)*batch['image'].to(device)+eps*torch.rand(*batch['image'].size()).to(device)
            labels=batch['label'].to(device)
            
            if model.__class__.__name__=='BLIP_ITM':
                img_features,_=model(image=images,caption=[''],match_head='itc')
            elif model.__class__.__name__=='CLIPModel':
                img_inputs=processor(images=images,return_tensors='pt',do_rescale=False).to(device)
                img_features=model.get_image_features(**img_inputs).detach()

            cosine=pairwise_cosine_similarity(img_features,caption_features)
            _,preds=torch.max(cosine,dim=1)
            correct+=(preds==labels).sum().item()
            total+=len(labels)
        acc=100*correct/total
        print(f'{model.__class__.__name__} accuracy with noise {eps}: {acc}')

if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _,dataloader=get_dataloader()

    resnet34=models.resnet34(weights='ResNet34_Weights.DEFAULT')
    convit=convit_base(pretrained=True)
    convnext=convnext_small(pretrained=True,in_22k=False)

    clip=AutoModel.from_pretrained('openai/clip-vit-base-patch32')
    clip_processor=AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    itm_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
    itm=blip_itm(pretrained=itm_url, image_size=224, vit='base').eval()

    for noise in torch.arange(0,0.55,.05):
        #compute_vision_model_acc(convnext,dataloader,device,eps=noise)
        compute_multimodal_acc(itm,dataloader,device,eps=noise)
