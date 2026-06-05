import torch
import sys
import argparse
import torchvision.io as io
import torchextractor as tx

from tqdm import tqdm
from transformers import AutoProcessor,AutoModel
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.models.feature_extraction import get_graph_node_names,create_feature_extractor
from torchvision import models

from load_data import LoadCOCO,LoadFlickr30K,LoadIMDB
sys.path.append('/home/fxu/alignment/MetaTransformer_changed/Data2Seq')
from Data2Seq import Data2Seq
from convit_models import convit_base
from ConvNeXt import convnext_small

sys.path.append('/home/fxu/alignment/BLIP_Official/models')
from BLIP_Official.models.blip import blip_feature_extractor
from BLIP_Official.models.blip_itm import blip_itm

from metatransformer_arch import metatransformer

class FeatureExtractor():    
    def __init__(self,model_name,device=0):
        self.model_name=model_name.lower()
        self.device=device
        
    def get_extractor(self):
        if self.model_name=='metatransformer':
            self.model=metatransformer(768).to(self.device)
            
            self.text_processor=Data2Seq(modality='text',dim=768,device_id=self.device)
            self.img_processor=Data2Seq(modality='image',dim=768,device_id=self.device)
            return self.model,self.text_processor,self.img_processor

        elif self.model_name=='clip':
            self.model=AutoModel.from_pretrained('openai/clip-vit-base-patch32').to(self.device)

            self.processor=AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
            return self.model,self.processor

        elif self.model_name=='blip':
            self.model=AutoModel.from_pretrained('Salesforce/blip-image-captioning-base').to(self.device)

            self.processor=AutoProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
            return self.model,self.processor

        elif self.model_name=='blip_official':
            model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth'
            self.model = blip_feature_extractor(pretrained=model_url, image_size=224, vit='base').to(self.device).eval()
            return self.model

        elif self.model_name=='blip_official_itm':
            model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
            self.model = blip_itm(pretrained=model_url, image_size=224, vit='base').to(self.device).eval()
            return self.model

        elif self.model_name=='resnetbert':
            self.resnet34=models.resnet34(weights='ResNet34_Weights.DEFAULT')
            self.nodes,_=get_graph_node_names(self.resnet34,suppress_diff_warning=True)
            self.resnet_extractor=create_feature_extractor(self.resnet34,return_nodes=[self.nodes[-2]]).to(self.device)

            self.processor=AutoProcessor.from_pretrained('bert-base-uncased')
            self.bert=AutoModel.from_pretrained('bert-base-uncased').to(self.device)
            
            return self.resnet_extractor,self.bert,self.processor

        elif self.model_name=='convitroberta':
            self.convit_base=convit_base(pretrained=True)
            self.convit_extractor=tx.Extractor(self.convit_base,['norm']).to(self.device)

            self.processor=AutoProcessor.from_pretrained('roberta-base')
            self.roberta=AutoModel.from_pretrained('roberta-base').to(self.device)
            
            return self.convit_extractor,self.roberta,self.processor

        elif self.model_name=='convnextxlnet':
            self.convnext_small=convnext_small(pretrained=True,in_22k=False)
            self.small_nodes,_=get_graph_node_names(self.convnext_small,suppress_diff_warning=True)
            self.feature_extractor_small=create_feature_extractor(self.convnext_small,return_nodes=[self.small_nodes[-2]]).to(self.device)

            self.processor=AutoProcessor.from_pretrained('xlnet-base-cased')
            self.xlnet=AutoModel.from_pretrained('xlnet-base-cased').to(self.device)
            
            return self.feature_extractor_small,self.xlnet,self.processor

class FeatureExtraction():
    def __init__(self,model_name,device=0):
        self.model_name=model_name.lower()
        self.device=device
        self.FeatureExtractor=FeatureExtractor(self.model_name,device=self.device)
    
    def no_gradients(self,model):
        for param in model.parameters():
            param.require_grad=False

    @torch.no_grad
    def get_features(self,images,texts):
        images=images.to(self.device)

        if self.model_name=='clip' or self.model_name=='blip':
            model,processor=self.FeatureExtractor.get_extractor()
            inputs=processor(text=texts,return_tensors='pt',padding=True,truncation=True).to(self.device)
            caption_features=model.get_image_features(images).detach()
            img_features=model.get_text_features(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask']).detach()
            
            return img_features,caption_features

        elif self.model_name=='blip_official_itm':
            model=self.FeatureExtractor.get_extractor()

            image_features,text_features = model(images, texts, match_head='itc')

            return image_features,text_features

        elif self.model_name=='metatransformer':
            model,text_processor,img_processor=self.FeatureExtractor.get_extractor()
            img_inputs=img_processor(images)
            caption_inputs=text_processor(texts)
            caption_features=model(caption_inputs).detach().squeeze(1)
            img_features=torch.mean(model(img_inputs).detach(),dim=1)
            
            return img_features,caption_features
        
        elif self.model_name=='resnetbert':
            img_encoder,txt_encoder,processor=self.FeatureExtractor.get_extractor() 
            caption_inputs=processor(texts,return_tensors='pt',truncation=True,padding=True).to(self.device)
            caption_features=txt_encoder(**caption_inputs).last_hidden_state.detach()[:,0,:]
            img_features=img_encoder(images)['flatten'].detach()
           
            print(img_encoder)
            return img_features,caption_features

        elif self.model_name=='convitroberta':
            img_encoder,txt_encoder,processor=self.FeatureExtractor.get_extractor() 
            caption_inputs=processor(texts,return_tensors='pt',truncation=True,padding=True).to(self.device)
            caption_features=txt_encoder(**caption_inputs).last_hidden_state.detach()[:,0,:]
            model_output,img_features=img_encoder(images)
            
            img_features=img_features['norm'].detach()[:,0,:]
            
            return img_features,caption_features

        elif self.model_name=='convnextxlnet':
            img_encoder,txt_encoder,processor=self.FeatureExtractor.get_extractor() 
            caption_inputs=processor(texts,return_tensors='pt',truncation=True,padding=True,max_length=512).to(self.device)
            caption_features=txt_encoder(**caption_inputs).last_hidden_state.detach()[:,0,:]
            img_features=img_encoder(images)['norm'].detach()
            
            return img_features,caption_features

        else:
            raise Exception('No such a model available.')

def set_config():
    parser=argparse.ArgumentParser()
    parser.add_argument('model_name',type=str)

    args=parser.parse_args()

    return args

if __name__=='__main__':
    torch.manual_seed(0)

    args=set_config()
    dataset=LoadIMDB().get_dataset()
    dataloader=DataLoader(dataset,batch_size=256,shuffle=True)
    feature_extractor=FeatureExtraction(
            args.model_name
            )
    for batch in tqdm(dataloader):
        images=batch['image']
        texts=batch['text']
        img_feat,txt_feat=feature_extractor.get_features(images,texts)
        print(img_feat.size())
        print(txt_feat.size())
