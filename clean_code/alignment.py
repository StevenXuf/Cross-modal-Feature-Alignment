import os
import torch
import torch.optim as optim

from tqdm import tqdm

from load_data import LoadIMDB,LoadCOCO,LoadFlickr30K

from torch.utils.data import DataLoader,ConcatDataset
from transformers import BertTokenizer
from accelerate import Accelerator
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from torchmetrics.classification import MulticlassPrecision,MulticlassRecall

from models import ResNet34Encoder,BertEncoder
from loss_module import InfoNCE,CustomContrastiveLoss

class Main():
    def __init__(self,seed,embed_dim,temp,batch_size,n_epochs,dataset,ckpt_path,vision_encoder,text_encoder,optimizer):
        # Parameters
        self.accelerate=Accelerator()
        self.accelerate.debug=True
        self.embed_dim = embed_dim
        self.temperature = temp
        self.batch_size=batch_size
        self.n_epochs=n_epochs
        self.device=self.accelerate.device
        self.seed=seed
        self.ckpt_path=ckpt_path
        self.vision_encoder,self.text_encoder,self.optimizer,self.dataset=self.accelerate.prepare(vision_encoder.to(self.device),text_encoder.to(self.device),optimizer,dataset)
        self.tokenizer=self.text_encoder.module.get_tokenizer()
        
    def load_ckpt(self,ckpt_path,modules,optimizer):
        if os.path.isfile(ckpt_path):
            ckpt=torch.load(ckpt_path,weights_only=True)
            for module_name in modules.keys():
                modules[module_name].load_state_dict(ckpt['model_state_dict'][module_name])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            return ckpt['epoch'], ckpt['loss']
        else:
            return 0,None
    
    def save_ckpt(self,epoch,model_dict,optimizer,loss,ckpt_path):
        torch.save({
            'epoch':epoch,
            'model_state_dict':model_dict,
            'optimizer_state_dict':optimizer.state_dict(),
            'loss':loss
            },ckpt_path)
        print('Model saved!')

    def train(self):
        # Initialize models, loss, optimizer
        #contrastive_loss = InfoNCE(self.temperature).to(self.device)
        contrastive_loss=CustomContrastiveLoss(temperature=.1).to(self.device)

        # Training
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        
        start_epoch,_=self.load_ckpt(self.ckpt_path,{'vision_proj':self.vision_encoder.module.fc,'text_proj':self.text_encoder.module.fc},self.optimizer)
        
        self.vision_encoder.train()
        self.text_encoder.train()

        for epoch in range(start_epoch,self.n_epochs):
            total_loss=.0
            for i,batch in enumerate(tqdm(dataloader)):
                images = batch['image'].to(self.device)
                encoded_captions= self.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True).to(self.device)
                # Forward pass
                image_embeddings = self.vision_encoder(images)
                text_embeddings = self.text_encoder(encoded_captions['input_ids'],encoded_captions['attention_mask'])
                
                # Compute loss
                loss = contrastive_loss(image_embeddings, text_embeddings)
                
                # Backpropagation
                self.optimizer.zero_grad()
                #loss.backward()
                self.accelerate.backward(loss)
                self.optimizer.step()

                total_loss+=loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss/(i+1):.4f}")
            print('*'*20)                
            self.save_ckpt(epoch+1,{'vision_proj':self.vision_encoder.module.fc.state_dict(),'text_proj':self.text_encoder.module.fc.state_dict()},self.optimizer,total_loss,self.ckpt_path)
    
    def get_metrics(self,cosine):
        n_class=cosine.size(1)
        labels=torch.arange(n_class).to(self.device)

        compute_precision=MulticlassPrecision(num_classes=n_class).to(self.device)
        compute_recall=MulticlassRecall(num_classes=n_class).to(self.device)

        precision=compute_precision(cosine,labels)
        recall=compute_recall(cosine,labels)
        
        return precision,recall

class Test():
    def __init__(vision_encoder,text_encoder,ckpt_path,dataset,batch_size):
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vision_encoder=vision_encoder
        self.text_encoder=text_encoder
        self.ckpt=ckpt_path
        self.dataset=dataset
        self.batch_size=batch_size

    @torch.no_grad
    def test(self):
        ckpt=torch.load(self.ckpt_path,weights_only=True)
        self.vision_encoder.fc.load_state_dict(ckpt['model_state_dict']['vision_proj'])
        self.text_encoder.fc.load_state_dict(ckpt['model_state_dict']['text_proj'])
        
        self.vision_encoder.eval()
        self.text_encoder.eval()

        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        img2txt_precision=.0
        img2txt_recall=.0
        
        txt2img_precision=.0
        txt2img_recall=.0
        
        for i,batch in enumerate(tqdm(dataloader)):
            images,texts=batch['image'].to(self.device),batch['text']
            encoded_captions= self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
            # Forward pass
            image_embeddings = self.vision_encoder(images)
            text_embeddings = self.text_encoder(encoded_captions['input_ids'],encoded_captions['attention_mask'])
            
            cosine=pairwise_cosine_similarity(image_embeddings,text_embeddings)
            img2txt_pre,img2txt_re=self.get_metrics(cosine)
            img2txt_precision+=img2txt_pre
            img2txt_recall+=img2txt_re

            txt2img_pre,txt2img_re=self.get_metrics(cosine.T)
            txt2img_precision+=txt2img_pre
            txt2img_recall+=txt2img_re

        print('Image-to-text')
        print(f'Precsion:{img2txt_precision/(i+1)*100:.2f}%')
        print(f'Recall:{img2txt_recall/(i+1)*100:.2f}%')

        print('Text-to-image')
        print(f'Precsion:{txt2img_precision/(i+1)*100:.2f}%')
        print(f'Recall:{txt2img_recall/(i+1)*100:.2f}%')

if __name__=='__main__':
    seed=0
    dim=256
    temp=.07
    lr=1e-4
    batch_size=64
    n_epochs=100
    dataset=LoadIMDB().get_dataset()
    ckpt_dir='/data/data_fxu/ckpt_resnet34_bert/'
    os.makedirs(ckpt_dir,exist_ok=True)
    ckpt_path=ckpt_dir+'imdb_custom_kpt.pth'
   
    torch.manual_seed(seed)
    vision_encoder = ResNet34Encoder(dim)
    text_encoder = BertEncoder(dim)
    optimizer = optim.AdamW(
            list(vision_encoder.parameters()) + list(text_encoder.parameters()), lr=lr)
    #lr_scheduler important for joint training since the loss plataeus!!!
    main=Main(seed,dim,temp,batch_size,n_epochs,dataset,ckpt_path,vision_encoder,text_encoder,optimizer)
    main.train()
    main.test(ckpt_path,dataset)
