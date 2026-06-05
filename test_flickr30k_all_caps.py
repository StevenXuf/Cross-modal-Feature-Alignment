import torch
import sys
import torch.nn as nn
import torchvision.transforms as transforms

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, AutoModel
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

sys.path.append('/home/fxu/alignment/BLIP_Official')
from models.blip_itm import blip_itm


def plot_norm_dist(embed):
    norms=embed.norm(p=2,dim=-1)
    plt.bar(range(norms.size(0)),norms)
    
    plt.savefig('./norm_dist.py')

transform=transforms.Compose([transforms.Resize((224,224),antialias=True),transforms.ToTensor()])

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

extractor='clip'
task='img2txt'

if extractor.lower()=='clip':
    model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
elif extractor.lower()=='blip':
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
    model = blip_itm(pretrained=model_url, image_size=224, vit='base').eval().to(device)

# Load the Flickr30k dataset from the Hugging Face Hub
dataset = load_dataset('nlphuji/flickr30k')['test']
test_dataset = dataset.filter(lambda example: example['split'] == 'test')

text_features=[]
image_features=[]
for i in tqdm(range(len(test_dataset))):
    example=test_dataset[i]
    image = example['image'].convert('RGB')  # PIL Image object
    captions = example['caption']  # List of 5 captions
    
    if extractor.lower()=='clip':
        caption_inputs=processor(text=captions,return_tensors='pt',padding=True,truncation=True).to(device)
        img_inputs=processor(images=image,return_tensors='pt').to(device)
        
        text_features.append(model.get_text_features(**caption_inputs).detach())
        image_features.append(model.get_image_features(**img_inputs).detach())
    elif extractor.lower()=='blip':
        image=transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_feature,text_feature = model(image, captions, match_head='itc')
        text_features.append(text_feature)
        image_features.append(image_feature)

text_features=torch.cat(text_features,dim=0)
image_features=torch.cat(image_features,dim=0)

print(text_features.size())

cosine=pairwise_cosine_similarity(text_features,image_features)

for k in [1,5,10]:
    cnt=0
    if task=='img2txt':
        vals,ids=torch.topk(cosine.T,k=k)
        for j in range(ids.size(0)):
            cnt+=len(list(set(ids[j].tolist())&set(range(j*5,(j+1)*5))))
        print(cnt)
        print(f'Top-{k} Precision: {cnt/(ids.size(0)*k)*100:.2f}')
    elif task=='txt2img':
        vals,ids=torch.topk(cosine,k=k)
        for j in range(ids.size(0)):
            cnt+=len(list(set(ids[j].tolist())&set(range(j//5*1,(j//5+1)*1))))
        print(cnt)
        print(f'Top-{k} Precision: {cnt/(ids.size(0)*k)*100:.2f}')
