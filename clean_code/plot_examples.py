import argparse
import torch
import torchvision.io as io
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Subset,DataLoader

from load_data import LoadCOCO,LoadFlickr30K,LoadIMDB
from compute_distance import ComputeDistance

class PlotExamples():
    def __init__(self):
        self.dataset_names=['IMDB','Flickr30k','MS-COCO']
        self.transform=transforms.Compose([
            lambda img: torch.cat((img,img,img),dim=0) if img.size(0)==1 else img,
            transforms.Resize((224,224),antialias=True),
            lambda img: img/255,
            ])

    def load_dataset(self,dataset_name):
        store_path='/data/data_fxu'
        if dataset_name=='MS-COCO':
            dataset,_=LoadCOCO().get_dataset()
            path=store_path+'/MS-COCO/train'
        elif dataset_name=='Flickr30k':
            dataset=LoadFlickr30K().get_dataset()
            path=store_path+'/Flickr30k/flickr30k_images'
        elif dataset_name=='IMDB':
            dataset=LoadIMDB().get_dataset()
            path=store_path+'/IMDB/images'
        else:
            raise Exception('No such a dataset.')
        return dataset,path

    def get_random_pair(self,dataset_name,ids,rand_sub_size=1000):
        dataset,path=self.load_dataset(dataset_name)
        img_ids=[]
        caps=[]
        
        dataset=Subset(dataset,torch.arange(rand_sub_size).tolist())
        subset=Subset(dataset,ids)
        loader=DataLoader(subset,batch_size=len(ids),shuffle=False)
        for _,samples in enumerate(loader):
            img_ids.extend(samples['img_id'])
            caps.extend(samples['text'])

        if 'COCO' in dataset_name:
            imgs=[io.read_image(path+f"/data/{'0'*(12-len(img_id))+img_id}.jpg") for img_id in img_ids]
        else:
            imgs=[io.read_image(path+f"/{img_id}") for img_id in img_ids]

        imgs=[self.transform(img) for img in imgs]
        return caps,imgs

    def show_imgs(self,ids):
        for i in range(len(self.dataset_names)):
            dataset_name=self.dataset_names[i]
            fig,ax=plt.subplots(1,1,figsize=(3,3),layout='constrained')
            
            cap,img=self.get_random_pair(dataset_name,ids)
            im=ax.imshow(img[0].permute(1,2,0))
            ax.axis('off')
            plt.savefig(f'./{self.dataset_names[i]}_examples.pdf')
            print(cap[0])
    
    def cut_caps(self,string,threshold=5):
        tokens=string.strip().split(' ')
        if tokens[-1]=='.':
            tokens[-2]+='.'
            tokens.pop()
        leng=len(tokens)
        res=''
        cnt=0
        for token in tokens:
            cnt+=1
            if cnt==leng:
                res+=token
            else:
                res+=token+' '
            if cnt%threshold==0 and cnt!=leng:
                res+='\n'
        return res

    def show_retrieve_examples(self,img_retrieval='yes',k=5):
        model_names=['clip','blip_official_itm','metatransformer']
        for model_name in model_names:
            for dataset_name in self.dataset_names:
                if dataset_name=='MS-COCO':
                    embed_name='coco_train'
                else:
                    embed_name=dataset_name.lower()
                cd=ComputeDistance(model_name,embed_name,img_retrieval)
                rand_sub_size=1000
                if img_retrieval=='no':
                    img_embed,cap_embed=cd.load_embeddings()
                else:
                    cap_embed,img_embed=cd.load_embeddings()
                current_id=torch.randint(0,rand_sub_size,(1,)).tolist()
                current_cap,current_img=self.get_random_pair(dataset_name,current_id)
                current_cap=list(map(self.cut_caps,current_cap,[4]))
                cosine=cd.get_distance(cap_embed[current_id,:],img_embed[:rand_sub_size,:])['cosine']
                vals,ids=torch.topk(cosine,k=k,dim=1)
                retrieved_caps,retrieved_imgs=self.get_random_pair(dataset_name,ids.flatten().tolist())
                retrieved_caps=list(map(self.cut_caps,retrieved_caps,[4]*len(retrieved_caps)))
                fig,axes=plt.subplots(1,k+1,figsize=(3*(k+1),3))
                for j in range(k+1):
                    axes[j].set_aspect('equal')
                    if img_retrieval=='yes':
                        if j==0:
                            axes[j].set_facecolor('#EEEEEE')
                            axes[j].text(.5,.5,current_cap[0],ha='center',va='center',fontsize=13)
                            for spine in axes[j].spines.values():
                                spine.set_visible(False)
                        else:
                            axes[j].imshow(retrieved_imgs[j-1].permute(1,2,0))
                            axes[j].set_title(f'Top {j}',fontweight='bold')
                            
                            if torch.equal(current_img[0],retrieved_imgs[j-1]):
                                print('*'*10+'True'+'*'*10)
                                for spine in axes[j].spines.values():
                                    spine.set_color('green')
                                    spine.set_linewidth(5)
                            else:
                                for spine in axes[j].spines.values():
                                    spine.set_visible(False)
                    else:
                        if j==0:
                            #print(current_cap[0])
                            axes[j].imshow(current_img[0].permute(1,2,0))
                            for spine in axes[j].spines.values():
                                    spine.set_visible(False)
                        else:
                            axes[j].set_facecolor('#EEEEEE')
                            axes[j].text(.5,.5,retrieved_caps[j-1],ha='center',va='center',fontsize=13)
                            axes[j].set_title(f'Top {j}',fontweight='bold')
                            if current_cap[0]==retrieved_caps[j-1]:
                                print('*'*10+'True'+'*'*10)
                                for spine in axes[j].spines.values():
                                    spine.set_color('green')
                                    spine.set_linewidth(5)
                            else:
                                for spine in axes[j].spines.values():
                                    spine.set_visible(False)
                    axes[j].set_xticks([])  # Hide x-axis ticks
                    axes[j].set_yticks([])  # Hide y-axis ticks

                margin_config={
                    'left':0.01,
                    'right':.99,
                    'bottom':0.01,
                    'top':.92
                    }
                plt.subplots_adjust(wspace=0, hspace=0,**margin_config)
                plt.savefig(f'./retrieval_img_{img_retrieval}_{model_name}_{dataset_name}.pdf')
                print('DONE.')

    def show_original_data(self):
        plt.subplots(1,3,figsize=(9,3))
        for dataset_name in self.dataset_names:
            dataset=self.load_dataset(dataset_name)


def config():
    parser=argparse.ArgumentParser()
    parser.add_argument('--img_retrieval',type=str,default='yes',help='Specify a dataset')
    args=parser.parse_args()

    return args

if __name__=='__main__':
    seed=10
    torch.manual_seed(seed)
    args=config()
    plot_examples=PlotExamples()
    plot_examples.show_retrieve_examples(args.img_retrieval)
