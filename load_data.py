import torch
import json
import os
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import webdataset as wds

from torchvision import transforms

class LoadCOCO():
    def __init__(self,path='/data/data_fxu/MS-COCO/raw'):
        self.train_path=path+'/captions_train2017.json'
        self.val_path=path+'/captions_val2017.json'
        self.cap_train=json.load(open(self.train_path))
        self.cap_val=json.load(open(self.val_path))

    def get_imgID_captions_train(self,store_path='/data/data_fxu/MS-COCO'):
        imgID_cap_train={}
        for i in range(len(self.cap_train['annotations'])):
            img_id=self.cap_train['annotations'][i]['image_id']
            img_id=str(img_id)
            cap=self.cap_train['annotations'][i]['caption']
            if img_id in imgID_cap_train.keys():
                imgID_cap_train[img_id].append(cap)
            else:
                imgID_cap_train[img_id]=[cap]
        torch.save(imgID_cap_train,store_path+f'/imgIDs_and_captions_train.json')
        return imgID_cap_train

    def get_imgID_captions_val(self,store_path='/data/data_fxu/MS-COCO'):
        imgID_cap_val={}
        for i in range(len(self.cap_val['annotations'])):
            img_id=self.cap_val['annotations'][i]['image_id']
            img_id=str(img_id)
            cap=self.cap_val['annotations'][i]['caption']
            if img_id in imgID_cap_val.keys():
                imgID_cap_val[img_id].append(cap)
            else:
                imgID_cap_val[img_id]=[cap]
        torch.save(imgID_cap_val,store_path+f'/imgIDs_and_captions_val.json')
        return imgID_cap_val
    
    def get_dataset(self,cap_idx=0):
        train_set=self.read_imgID_captions_train()
        val_set=self.read_imgID_captions_val()
        train_img_ids=list(map(lambda x: '0'*(12-len(x))+x,train_set.keys()))
        train_caps=[]

        val_img_ids=list(map(lambda x: '0'*(12-len(x))+x,val_set.keys()))
        val_caps=[]

        for i in range(len(train_img_ids)):
            train_caps.append(train_set[list(train_set.keys())[i]][cap_idx])
        for j in range(len(val_img_ids)):
            val_caps.append(val_set[list(val_set.keys())[j]][cap_idx])
        
        return CustomDataset(train_img_ids,train_caps),CustomDataset(val_img_ids,val_caps)

    def read_imgID_captions_train(self,path='/data/data_fxu/MS-COCO/imgIDs_and_captions_train.json'):
        if os.path.exists(path)==False:
            imgID_cap_train=self.get_imgID_captions_train()
            return imgID_cap_train
        else:
            return torch.load(path,weights_only=True)
    
    def read_imgID_captions_val(self,path='/data/data_fxu/MS-COCO/imgIDs_and_captions_val.json'):
        if os.path.exists(path)==False:
            imgID_cap_val=self.get_imgID_captions_val()
            return imgID_cap_val
        else:
            return torch.load(path,weights_only=True)


class LoadFlickr30K():
    def __init__(self,path='/data/data_fxu/Flickr30k'):
        self.path=path
        self.df=pd.read_csv(self.path+'/results.csv')

    def get_names_caps(self,cap_idx=0):
        img_names_caps={}
        info=self.df[self.df['comment_number']==cap_idx]
        img_names_caps['img_names']=info['image_name'].tolist()
        img_names_caps['img_caps']=info['comment'].tolist()

        return img_names_caps

    def get_dataset(self,idx=0):
        img_names_caps=self.get_names_caps(cap_idx=idx)
        return CustomDataset(img_names_caps['img_names'],img_names_caps['img_caps'])


class LoadCelebA():
    def __init__(self,data_folder='/data/data_fxu/CelebA'):
        self.data_folder=data_folder
        self.attr_path=self.data_folder+'/list_attr_celeba.txt'
        self.celeb_attr=pd.read_csv(self.attr_path,header=[0],sep=r'\s+')
        

    def get_imgID_texts(self):
        n_imgs=self.celeb_attr.shape[0]
        self.celeb_attr['Concatenated_Attr']=None
        for i in range(n_imgs):
            self.celeb_attr.at[i,'Concatenated_Attr']=','.join([' '.join(colname.split('_')) for colname in (self.celeb_attr.iloc[i].values[1:]==1)*self.celeb_attr.columns.values[1:] if len(colname)>0])
        
        return {'img_ids':self.celeb_attr['img_id'].tolist(),'texts':self.celeb_attr['Concatenated_Attr'].tolist()}

    def get_dataset(self):
        res=self.get_imgID_texts()

        return CustomDataset(res['img_ids'],res['texts'])


class LoadIMDB():
    def __init__(self,data_folder='/data/data_fxu/IMDB'):
        self.data_folder=data_folder
        self.movies=pd.read_csv(self.data_folder+'/movies.csv')

    def get_posterpath_titles(self):
        valid_poster_paths=[]
        valid_titles=[]
        existing_imgs=os.listdir(self.data_folder+'/images')
        for img in existing_imgs:
            local_path=f'images/{img}'
            if local_path in self.movies['local_image_path'].tolist():
                valid_poster_paths.append(img)
                valid_titles.append(self.movies[self.movies['local_image_path']==local_path]['Title'].values[0])

        return {'poster_paths':valid_poster_paths,
                'titles':valid_titles}

    def get_dataset(self):
        res=self.get_posterpath_titles()
        return CustomDataset(res['poster_paths'],res['titles'])


class CustomDataset(Dataset):
    def __init__(self,img_ids,caps):
        self.img_ids=img_ids
        self.caps=caps

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self,idx):
        img_id=self.img_ids[idx]
        cap=self.caps[idx]
        sample={'img_id':img_id,'text':cap}
        
        return sample

class LoadWebDataset():
    def __init__(self):
        self.transform=transforms.Compose([
                transforms.Resize((224,224),antialias=True),
                transforms.ToTensor(),
                lambda img: torch.cat((img,img,img)) if img.size(0)==1 else img,
                lambda img: img/255
])
        self.identity=lambda x: x
    
    def get_dataloaders(self,batch_size=512,webdataset_path='/SBU/sbucaptions'):
        path=f'/data/data_fxu{webdataset_path}'
        tarfiles=[]
        for file in os.listdir(path):
            if file.endswith('.tar'):
                tarfiles.append(file)
        
        dataloaders=[]
        for tarfile in tarfiles:
            dataset=wds\
                .WebDataset(path+f'/{tarfile}')\
                .decode('pil')\
                .to_tuple('jpg;png','txt')\
                .map_tuple(self.transform,self.identity)

            dataloaders.append(DataLoader(dataset,batch_size=batch_size))
        
        return dataloaders

if __name__=='__main__':
    batch_size=1024

    load_coco=LoadCOCO()
    coco_train_set,coco_val_set=load_coco.get_dataset()

    load_flickr30k=LoadFlickr30K()
    flickr_set=load_flickr30k.get_dataset()
    
    #load_celebA=LoadCelebA()
    #celebA_set=load_celebA.get_dataset()
    
    load_imdb=LoadIMDB()
    imdb_set=load_imdb.get_dataset()
    print(type(imdb_set))
    #load_webdata=LoadWebDataset()
    #sbu_dataloader=load_webdata.get_dataloaders(webdataset_path='/CC3M/cc3m')
    #print(len(sbu_dataloader))
        
    for dataset in [coco_train_set,coco_val_set,imdb_set,flickr_set]:
        print(len(dataset))
        loader=DataLoader(
            dataset,
            batch_size=batch_size
            )

    '''
    for i,batch in enumerate(celebA_loader):
        if i==1:
            print(batch['text'])
    '''

