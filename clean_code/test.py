import torch
import argparse
import torch.nn.functional as F

from tqdm import tqdm
from torchmetrics.classification import MulticlassPrecision

from NN import MLP
from train import Train
from feature_extractor import FeatureExtraction

class TestModel():
    def __init__(self,
            model_name,
            dataset,
            in_size,
            epoch_checkpoint,
            loss_func='mse',
            batch_size=256,
            img_retrieval='yes',
            gpu_id=0):
        self.device=torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.model_name=model_name
        self.train=Train(self.model_name,dataset,in_size,img_retrieval=img_retrieval,loss_func=loss_func)
        self.dataloader=self.train.test_loader
        self.loss_func=loss_func
        self.img_retrieval=img_retrieval
        self.model=MLP(in_size=in_size)
        self.model_dict_path=f'/data/data_fxu/alignment_features_journal_paper/{self.model_name.upper()}/model_checkpoints/{self.model_name}_{dataset}_{self.loss_func}_epoch{epoch_checkpoint}_{img_retrieval}.pt'
        self.model.load_state_dict(torch.load(self.model_dict_path))
        self.model.eval()
        self.model.to(self.device)
    
    @torch.no_grad
    def test(self):
        topks=[1,5,10]
        for k in topks:
            precision=.0
            for i,batch in enumerate(tqdm(self.dataloader)):
                images=batch['image']
                texts=batch['text']

                img_feats,txt_feats=FeatureExtraction(self.model_name,self.device).get_features(images,texts)
                img_feats=F.normalize(img_feats)
                txt_feats=F.normalize(txt_feats)

                if self.img_retrieval=='yes':
                    img_feats,txt_feats=txt_feats,img_feats
                
                img_feat_repeat=img_feats.repeat_interleave(txt_feats.size(0),dim=0)
                txt_feat_repeat=txt_feats.repeat(img_feats.size(0),1)
                inputs=torch.cat((img_feat_repeat,txt_feat_repeat),dim=1).to(self.device)
                preds=self.model(inputs).reshape(img_feats.size(0),txt_feats.size(0))

                compute_precision=MulticlassPrecision(num_classes=preds.size(1),top_k=k).to(self.device)
                tars=torch.arange(preds.size(1)).to(self.device)
                precision+=compute_precision(preds,tars).item()
            print(f'top{k} precision: {precision/(i+1)*100:.2f}%')

def set_config():
    parser=argparse.ArgumentParser()
    parser.add_argument('model_name',type=str,help='Model name')
    parser.add_argument('dataset',type=str,help='specify a feature dataset')
    parser.add_argument('in_size',type=int)
    parser.add_argument('epoch_checkpoint',type=int,help='model epoch checkpoint')
    parser.add_argument('loss_func',type=str,help='loss function')
    parser.add_argument('img_retrieval',type=str,default='no')
    #parser.add_argument('hidden_sizes',type=int,nargs='+',help='number of hidden nuerons')
    args=parser.parse_args()
    
    return args

if __name__=='__main__':
    torch.manual_seed(0)

    args=set_config()
    
    testmodel=TestModel(
            args.model_name,
            args.dataset,
            args.in_size,
            args.epoch_checkpoint,
            loss_func=args.loss_func,
            img_retrieval=args.img_retrieval,
            )
    testmodel.test()
