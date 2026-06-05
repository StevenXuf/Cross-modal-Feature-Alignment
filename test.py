import torch
import argparse
from NN import MLP
from tqdm import tqdm
from train import Train

class TestModel():
    def __init__(self,
            model_name,
            dataset,
            loss_func,
            img_retrieval,
            epoch_checkpoint,
            hidden_sizes,
            gpu_id=0):
        self.device=torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.train=Train(model_name=model_name,dataset=dataset,img_retrieval=img_retrieval,loss_func=loss_func)
        self.caption_embed_test=self.train.caption_embed_test.cpu()
        self.img_embed_test=self.train.img_embed_test.cpu()
        self.loss_func=loss_func
        print(self.img_embed_test.size())
        self.model=MLP(in_size=self.caption_embed_test.size(1)+self.img_embed_test.size(1),hidden_sizes=hidden_sizes)
        self.model_dict_path=f'/data/data_fxu/alignment_features/{model_name.upper()}/model_checkpoints/mlp_{model_name}_{dataset}_{self.loss_func}_epoch{epoch_checkpoint}_{img_retrieval}.pt'
        self.model.load_state_dict(torch.load(self.model_dict_path))
        self.model.eval()
        self.model.to(self.device)
    
    def cosine(self):
        sim=self.train._pairwise_cosine_distance(self.caption_embed_test,self.img_embed_test,self.device)
        return sim
    
    @torch.no_grad
    def test(self):
        topks=[1,5,10]
        for k in topks:
            cnt=0
            for j in tqdm(range(self.caption_embed_test.size(0))):
                data=torch.cat((self.caption_embed_test[j,:]+torch.zeros(self.img_embed_test.size(0),self.caption_embed_test.size(1)),self.img_embed_test),dim=1)
                preds=self.model(data.to(self.device)).flatten()
                val,ids=torch.topk(preds,k=k)
                if j in ids.flatten().tolist():
                    cnt+=1
            acc=round(cnt/self.caption_embed_test.size(0)*100,2)
            print(f'top{k}_test_acc: {acc}')

def set_config():
    parser=argparse.ArgumentParser()
    parser.add_argument('model_name',type=str,help='Model name')
    parser.add_argument('dataset',type=str,help='specify a feature dataset')
    parser.add_argument('loss_func',type=str,help='loss function')
    parser.add_argument('img_retrieval',type=str,default='no')
    parser.add_argument('epoch_checkpoint',type=int,help='model epoch checkpoint')
    parser.add_argument('hidden_sizes',type=int,nargs='+',help='number of hidden nuerons')
    args=parser.parse_args()

    return args

if __name__=='__main__':
    torch.manual_seed(0)

    args=set_config()
    
    print(args.hidden_sizes)
    testmodel=TestModel(
            args.model_name,
            args.dataset,
            args.loss_func,
            args.img_retrieval,
            args.epoch_checkpoint,
            args.hidden_sizes
            )
    testmodel.test()
