import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,
            in_size=1536,
            hidden_sizes=[500,300,100],
            out_size=1,
            init='kaiming'):

        super(MLP,self).__init__()
        self.in_size=in_size
        self.hidden_sizes=hidden_sizes
        self.out_size=out_size
        self.n_hidden_layers=len(hidden_sizes)
        self.init=init

        self.layers=[]
        
        for i in range(self.n_hidden_layers+1):
            name='linear'+str(i)
            if i<self.n_hidden_layers:
                self.__setattr__(name,nn.Sequential(
                    nn.Linear(self.in_size,self.hidden_sizes[i]),
                    nn.ReLU()))
                self.in_size=self.hidden_sizes[i]
            else:
                self.in_size=self.hidden_sizes[-1]
                self.__setattr__(name,nn.Sequential(
                    nn.Linear(self.in_size,self.out_size),
                    nn.Tanh()))

            self.layers.append(self.__getattr__(name))

        if self.init=='xavier':
            for module in self.modules():
                if isinstance(module,nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
        elif self.init=='kaiming':
            for module in self.modules():
                if isinstance(module,nn.Linear):
                    nn.init.kaiming_uniform_(module.weight)

    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return x

class BranchMLP(nn.Module):
    def __init__(self,
            branch_in_size=768,
            hidden_size=256,
            out_size=1,
            branch_layers=2,
            init='kaiming'):

        super(BranchMLP,self).__init__()
        self.branch_in_size=branch_in_size
        self.hidden_size=hidden_size
        self.out_size=out_size
        self.branch_layers=branch_layers
        self.init=init

        self.layers=[]
        
        for j in range(2):
            for i in range(self.branch_layers):
                name=f'branch{j}_linear{i}'
                if i==0:
                    self.__setattr__(name,nn.Sequential(
                        nn.Linear(self.branch_in_size,self.hidden_size),
                        nn.ReLU()))
                else:
                    self.__setattr__(name,nn.Sequential(
                        nn.Linear(self.hidden_size,self.hidden_size),
                        nn.ReLU()))
                self.layers.append(self.__getattr__(name))

        self.concate_layer=nn.Sequential(
                nn.Linear(self.hidden_size*2,self.out_size),
                nn.Tanh())
        
        if self.init=='xavier':
            for module in self.modules():
                if isinstance(module,nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
        elif self.init=='kaiming':
            for module in self.modules():
                if isinstance(module,nn.Linear):
                    nn.init.kaiming_uniform_(module.weight)

    def forward(self,x,y):
        for i in range(self.branch_layers):
            x=self.layers[i](x)
            y=self.layers[2+i](y)
        out=self.concate_layer(torch.cat((x,y),dim=1))
        return out

if __name__=='__main__':
    mlp=MLP(hidden_sizes=[100,200,300,400])
    print(mlp)
    #print(sum([p.numel() for p in mlp.parameters()]))

'''
    branch_mlp=BranchMLP()
    print(branch_mlp)
    print(sum([p.numel() for p in branch_mlp.parameters()]))
'''
