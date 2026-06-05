"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""

import os
import argparse
import optuna

from optuna.trial import TrialState
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from feature_processing import process_features,compute_loss

def set_config():
    parser=argparse.ArgumentParser()
    parser.add_argument('model_name',type=str,help='Multimodal name')
    parser.add_argument('dataset',type=str,help='imdb, flickr30k, or coco_train')
    parser.add_argument('img_retrieval',type=str,help='yes or no')
    parser.add_argument('loss_func',type=str,help='mse or contrastive')
    parser.add_argument('--n_train_batch_sub',default=10,type=int,help='number of selected train batches')
    parser.add_argument('--n_val_batch_sub',default=5,type=int,help='number of selected val batches')
    parser.add_argument('--batch_size',default=64,type=int,help='batch size')
    parser.add_argument('--n_epochs',default=10,type=int,help='number of epochs')
    parser.add_argument('--device',default='0',type=str,help='device id')
    parser.add_argument('--n_trials',default=10,type=int,help='number of trials')
    args=parser.parse_args()
    return args

#python3 -u hyperparam_tuning.py resnetbert imdb yes mse --device=0


def define_model(trial,in_features):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 5)
    layers = []

    #in_features=cap_train.size(1)+img_train.size(1)
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 100,1100,step=200)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        in_features = out_features
    layers.append(nn.Linear(in_features,1))
    layers.append(nn.Tanh())
    
    for module in layers:
        if isinstance(module,nn.Linear):
            nn.init.kaiming_uniform_(module.weight)

    return nn.Sequential(*layers)


def objective(trial,
    model_name='clip',
    dataset='imdb',
    img_retrieval='yes',
    loss_func='mse',
    n_train_batch_sub=10,
    n_val_batch_sub=5,
    batch_size=64,
    n_epochs=10,
    device='cpu',
    patience=5,
    tolerance=.01
): 
    cap_train,cap_test,cap_val,img_train,img_test,img_val=process_features(model_name,dataset,img_retrieval)
    in_size=cap_train.size(1)+img_train.size(1)

    model = define_model(trial,in_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch: 1 / (2**(.5*epoch) + 1))
    #scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',patience=2,threshold=1e-8)
    
    #assert cap_train.size()==img_train.size()

    n_train_batches=cap_train.size(0)//batch_size if cap_train.size(0)%batch_size==0 else cap_train.size(0)//batch_size+1
    batch_idx_train=torch.randperm(n_train_batches)[:n_train_batch_sub]
    
    n_val_batches=cap_val.size(0)//batch_size if cap_val.size(0)%batch_size==0 else cap_val.size(0)//batch_size+1
    batch_idx_val=torch.randperm(n_val_batches)[:n_val_batch_sub]
    
    last_loss=torch.inf
    trigger_count=0

    # Training of the model.
    for epoch in tqdm(range(n_epochs)):
        running_loss=0.0
        model.train()
        for i in range(len(batch_idx_train)):
            optimizer.zero_grad()
            loss=compute_loss(batch_idx_train[i],batch_size,cap_train,img_train,device,loss_func,model)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        epoch_loss=running_loss/len(batch_idx_train)

        # Validation of the model.

        model.eval()
        val_running_loss=0.0
        with torch.no_grad():
            for j in range(len(batch_idx_val)):
                    val_loss=compute_loss(batch_idx_val[j],batch_size,cap_val,img_val,device,loss_func,model)
                    val_running_loss+=val_loss.item()

            val_epoch_loss=val_running_loss/len(batch_idx_val)

        scheduler.step()
        current_lr = scheduler.get_last_lr()
        trial.report(val_epoch_loss,epoch)
        
        print(f'Epoch: {epoch+1}, lr: {current_lr[0]}, training loss: {epoch_loss}, validation loss: {val_epoch_loss}.')
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        if (last_loss-val_epoch_loss)/last_loss<tolerance:
            trigger_count+=1
            print(f'Trigger count: {trigger_count}')
            if trigger_count>=patience:
                print('*'*5+f'Early stopped at epoch {epoch+1}'+'*'*5)
                break
        else:
            trigger_count=0

        last_loss=val_epoch_loss

    return val_epoch_loss


    
def run_trial():
    args=set_config()
    print(args)
    device=torch.device('cpu') if args.device=='cpu' else torch.device(f'cuda:{args.device}')

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial:objective(trial,model_name=args.model_name,
    dataset=args.dataset,
    img_retrieval=args.img_retrieval,
    loss_func=args.loss_func,
    n_train_batch_sub=args.n_train_batch_sub,
    n_val_batch_sub=args.n_val_batch_sub,
    batch_size=args.batch_size,
    n_epochs=args.n_epochs,device=device),
    n_trials=args.n_trials) #timeout=1000

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    
    hidden_sizes=[]
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    return study.best_params


if __name__ == "__main__":
    torch.manual_seed(0)
    best_params=run_trial()
