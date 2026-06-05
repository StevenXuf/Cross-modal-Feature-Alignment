import torch
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP

import sys
sys.path.append('/home/users/u101139/alignment')
from load_data import LoadIMDB
from NN import MLP
from sampler import DistributedEvalSampler

dist.init_process_group('nccl')
rank=dist.get_rank()
world_size=dist.get_world_size()
device=rank%torch.cuda.device_count()

dataset=LoadIMDB().get_dataset()
sampler=DistributedSampler(dataset)
batch_size=16
dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=False,sampler=sampler)
print(f'batch size: {batch_size}')

model=MLP().to(device)
model=DDP(model,device_ids=[device])

print(f'length of dataset loader: {len(dataloader)}')
for i,batch in enumerate(dataloader):
    if i==1:
        print(f'current batch index: {i+1}')
        print(f"minibatch size on machine {rank}: {len(batch['text'])}")
        print(batch)
        print(batch['text'])
        print(batch['img_id'])
    dist.barrier()
