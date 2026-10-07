"""Small runnable visual BC baseline and data-to-gradient integration check.

This checks training usability; one episode cannot establish generalization.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

class VisualBC(nn.Module):
    def __init__(self,state_dim=93,action_dim=172):
        super().__init__()
        self.vision=nn.Sequential(nn.Conv2d(6,16,5,2),nn.ReLU(),nn.Conv2d(16,32,3,2),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(32+state_dim,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,action_dim))
    def forward(self,head,wrist,state):
        x=torch.cat([head,wrist],dim=1);x=nn.functional.interpolate(x,size=(96,128),mode='bilinear',align_corners=False)
        return self.head(torch.cat([self.vision(x),state],1))

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=30);a=p.parse_args()
    torch.manual_seed(17);torch.set_num_threads(4)
    ds=LeRobotDataset('local/g1_window_pickup',root=a.dataset,video_backend='pyav')
    # Parquet-only extraction avoids decoding all videos to compute normalization.
    states=np.stack(ds.hf_dataset['observation.state']);actions=np.stack(ds.hf_dataset['action'])
    sm=torch.from_numpy(states.mean(0)).float();ss=torch.from_numpy(np.maximum(states.std(0),.03)).float();am=torch.from_numpy(actions.mean(0)).float();ast=torch.from_numpy(np.maximum(actions.std(0),.03)).float()
    loader=DataLoader(ds,batch_size=min(8,len(ds)),shuffle=True,num_workers=0)
    model=VisualBC(states.shape[1],actions.shape[1]);opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    losses=[];iterator=iter(loader);initial=[p.detach().clone() for p in model.parameters()]
    for step in range(a.steps):
        try:batch=next(iterator)
        except StopIteration:iterator=iter(loader);batch=next(iterator)
        target=(batch['action']-am)/ast
        pred=model(batch['observation.images.head'],batch['observation.images.right_wrist'],(batch['observation.state']-sm)/ss)
        loss=nn.functional.mse_loss(pred,target);assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());opt.step();losses.append(float(loss.detach()))
    changed=any(not torch.equal(old,new) for old,new in zip(initial,model.parameters()));assert changed
    a.output.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.state_dict(),'state_mean':sm,'state_std':ss,'action_mean':am,'action_std':ast,'state_dim':states.shape[1],'action_dim':actions.shape[1],'dataset':str(a.dataset),'execution':'reshape model output(172,) to(2,86); each row 43 absolute position targets + 43 stiffness; execute at50Hz with dataset controller_config.json'},a.output/'visual_bc_smoke.pt')
    report={'official_lerobot_loader':True,'frames':len(ds),'steps':a.steps,'losses':losses,'all_losses_gradients_finite':True,'parameters_updated':changed,'model':'small dual-RGB + proprioceptive BC baseline','held_out_evaluation':False,'claim':'training pipeline integration only; not a validated robot policy'}
    (a.output/'training_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
