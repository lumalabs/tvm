import math

import torch
import torch.nn as nn
from torch_utils import persistence 

import numpy as np 
 

@persistence.persistent_class
class TVMLoss:
    def __init__(
        self,  
        s_P_mean=-0.4,
        s_P_std=1.0, 
        gap_P_mean=-0.8,
        gap_P_std=1.0,
        cfg=None,
        max_cfg=None,
        min_cfg=None, 
        label_dropout=0.1, 
        **kwargs,
    ):  
        super().__init__()
        self.s_P_mean = s_P_mean
        self.s_P_std = s_P_std
        self.gap_P_mean = gap_P_mean
        self.gap_P_std = gap_P_std 
    
        self.cfg = cfg
        self.max_cfg = max_cfg
        self.min_cfg = min_cfg
        if max_cfg is not None:
            assert min_cfg is not None 

        self.label_dropout = label_dropout 
        

    def sample_ts(self,  images,): 
 

        log_len = torch.randn([images.shape[0], 1, 1, 1], device=images.device) * self.gap_P_std + self.gap_P_mean
        len_ = log_len.sigmoid() 
        maxs = 1 - len_
        max_logns = torch.log(maxs/(1-maxs))

        dist = torch.distributions.Normal(
            loc=torch.full_like(len_, self.s_P_mean),
            scale=torch.full_like(len_, self.s_P_std),
        )

        cdf = torch.rand_like(len_) * (
            dist.cdf(max_logns) - dist.cdf(torch.full_like(len_, -np.inf))
        ) + dist.cdf(torch.full_like(len_, -np.inf))
        log_ns = dist.icdf(cdf)  
        s = log_ns.sigmoid() 

        t = (s + len_).clamp(max=1)

        
        log_ns_prime = torch.randn_like(len_) * self.s_P_std + self.s_P_mean
        s_prime = log_ns_prime.sigmoid() 

        return t, s, s_prime


    def __call__(self, net, images, labels=None, teacher_model=None, **kwargs): 
        
        # Augmentation if needed 
        t,s,s_prime = self.sample_ts(images)
          
        if self.cfg is not None:
            beta_scale =  torch.ones_like(t)  / self.cfg 
        elif self.max_cfg is not None and self.min_cfg is not None: 
            beta_scale = 1 / (torch.rand_like(t) * (self.max_cfg - self.min_cfg) + self.min_cfg)
        else:
            beta_scale = torch.rand_like(t)  
         
        noise = torch.randn_like(images)
        yt = (1 - t) * images + t * noise  
        ys_prime = (1 - s_prime) * images + s_prime * noise  
        
        if labels is not None and self.label_dropout is not None and self.label_dropout > 0:
            uncond_sel = torch.rand(labels.shape[0], device=images.device) < self.label_dropout
            labels[uncond_sel] = 0 
            beta_scale[uncond_sel] = 1.0 
         
        loss, logs = net(  
            yt, 
            ys_prime, 
            t,
            s,  
            s_prime,
            labels,  
            beta_scale,
            (noise - images),
            teacher_model=teacher_model, 
        )  
         
        return loss, logs
 