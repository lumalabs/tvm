# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch 
from torch_utils import persistence 
from training.dit import *   
 
@persistence.persistent_class
class TVMWrapper(torch.nn.Module):

    def __init__(
        self,
        img_resolution,  # Image resolution.
        img_channels,  # Number of color channels.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        mixed_precision=None,    
        model_type="DiT_XL_2_18h",    
        data_type='bf16',
        detach_jvp=False,
        do_scale_param=False,
        **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__()
 

        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.label_dim = label_dim
        if data_type == 'fp16':
            self.dtype = torch.float16
        elif data_type == 'bf16':
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32 
          
        self.model_kwargs = model_kwargs 
        self.model = globals()[model_type](
            img_resolution=img_resolution,
            img_channels=img_channels,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim, 
            **model_kwargs, 
        )
        print('Model # Mparams:', sum(p.numel() for p in self.model.parameters()) / 1000000)
           
 
        self.do_scale_param = do_scale_param
        self.detach_jvp = detach_jvp
   
    def forward_model(
        self, 
        x,
        t,
        s, 
        class_labels=None, 
        beta_scale=None,
        force_fp32=False,  
    ):  
        # beta = 1 / cfg
        with torch.amp.autocast('cuda', enabled=(self.dtype != torch.float32) and not force_fp32, dtype=self.dtype):
            F_x = self.model( 
                x.to(torch.float32) ,
                t.to(torch.float32).flatten() , 
               (t - s).to(torch.float32).flatten() , 
                class_labels=class_labels, 
                guidance_scale=beta_scale.to(torch.float32).flatten() if beta_scale is not None else None,  
            )   
        return F_x.to(t.dtype)
 



    def get_dfds(
        self,  
        x, 
        t,
        s, 
        class_labels=None, 
        beta_scale=None,
        force_fp32=False, 
        **model_kwargs,
    ):  
        def model_wrapper(x_,  t_ , s_):  
            return self.model(x_  ,    t_ ,  (t_ - s_) ,    class_labels,  beta_scale.flatten(), use_sdpa_jvp=True, **model_kwargs) 
            
        with torch.amp.autocast('cuda' , enabled=(self.dtype != torch.float32) and not force_fp32, dtype=self.dtype):
            
            F_ts, F_ts_grad = torch.func.jvp(
                model_wrapper, 
                (x,     t.flatten(),   s.flatten()), 
                (torch.zeros_like(x) ,  torch.zeros_like(s).flatten() , torch.ones_like(s).flatten() )
            ) 
        F_ts = F_ts.to(t.dtype)
        F_ts_grad = F_ts_grad.to(t.dtype)
        if self.detach_jvp:
            F_ts_grad = F_ts_grad.detach() 
 
        out_scale = 1.0
        if self.do_scale_param: 
            w_scale = 1 / beta_scale.to(torch.float32)
            out_scale = w_scale
        f_st = x + (s-t) * F_ts   * out_scale 
        df_ds = out_scale * (F_ts +  (s - t) * F_ts_grad   )   
        return f_st, df_ds 

 
    
    def forward(
        self, 
        xt, 
        xs_prime,
        t,
        s,  
        s_prime,
        class_labels, 
        beta_scale, 
        v_target, 
        teacher_model=None,  
        **model_kwargs,
    ):
        """
        note: safer to wrap jvp inside DDP forward since PyTorch doesn't have good JVP support
        """ 
        class_labels = (
            None
            if self.label_dim == 0
            else (
                torch.zeros([1, self.label_dim], device=xt.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )
        )   
        
        f_ts, df_ds = self.get_dfds(   
            xt.to(torch.float32),
            t.to(torch.float32).reshape(-1, 1, 1, 1),
            s.to(torch.float32).reshape(-1, 1, 1, 1) , 
            class_labels, 
            beta_scale.to(torch.float32).reshape(-1, 1, 1, 1),  
        )   
        
        
        with torch.no_grad():  
            F_ss_from_fts, F_ss_uncond = teacher_model.forward_model( 
                torch.cat([f_ts, xs_prime], dim=0).to(torch.float32),
                torch.cat([s, s_prime], dim=0).to(torch.float32).reshape(-1, 1, 1, 1), 
                torch.cat([s, s_prime], dim=0).to(torch.float32).reshape(-1, 1, 1, 1), 
                torch.cat([class_labels, torch.zeros_like(class_labels)], dim=0) if class_labels is not None else None,
                beta_scale=torch.cat([beta_scale, torch.ones_like(beta_scale)], dim=0).to(torch.float32).reshape(-1, 1, 1, 1), 
            ).chunk(2, dim=0)          
          
        F_ss = self.forward_model( 
            xs_prime.to(torch.float32),
            s_prime.to(torch.float32).reshape(-1, 1, 1, 1),
            s_prime.to(torch.float32).reshape(-1, 1, 1, 1), 
            class_labels,
            beta_scale=beta_scale.to(torch.float32).reshape(-1, 1, 1, 1), 
        )   
           
  
        w_scale = 1 / beta_scale.to(torch.float32) 
        if self.do_scale_param: 
            F_ss_from_fts = w_scale * F_ss_from_fts  
            F_ss = w_scale * F_ss


        terminal_loss =  (((df_ds   - F_ss_from_fts) * beta_scale) ** 2).flatten(1).mean(-1)  

        v_target = w_scale * v_target + (1 - w_scale) * F_ss_uncond
        fm_loss = (((F_ss  - v_target )  * beta_scale ) ** 2).flatten(1).mean(-1)  
          
        if torch.isclose(s, t).any():
            terminal_loss[torch.isclose(s, t).flatten()] = 0 * terminal_loss[torch.isclose(s, t).flatten()]
 

        loss = terminal_loss + fm_loss

        if loss.isnan().any():
            print(f'nan loss {loss.shape} at {t.flatten()}')  
            loss = loss.nan_to_num(nan=0.0)
 
           
        logs = {'ts': t.flatten(), 
                'terminal_loss': terminal_loss.detach(), 
                'fm_loss': fm_loss.detach(), 
                }
        return loss, logs
  
     
