import os
import re
import json 

import pickle 
import functools 
import numpy as np

import torch
import dnnlib  

import torchvision.utils as vutils
import warnings
  
from omegaconf import OmegaConf
from torch_utils import misc 
import hydra

warnings.filterwarnings(
    "ignore", "Grad strides do not match bucket view strides"
)  # False warning printed by PyTorch 1.12.

 
 

# ----------------------------------------------------------------------------
 
 
 
 
@torch.no_grad()
def generator_fn(net, latents,  class_labels=None,   num_steps=None,  cfg_scale=None):
    # Time step discretization. 
    num_steps = 1 if num_steps is None else num_steps
    t_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64, device=latents.device) 

 
    # Sampling steps
    x =  latents.to(torch.float64)   
    beta_scale = (1/ torch.as_tensor(cfg_scale, device=x.device, dtype=torch.float64).view(-1, 1, 1, 1)) if cfg_scale is not None else None
    beta_scale = beta_scale.repeat(x.shape[0], 1, 1, 1)

    out_scale = 1.0
    if beta_scale is not None and net.do_scale_param:    
        out_scale = cfg_scale
            
        
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
          

        F_ts =   net.forward_model(x,  t_cur, t_next,   class_labels=class_labels, beta_scale=beta_scale  ).to(
            torch.float64
        )   * out_scale
        x = x + (t_next - t_cur) * F_ts  
         
    return x
   
 
# ----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs")
def main(cfg):

    device = torch.device("cuda")
    config = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True)) 
  
    # Random seed.
    if config.eval.seed is None:

        seed = torch.randint(1 << 31, size=[], device=device)
        torch.distributed.broadcast(seed, src=0)
        config.eval.seed = int(seed)

    # Checkpoint to evaluate.
    resume_pkl = cfg.eval.resume 
    cudnn_benchmark = config.eval.cudnn_benchmark  
    seed = config.eval.seed
    encoder_kwargs = config.encoder
    
    batch_size = config.eval.batch_size
    sample_kwargs_dict = config.get('sampling', {})
    # Initialize.
    np.random.seed(seed % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
  
    print('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs) 
 
    # Construct network.
    print("Constructing network...") 

    interface_kwargs = dict(
        img_resolution=config.resolution,
        img_channels=config.channels,
        label_dim=config.label_dim,
    )
    if config.get('network', None) is not None:
        network_kwargs = config.network
        net = dnnlib.util.construct_class_by_name(
            **network_kwargs, **interface_kwargs
        )  # subclass of torch.nn.Module
        net.eval().requires_grad_(False).to(device)
  
    # Resume training from previous snapshot.   
    with dnnlib.util.open_url(resume_pkl, verbose=True) as f:
        data = pickle.load(f) 
    
    if config.get('network', None) is not None: 
        misc.copy_params_and_buffers(
            src_module=data['ema'], dst_module=net, require_all=True
        ) 
    else:
        net = data['ema'].eval().requires_grad_(False).to(device)
                
       
    grid_z = torch.randn(
        [batch_size, net.img_channels, net.img_resolution, net.img_resolution],
        device=device,
    )  
    if net.label_dim > 0: 
        labels = torch.randint(0, net.label_dim, (batch_size,), device=device)
        grid_c = torch.nn.functional.one_hot(labels,  num_classes=net.label_dim)
    else:
        grid_c = None
 
    # Few-step Evaluation. 
    generator_fn_dict = {k: functools.partial(generator_fn, **sample_kwargs) for k, sample_kwargs in sample_kwargs_dict.items()}
    print("Sample images...") 
    res = {}
    for key, gen_fn in generator_fn_dict.items():
        images = gen_fn(net, grid_z, grid_c)   
        images = encoder.decode(images.to(device) ).detach().cpu() 
        
        vutils.save_image(
            images / 255.,
            os.path.join(f"{key}_samples.png"),
            nrow=int(np.sqrt(images.shape[0])),
            normalize=False,
        )
        
        res[key] = images  
    
    print('done.')

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
