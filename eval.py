import os
import re
import json 

import pickle 
import functools
import PIL.Image
import numpy as np

import torch
import dnnlib
from torch_utils import distributed as dist 
from torch_utils import misc

from metrics import metric_main

import warnings

import wandb
from einops import rearrange

from omegaconf import OmegaConf
import hydra

from generate_images import generator_fn

warnings.filterwarnings(
    "ignore", "Grad strides do not match bucket view strides"
)  # False warning printed by PyTorch 1.12.


@hydra.main(version_base=None, config_path="configs")
def main(cfg):

    config = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True))
    torch.multiprocessing.set_start_method("spawn")
    dist.init()

    config.dataset.xflip = False
    # Validate dataset options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**config.dataset)
        dataset_name = dataset_obj.name
        config.dataset.resolution = (
            dataset_obj.resolution
        )  # be explicit about dataset resolution
        config.dataset.max_size = len(dataset_obj)  # be explicit about dataset size

        del dataset_obj  # conserve memory
    except IOError as err:
        raise ValueError(f"data: {err}")

    # Random seed.
    if config.eval.seed is None:

        seed = torch.randint(1 << 31, size=[], device=torch.device("cuda"))
        torch.distributed.broadcast(seed, src=0)
        config.eval.seed = int(seed)

    # Checkpoint to evaluate.
    resume_pkl = cfg.eval.resume

    # Description string. 
    desc = f"-{config.name}"

    outdir = os.path.join("outputs", config.logger.project)

    # Pick output directory.
    if dist.get_rank() != 0:
        run_dir = None

    else:
        prev_run_dirs = []
        if os.path.isdir(outdir):
            prev_run_dirs = [
                x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))
            ]
        prev_run_ids = [re.match(r"^\d+", x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        run_dir = os.path.join(outdir, f"{cur_run_id:05d}-{desc}")
        assert not os.path.exists(run_dir)

    # Print options.
    dist.print0()
    dist.print0("Training options:")
    dist.print0(json.dumps(OmegaConf.to_container(config), indent=2))
    dist.print0()
    dist.print0(f"Output directory:        {run_dir}")
    dist.print0(f"Dataset path:            {config.dataset.path}")
    dist.print0(f"Class-conditional:       {config.dataset.use_labels}")
    dist.print0(f"Number of GPUs:          {dist.get_world_size()}")  
    dist.print0()

    # Create output directory.
    dist.print0("Creating output directory...")
    if dist.get_rank() == 0:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "training_options.json"), "wt") as f:
            json.dump(OmegaConf.to_container(config), f, indent=2)
        dnnlib.util.Logger(
            file_name=os.path.join(run_dir, "log.txt"),
            file_mode="a",
            should_flush=True,
        )

    # Eval.
    evaluation(config, resume_pkl, run_dir)


# ----------------------------------------------------------------------------


def setup_snapshot_image_grid(training_set, random_seed=0, gw=None, gh=None):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 16) if gw is None else gw
    gh = np.clip(4320 // training_set.image_shape[1], 4, 16) if gh is None else gh

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict()  # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [
                indices[(i + gw) % len(indices)] for i in range(len(indices))
            ]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)


# ----------------------------------------------------------------------------


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], "L").save(fname)
    if C == 3:
        PIL.Image.fromarray(img, "RGB").save(fname)

 
 
# ----------------------------------------------------------------------------


def evaluation(
    config,
    resume_pkl,
    run_dir=".",  # Output directory.
    device=torch.device("cuda"),
):
    batch_size = config.eval.batch_size
    batch_gpu = config.eval.batch_gpu
    cudnn_benchmark = config.eval.cudnn_benchmark 
    metrics = config.eval.metrics
    seed = config.eval.seed

    dataset_kwargs = config.dataset 
    encoder_kwargs = config.encoder
    
    sample_kwargs_dict = config.get('sampling', {})
    # Initialize.
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu = batch_size // dist.get_world_size()

    # Load dataset.
    dist.print0("Loading dataset...")
    # dataset_kwargs.use_labels = True
    dataset_obj = dnnlib.util.construct_class_by_name(
        **dataset_kwargs
    )  # subclass of training.dataset.Dataset
 
    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs) 
 
    # Construct network.
    dist.print0("Constructing network...")
    interface_kwargs = dict(
        img_resolution=dataset_obj.resolution,
        img_channels=dataset_obj.num_channels//2 if dataset_obj.num_channels > 3 else dataset_obj.num_channels,
        label_dim=dataset_obj.label_dim,
    )
    if config.get('network', None) is not None:
        network_kwargs = config.network
        net = dnnlib.util.construct_class_by_name(
            **network_kwargs, **interface_kwargs
        )  # subclass of torch.nn.Module
        net.eval().requires_grad_(False).to(device)
  
    if dist.get_rank() == 0:
        if config.logger.get("api_key", None):
            wandb.login(key=config.logger.pop("api_key"))
        wandb.init(
            **config.logger,
            name=config.name,
            config=OmegaConf.to_container(config),
        ) 

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
 
    
    # Few-step Evaluation. 
    generator_fn_dict = {k: functools.partial(generator_fn, **sample_kwargs) for k, sample_kwargs in sample_kwargs_dict.items()}
         

    dist.print0("Evaluating few-step generation...")
    resume_pkl_list = [resume_pkl] if resume_pkl.endswith(".pkl") else sorted([k for k in dnnlib.util.glob.glob(os.path.join(resume_pkl, "*.pkl")) if os.path.basename(k).split("-")[-1].split(".")[0] != 'latest'],
                                                                              key=lambda x: int(os.path.basename(x).split("-")[-1].split(".")[0]))
    
 
    for metric in metrics:
        for i, resume_pkl_ in enumerate(resume_pkl_list):
            
            # Resume training from previous snapshot. 
            dist.print0(f'Loading network weights from "{resume_pkl_}"...')
            if dist.get_rank() != 0:
                torch.distributed.barrier()  # rank 0 goes first
            with dnnlib.util.open_url(resume_pkl_, verbose=(dist.get_rank() == 0)) as f:
                data = pickle.load(f)
            if dist.get_rank() == 0:
                torch.distributed.barrier()  # other ranks follow 
                
            if config.get('network', None) is not None:   
                misc.copy_params_and_buffers(
                    src_module=(data['net']._orig_mod if hasattr(data['net'], '_orig_mod') else data['net']), dst_module=net, require_all=True
                ) 
            else:
                net = data['net'].eval().requires_grad_(False).to(device)
                
             
            del data  # conserve memory
             
            if i == len(resume_pkl_list) - 1 and  dist.get_rank() == 0: 
                grid_size, images, labels = setup_snapshot_image_grid(training_set=dataset_obj )  
                
                images = encoder.decode(encoder.encode(  torch.as_tensor(images, device=device) )  ).detach().cpu().numpy() 
                save_image_grid(
                    images,
                    os.path.join(run_dir, "data.png"),
                    drange=[0, 255],
                    grid_size=grid_size,
                ) 
                wandb.log(
                    {
                        "groundtruth": [
                            wandb.Image(
                                rearrange(
                                    images[i],
                                    "c h w -> h w c",
                                ),
                            )
                            for i in range(min(64, images.shape[0]))
                        ],
                    },
                )
                 
                grid_z = torch.randn(
                    [labels.shape[0], net.img_channels, net.img_resolution, net.img_resolution],
                    device=device,
                ) 
                grid_z = grid_z.split(batch_gpu)

                grid_c = torch.nn.functional.one_hot(torch.randint(dataset_obj.label_dim, (labels.shape[0],), device=device) , 
                                                    num_classes=dataset_obj.label_dim) if dataset_obj.has_labels else torch.as_tensor(labels, device=device)
                     
                grid_c = grid_c.split(batch_gpu) 
                dist.print0("Exporting final sample images...") 
                res = {}
                for key, gen_fn in generator_fn_dict.items():
                    images = [gen_fn(net, z, c)  for z, c in zip(grid_z, grid_c)]
                    images = torch.cat(images)
                    images = encoder.decode(images.to(device) ).detach().cpu().numpy()
                    save_image_grid(
                        images,
                        os.path.join(run_dir, f"{key}_samples.png"), 
                        drange=[0, 255],
                        grid_size=grid_size,
                    )
                     
                    res[key] = images 
                wandb.log(
                    {
                        f"{key}_sample": [
                            wandb.Image(
                                rearrange( dec_sample,
                                    "c h w -> h w c",
                                ),
                                caption=f"max: {dec_sample.max():.2f}, min: {dec_sample.min():.2f}, std: {dec_sample.std():.2f}, mean: {dec_sample.mean():.2f}",
                            )
                                for dec_sample in  images
                        ] for key, images in res.items()
                    },
                )   
                
            
            name = os.path.basename(resume_pkl_).replace(".pkl", "")
        
            for key, gen_fn in generator_fn_dict.items():
                result_dict = metric_main.calc_metric(
                    metric=metric,
                    generator_fn=gen_fn,
                    decode_fn=encoder.decode,
                    G=net,
                    G_kwargs={},
                    dataset_kwargs=dataset_kwargs,
                    detector_kwargs=config.eval.get('detector_kwargs', {}),
                    detector_url=config.eval.detector_url,
                    num_gpus=dist.get_world_size(),
                    rank=dist.get_rank(),
                    device=device,
                    ref_stats=config.eval.get('ref_stats', None), 
                    batch_size=config.eval.get('batch_size', 128), 
                )
                if dist.get_rank() == 0: 
                    step = int(name.split("-")[-1]) if isinstance(name.split("-")[-1], int) else None
                    wandb.log( {f"{name}/{key}_{k}": v for k, v in result_dict['results'].items()}, step=step)
                    os.makedirs(os.path.join(run_dir, name, key), exist_ok=True)
                    print(f'{name} FID:', key)
                    metric_main.report_metric(
                        result_dict,
                        run_dir=os.path.join(run_dir, name, key),
                        snapshot_pkl="network-snapshot-latest.pkl",
                    )

    # Done.
    dist.print0()
    dist.print0("Exiting...")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
