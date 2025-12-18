# Terminal Velocity Matching


Official Implementation of [Terminal Velocity Matching](https://arxiv.org/abs/2511.19797)

<p align="center"> 
  <img src="assets/teaser_mixed.png" width="90%"/> 
</p>

<div align="center">
  <span class="author-block">
    <a href="https://alexzhou907.github.io/">Linqi Zhou</a>,</span> 
  <span class="author-block">
    <a href="https://dabeschte.github.io/">Mathias Parger</a>,
  </span>
    <span class="author-block">
    <a href="https://ayaanzhaque.me/">Ayaan Haque</a>,
  </span>
  <span class="author-block">
    <a href="https://tsong.me/">Jiaming Song</a> 
  </span>
</div>

<div align="center">
  <span class="author-block">Luma AI</span>
</div>
<div align="center">
<a href="https://arxiv.org/abs/2511.19797">[Paper]</a>
<a href="https://lumalabs.ai/blog/engineering/tvm">[Blog]</a> 
</div>
</br>

Also check out our previous paper [Inductive Moment Matching](https://arxiv.org/abs/2503.07565).


# Checklist

- [x] Add model weights and model definitions. 
- [x] Add evaluation scripts.
- [x] Add training scripts.


# Dependencies

To install all packages in this codebase along with their dependencies, run
```sh
conda env create -f env.yml
``` 
For multi-node jobs, we use Slurm via `submitit`, which can be installed via
```sh
pip install submitit
```


# Pre-Trained Models

We provide pretrained checkpoints through our [repo](https://huggingface.co/lumaai/tvm) on Hugging Face:

<table>
  <tr>
    <th colspan="4">ImageNet-256x256</th>
  </tr>
  <tr>
    <td>CFG</td>
    <td>NFE</td>
    <td>FID</td>
    <td>Link</td>
  </tr>
  <tr>
    <td>2.0</td>
    <td>1</td>
    <td>3.29</td>
    <td> <a/ link="https://huggingface.co/lumaai/tvm/resolve/main/im256_w2.pkl"> im256_w2.pkl</td>
  </tr>
  <tr>
    <td>1.75</td>
    <td>4</td>
    <td>1.99</td>
    <td> <a/ link="https://huggingface.co/lumaai/tvm/resolve/main/im256_w1p75.pkl"> im256_w1p75.pkl</td>
  </tr>
  <tr>
    <th colspan="4">ImageNet-512x512</th>
  </tr>
  <tr>
    <td>2.5</td>
    <td>1</td>
    <td>4.32</td>
    <td> <a/ link="https://huggingface.co/lumaai/tvm/resolve/main/im512_w2p5.pkl"> im512_w2p5.pkl</td>
  </tr>
</table>

 


# Datasets

Datasets are stored as uncompressed ZIP archives containing uncompressed PNG or NPY files, along with a metadata file `dataset.json` for labels. When using latent diffusion, it is necessary to create two different versions of a given dataset: the original RGB version, used for evaluation, and a VAE-encoded latent version, used for training.

Please refer to [IMM](https://github.com/lumalabs/imm) for dataset setups. ImageNet-512x512 follows similar procedures. 




# Training

The default configs are in `configs/`. Before training, properly replace the `logger` and `dataset.path` fields to your own choice. You can train on a single node by

```bash
bash train.sh NUM_GPUS_PER_NODE CONFIG_NAME REPLACEMENT_ARGS
```

If you want to replace any config args, for example, `network.do_scale_param` on ImageNet-256x256 to `false`, you can run
```bash
bash train.sh 8 im256_w2.yaml network.do_scale_param=false
```

To train multi-node, we use submitit and run
```bash
python launch.py --ngpus=NUM_GPUS_PER_NODE --nodes=NUM_NODES --config-name=CONFIG_NAME
```

And output folder will be created under `./outputs/`

# Evaluation

The eval scripts calculate FID scores. We provide the data reference stats at [img256.pkl](https://huggingface.co/lumaai/tvm/resolve/main/dataset-refs/img256.pkl) and [img512.pkl](https://huggingface.co/lumaai/tvm/resolve/main/dataset-refs/img512.pkl).

```bash
bash eval.sh NUM_GPUS_PER_NODE CONFIG_NAME eval.resume=YOUR_MODEL_PKL_PATH network.compile=false
```

For example, to evaluate on ImageNet-256x256 with your model saved at `./img256_w2.pkl` with 8 GPUs on a single node, run
```bash
bash eval.sh 8 im256_w2.yaml eval.resume=./img256_w2.pkl
```

`YOUR_MODEL_PKL_PATH` can also be a directory containing all checkpoints of a run labeled with training iterations (i.e. your training directory). It will sort from the latest checkpoint to earliest and evaluate in that order.
 

# ImageNet Samples

<p align="center"> 
  <img src="assets/exp_samples.png" width="90%"/> 
</p>
 

# Citation

```
 @misc{zhou2025terminal,
      title={Terminal Velocity Matching}, 
      author={Linqi Zhou and Mathias Parger and Ayaan Haque and Jiaming Song},
      year={2025},
      eprint={2511.19797},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.19797}, 
}
```