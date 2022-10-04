# improved-diffusion iDDPM

Thanks a lot for the official code https://github.com/openai/improved-diffusion

However, I fail to install `mpi4py` for multi/distributed training. And I think `mpi4py` and `torch.distributed` are not for prototyping. So I remove those components and the code should run on a single GPU without `mpi4py`.

I generate and save the generated images, and evaluate the IS/FID/KID during training. I keep a replay buffer to store generated images for IS/FID/KID


```
pip install torch  torch_fidelity
```

optional: wandb

# Usage

## Training

The training command is simple.

```
python iddpm_main.py --gpu-id 0 --print-freq 100 --note warmup    [--no_fid]
```

The hyperparameters please refer to https://github.com/openai/improved-diffusion#models-and-hyperparameters 
and change iddpm/script_util.py. 

**Note** 

A larger `diffusion_steps` will cause a much longer generation during traing.  You can use `--no_fid` to ignore the sampling.

## Sampling and Evaluate IS/FID/KID

I use ```torch_fidelity``` 

The comparison to TF https://github.com/sndnyang/inception_score_fid/blob/master/eval_is_fid_torch_fidelity.ipynb 

TODO
