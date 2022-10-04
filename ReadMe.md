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


### My results

The figure of traing curves and logs, Coming soon

- Due to computational resource limitation, I use `num_channels=64` for CIFAR10 (reduce Number of Params from 52M to 13M, GPU memory 11.9MiB). `T=1000`.
- I found the EMA 0.9999 is very slow, so I use 0.999.  I found warmup doesn't help at begining.
- The number of iterations is 500 (epochs) * 390(iterations/epoch), batch size is 128. 
- Every epoch takes more than 2.6 minutes. The sampling of 100 images with `T=1000` takes about 100 seconds.
- The training time should be ~21:40:00, not including the sampling and evaluation.




## Sampling and Evaluate IS/FID/KID

I use ```torch_fidelity``` 

To reload other models/other datasets, check `iddpm/script_util.py` and `utils/eval_quality.py`

```
python iddpm_eval.py --eval gen --resume path/your/checkpoint.pth --gpu-id 0
```

- The saved images and log file are under `path/your/checkpoint.pth_eval_gen_date`
- The sampling with `T=1000` takes about 10 minutes to generate 1000 32x32 images, using Unet with `num_channels=64`. 

The comparison to TF https://github.com/sndnyang/inception_score_fid/blob/master/eval_is_fid_torch_fidelity.ipynb 

# Change Log


2022.10.04

Evaluation

1. Generation and Evaluate IS/FID/KID

TODO

1. Faster sampling
2. Evaluate NLL

2022.10.03

Init code

1. Training  iddpm_main.py