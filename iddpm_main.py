import os
import sys
import time
import copy
import argparse
from datetime import timedelta
import functools

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from ExpUtils import *
from utils.dataloader import datainfo, dataload
from utils.utils import ema, sample_ema_iddpm, iddpm_sample_q
from iddpm.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from iddpm.resample import create_named_schedule_sampler
import warnings
warnings.filterwarnings("ignore", category=Warning)

best_acc1 = 0


def init_parser():
    arg_parser = argparse.ArgumentParser(description='CIFAR10 quick training script')

    # Data args
    arg_parser.add_argument('--data_path', default='../../data', type=str, help='dataset path')
    arg_parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tinyimg', 'svhn', 'mnist', 'stl10'], type=str, help='Image Net dataset path')
    arg_parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    arg_parser.add_argument('--print-freq', default=500, type=int, metavar='N', help='log frequency (by iteration)')

    # Optimization hyperparams
    arg_parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
    arg_parser.add_argument('--warmup', default=10, type=int, metavar='N', help='number of warmup epochs')
    arg_parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)', dest='batch_size')
    arg_parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')

    arg_parser.add_argument('--wd', default=0, type=float, help='weight decay (default: 0)')

    arg_parser.add_argument('--t_step', default=1000, type=int, metavar='N', help='T, but not use it')
    arg_parser.add_argument('--beta_schd', type=str, default='cosine', choices=['cosine', 'linear'], help='not use it')
    arg_parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'l2'], help='not use it')

    arg_parser.add_argument('--no_cuda', action='store_true', help='disable cuda')
    
    arg_parser.add_argument('--resume', default=False, help='Version')

    arg_parser.add_argument("--log_dir", type=str, default='./runs')
    arg_parser.add_argument("--log_arg", type=str, default='iDDPM-epoch-wd')
    arg_parser.add_argument("--novis", action="store_true", help="")
    arg_parser.add_argument("--no_fid", action="store_true", help="")
    arg_parser.add_argument("--debug", action="store_true", help="")
    arg_parser.add_argument("--exp_name", type=str, default="iDDPM", help="exp name, for description")
    arg_parser.add_argument("--seed", type=int, default=1)
    arg_parser.add_argument("--gpu-id", type=str, default="0")
    arg_parser.add_argument("--note", type=str, default="")

    arg_parser.add_argument("--wandb", action="store_true", help="If set, use wandb")

    return arg_parser


def main(arg):
    global best_acc1

    # torch.cuda.set_device(arg.gpu)
    data_info = datainfo(logger, arg)
    
    model, diffusion = create_model_and_diffusion(
        **model_and_diffusion_defaults()
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Creating model: Unet")
    print(f'Number of params: {format(n_parameters, ",")}')

    if ',' in args.gpu_id:
        model = nn.DataParallel(model, device_ids=range(len(arg.gpu_id.split(','))))
    else:
        model.to(args.device)
    print(f'Initial learning rate: {arg.lr:.6f}')
    print(f"Start training for {arg.epochs} epochs")

    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)

    normalize = [transforms.Normalize(mean=data_info['stat'][0], std=data_info['stat'][1])]

    '''
        Data Augmentation
    '''
    augmentations = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        *normalize
    ]

    augmentations = transforms.Compose(augmentations)

    train_dataset, _ = dataload(arg, augmentations, normalize, data_info)

    # you can reduce the batch size for p(x), reduce the training time a bit
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=arg.batch_size, shuffle=True, pin_memory=True, num_workers=arg.workers)

    '''
        Training
    '''
    ema_model = copy.deepcopy(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=arg.lr, weight_decay=arg.wd)

    n_ch = 3
    im_sz = arg.img_size
    buffer = torch.FloatTensor(10000 if arg.dataset != 'stl10' else 5000, n_ch, im_sz, im_sz).uniform_(-1, 1)
    # summary(model, (3, data_info['img_size'], data_info['img_size']))

    if arg.resume:
        checkpoint = torch.load(arg.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        ema_model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            final_epoch = arg.epochs
            arg.epochs = final_epoch - (checkpoint['epoch'] + 1)

    print("Beginning training")
    test_acc = 0
    sample_time = 0
    for epoch in range(arg.epochs):
        epoch_start = time.time()
        output = train(train_loader, 
                       model, ema_model, diffusion, schedule_sampler,
                       optimizer, epoch, arg)
        lr, avg_loss, avg_dif = output
        metrics = {'lr': lr, 'loss': avg_loss}
        tf_metrics = {"lr": lr, "Train/Loss": avg_loss, "Train/DifLoss": avg_dif}
        end = time.time()

        torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(arg.save_path, 'checkpoint.pth'))

        if arg.dataset in ['cifar10', 'cifar100'] and not arg.no_fid:
            sample_start = time.time()
            # sample_ema(diffusion_model, model_buffer, epoch, arg, title=None)
            # model.train()
            iddpm_sample_q(model, diffusion, epoch, arg, num=16, save=True, i=0, title='_o')

            inc, fid = sample_ema_iddpm(ema_model, diffusion, buffer, epoch, arg)
            sample_end = time.time()
            print(f'sample takes {sample_end - sample_start}')
            sample_time += sample_end - sample_start
            if fid != 0:
                metrics['IS'] = inc
                metrics['fid'] = fid

        for k in tf_metrics:
            v = tf_metrics[k]
            arg.writer.add_scalar(k, v, epoch)

        if arg.wandb:
            import wandb
            wandb.log(metrics)

        remain_time = (args.epochs - epoch) * (end - epoch_start)
        total_time = args.epochs * (end - epoch_start)
        print(f'PID {arg.pid} Total ~ {str(timedelta(seconds=total_time))}, '
              f'epoch {str(timedelta(seconds=end-epoch_start))},'
              f'remain {str(timedelta(seconds=remain_time))}')

    print(f'total sample time {str(timedelta(seconds=sample_time))}')
    print(f"Creating model: {arg.model}")
    print(f'Number of params: {format(n_parameters, ",")}')
    print(f'Initial learning rate: {arg.lr:.6f}')
    print(f'best top-1: {best_acc1:.2f}, final top-1: {test_acc:.2f}')
    torch.save({'model_state_dict': model.state_dict(), 'epoch': args.epochs - 1, 'optimizer_state_dict': optimizer.state_dict()},
               os.path.join(arg.save_path, 'checkpoint.pth'))
    torch.save({'model_state_dict': ema_model.state_dict()}, os.path.join(arg.save_path, 'ema_checkpoint.pth'))


def train(train_loader,
          model, ema_model, diffusion, schedule_sampler,
          optimizer, epoch, arg):
    model.train()
    n = 0
    avg_dif_loss = 0
    lr = arg.lr
    for i, (x_p, y_p) in enumerate(train_loader):
        x_p = x_p.to(arg.device)

        y_p = y_p.to(arg.device)
        # assign y_p to model_kwargs for conditional training
        n += x_p.size(0)

        optimizer.zero_grad()

        t, weights = schedule_sampler.sample(x_p.shape[0], arg.device)
        compute_losses = functools.partial(
            diffusion.training_losses,
            model,
            x_p,
            t,
            model_kwargs=None,
        )

        losses = compute_losses()
        dif_loss = (losses["loss"] * weights).mean()

        loss = dif_loss
        avg_dif_loss += float(dif_loss.item() * x_p.size(0))
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 0.999? or 0.995?
        ema(model, ema_model, 0.999)

        lr = optimizer.param_groups[0]["lr"]

        if arg.print_freq >= 0 and i % arg.print_freq == 0:
            avg_dif = avg_dif_loss / n
            avg_loss = avg_dif
            size = len(train_loader)
            print(f'[Epoch {epoch+1}/{arg.epochs}][{i:4d}:{size}]  Loss: {avg_loss:.4f} Dif: {avg_dif:.4f} LR: {lr:.6f}')

    avg_dif = avg_dif_loss / n
    avg_loss = avg_dif
    return lr, avg_loss, avg_dif


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    print = wlog
    print(' '.join(sys.argv))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    init_env(args, logger)
    print(args.dir_path)

    main(args)

    print(args.dir_path)
    print(' '.join(sys.argv))
