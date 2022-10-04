import argparse
import numpy as np
import torch.optim
import torch.utils.data
import torch.nn as nn
import torchvision.transforms as transforms

from ExpUtils import *
from utils.dataloader import datainfo, dataload
from eval_tasks import *
from iddpm.script_util import create_model_and_diffusion, model_and_diffusion_defaults
import warnings
warnings.filterwarnings("ignore", category=Warning)

best_acc1 = 0


def init_parser():
    arg_parser = argparse.ArgumentParser(description='evaluate script')
    arg_parser.add_argument("--eval", default="OOD", type=str, choices=["gen", 'nll'])

    # Data args
    arg_parser.add_argument('--data_path', default='../../data', type=str, help='dataset path')
    arg_parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tinyimg', 'svhn', 'mnist', 'stl10',
                                                                     'celeba', 'img32', 'img128', 'img12810', 'imgnet', 'img10'],
                            type=str, help='Image Net dataset path')
    arg_parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    arg_parser.add_argument('--print-freq', default=500, type=int, metavar='N', help='log frequency (by iteration)')

    arg_parser.add_argument('--resume', default=False, help='Version')
    arg_parser.add_argument('--buffer_path', default='', type=str, help='path for saved buffer')

    # Optimization hyperparams
    arg_parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)', dest='batch_size')
    arg_parser.add_argument('--t_step', default=1000, type=int, metavar='N', help='T')
    arg_parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'l2'])

    arg_parser.add_argument('--no_cuda', action='store_true', help='disable cuda')
    
    arg_parser.add_argument("--multi", action="store_true", help="maybe the model is trained with DataParallel")

    arg_parser.add_argument("--log_dir", type=str, default='eval')
    arg_parser.add_argument("--log_arg", type=str, default='model-pyx-px')
    arg_parser.add_argument("--novis", action="store_true", help="")
    arg_parser.add_argument("--no_fid", action="store_true", help="")
    arg_parser.add_argument("--debug", action="store_true", help="")
    arg_parser.add_argument("--exp_name", type=str, default="iDDPM", help="exp name, for description")
    arg_parser.add_argument("--seed", type=int, default=1)
    arg_parser.add_argument("--gpu-id", type=str, default="0")
    arg_parser.add_argument("--note", type=str, default="")

    arg_parser.add_argument("--wandb", action="store_true", help="If set, use wandb")

    return arg_parser


def run_bpd_on_dataset(nll_model, f, loader, arg):
    all_bpd = []
    c = 0
    start = time.time()
    for images, _ in loader:
        images = images.to(arg.device)
        c += images.shape[0]
        minibatch_metrics = nll_model.calc_bpd_loop(f, images, clip_denoised=True)
        total_bpd = minibatch_metrics["total_bpd"]
        total_bpd = total_bpd.mean()
        all_bpd.append(total_bpd.item())
        if c % 1000 == 0:
            print(f'{c} bpd: {total_bpd.item()}')
    bpd = np.mean(all_bpd)
    end = time.time()
    print(f"done {c} samples: bpd={bpd}, takes {end - start}")
    return bpd


def main(arg):
    global best_acc1

    data_info = datainfo(logger, arg)

    if arg.eval == 'buffer':
        buffer = torch.load(arg.resume)
        eval_buffer(buffer)

    normalize = [transforms.Normalize(mean=data_info['stat'][0], std=data_info['stat'][1])]
    augmentations = transforms.Compose([
        transforms.ToTensor(),
        *normalize
    ])

    '''
        model
    '''

    model, diffusion = create_model_and_diffusion(
        **model_and_diffusion_defaults()
    )
    if arg.multi:
        model = nn.DataParallel(model, device_ids=range(len(arg.gpu_id.split(','))))
    else:
        model.to(args.device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of params: {format(n_parameters, ",")}')

    checkpoint = torch.load(arg.resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    if arg.eval == 'gen':
        new_samples(model, diffusion, arg)

    if arg.eval == 'nll':
        assert arg.dataset == 'cifar10', "It's simple to load other datasets"
        # https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py#L709
        from iddpm.gaussian_diffusion import GaussianDiffusion as NllDiffusion, LossType, ModelVarType, ModelMeanType
        nll_model = NllDiffusion(
            betas=diffusion.betas,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.LEARNED,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )
        train_set, val_dataset = dataload(arg, augmentations, normalize, data_info)
        # make sure that batch size doesn't matter, since the model use laynorm, not batch norm. small batch size is very slow.
        train_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, pin_memory=True, num_workers=arg.workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, pin_memory=True, num_workers=arg.workers)

        val_bpd = run_bpd_on_dataset(nll_model, model, val_loader, arg)
        train_bpd = run_bpd_on_dataset(nll_model, model, train_loader, arg)
        print(f'train bpd {train_bpd}, test bpd {val_bpd}')


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.SPT = None
    args.LSA = None

    print = wlog
    print(' '.join(sys.argv))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    run_time = time.strftime('%m%d%H%M%S', time.localtime(time.time()))
    if args.log_dir == 'eval':
        # by default to eval the model
        args.dir_path = args.resume + "_eval_%s_%s" % (args.eval, run_time)
    set_file_logger(logger, args)
    print(args.dir_path)

    main(args)

    print(args.dir_path)
    print(' '.join(sys.argv))
