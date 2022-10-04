import time
import torch
from datetime import timedelta

from ExpUtils import *
from utils.utils import plot
from utils.eval_quality import eval_is_fid

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
seed = 1
n_ch = 3
n_classes = 10
print = wlog


def eval_buffer(buffer, arg):
    eval_start = time.time()
    plot('{}/samples_0.png'.format(arg.dir_path), buffer[:100])
    metrics = eval_is_fid(buffer, dataset=arg.dataset, args=arg)
    inc_score = metrics['inception_score_mean']
    fid = metrics['frechet_inception_distance']
    print("with %d" % buffer.shape[0])
    print("Inception score of {}".format(inc_score))
    print("FID of score {}".format(fid))


def new_samples(model, diffusion, arg):
    im_sz = arg.image_size
    start = time.time()
    batches = 100 if arg.dataset != 'stl10' else 50
    replay_buffer = torch.FloatTensor(batches * 100, n_ch, im_sz, im_sz).uniform_(-1, 1)
    eval_time = 0
    bs = 100

    # sample_fn = (
    #     diffusion.p_sample_loop if not arg.use_ddim else diffusion.ddim_sample_loop
    # )

    for i in range(batches):

        samples = diffusion.p_sample_loop(
            model, (bs, 3, im_sz, im_sz), clip_denoised=True, model_kwargs=None
        )
        replay_buffer[i * bs: (i + 1) * bs] = (samples + 1) / 2

        if (i + 1) % 10 == 0:
            now = time.time()
            print(f'batch {i + 1} {str(timedelta(seconds=now - start - eval_time))}')

        if (i + 1) in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]:
            eval_start = time.time()
            plot('{}/samples_{}.png'.format(arg.dir_path, i+1), samples)
            metrics = eval_is_fid(replay_buffer[:(i + 1) * bs] * 255, dataset=arg.dataset, args=arg)
            inc_score = metrics['inception_score_mean']
            fid = metrics['frechet_inception_distance']
            print("sample with %d" % (i * bs + bs))
            print("Inception score of {}".format(inc_score))
            print("FID of score {}".format(fid))
            eval_time += time.time() - eval_start
    if "0_check" in arg.resume:
        torch.save(replay_buffer, '{}/buffer.pt'.format(arg.dir_path))
