import matplotlib.pyplot as plt
from edflow.data.util import adjust_support
from abc_interpolation.datasets.human_gait.human_gait import HumanGaitFixedBox
from abc_interpolation.datasets.human_gait.human_gait import HG_base
from edflow.util import edprint
import os
import numpy as np
from PIL import Image

from reference_sampling.ref_sample import ReferenceSampler


def image_plot(start):

    N = 20
    K = 100

    HG = HumanGaitFixedBox({'data_split': 'train'})
    edprint(HG.labels)

    kps = HG.labels['kps_fixed_rel'][..., :2].astype('float32')
    print(kps)

    kp_hidden = kps[:int(0.84 * len(HG))]

    R = ReferenceSampler(kp_hidden, k=K)

    start_ = int(0.84 * len(HG))
    start += start_
    end = start + N
    q_idxs = list(range(start, end))
    print(q_idxs)
    q = kps[start: end]

    opt = R(q)

    print(opt)


    f, AX = plt.subplots(2, len(opt), figsize=[N/10*12.8, 7.2], dpi=100, constrained_layout=True)

    HG.expand = True

    for Ax, indices in zip(AX, [q_idxs, opt]):
        for ax, idx in zip(Ax, indices):
            im = adjust_support(HG[idx]['target'], '0->1')
            ax.imshow(im)
            ax.axis('off')

    f.savefig(f'viterbi_{start}.pdf')

def vid(start, length=250, K=100):

    save_root = f'./video/{start}/'
    query_root = os.path.join(save_root, 'query')
    ref_root = os.path.join(save_root, 'reference')

    os.makedirs(save_root, exist_ok=True)
    os.makedirs(query_root, exist_ok=True)
    os.makedirs(ref_root, exist_ok=True)

    N = length

    HG = HumanGaitFixedBox({'data_split': 'train'})
    vidps = HG.labels['video_path']
    vidns = np.char.rpartition(vidps, '/')[:, -1]
    healths = np.char.rpartition(vidns, 'H')[:, 1]

    healthy = healths == 'H'
    unhealthy = healths == ''

    h_indices = np.arange(len(HG))[healthy]
    i_indices = np.arange(len(HG))[unhealthy]

    kps = HG.labels['kps_fixed_rel'][..., :2].astype('float32')

    kp_hidden = kps[healthy]

    R = ReferenceSampler(kp_hidden, k=K)

    unhealthy_kps = kps[unhealthy]

    end = start + N
    q_idxs = list(range(start, end))
    q = unhealthy_kps[start: end]

    opt = R(q)

    print(opt)

    HG.expand = True

    vids = []
    for which, indices in [['query', q_idxs], ['reference', opt]]:
        save = os.path.join(save_root, which)
        for i, idx in enumerate(indices):
            if which == 'reference':
                im = HG[h_indices[idx]]['target']
            else:
                im = HG[i_indices[idx]]['target']
            im = adjust_support(im, '0->255')
            Image.fromarray(im).save(os.path.join(save, f'{i:0>3}.png'))
        vid = os.path.join(save_root, f'{which}.mp4')
        vids += [vid]
        command = f'ffmpeg -y -i {save}/%3d.png {vid}'
        os.system(f'{command}')

    stack = os.path.join(save_root, f'query_vs_ref_{start}-{length}-{K}.mp4')
    command = f'ffmpeg -y -i {vids[0]} -i {vids[1]} -filter_complex hstack {stack}'
    os.system(f'{command}')


if __name__ == '__main__':
    from argparse import ArgumentParser

    A = ArgumentParser()

    A.add_argument('start', default=0, type=int)
    A.add_argument('-l', '--length', default=25, type=int)
    A.add_argument('-k', default=100, type=int)

    args = A.parse_args()

    s = args.start
    l = args.length
    k = args.k

    vid(s, l, k)
