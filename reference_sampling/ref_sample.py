import torch.nn as nn
import faiss
import numpy as np
import pickle

from tqdm.auto import trange

from edflow.custom_logging import get_logger
from viterbi.transition_network.utilities.create_embeddings import TransProb


class ReferenceSampler():
    '''Using Viterbi as a backend'''

    def __init__(self, targets, k=100):
        '''
        Arguments
        ---------
        targets : torch.Tensor
            All ``M`` possible reference poses. Shape ``[M, Z]``
        transition_probability : Callable
            A function which takes a batch of pairs of pose encodings and
            returns the transition probability from the first to the second.
        '''
        self.logger = get_logger(self)

        self.k = k

        self.targets = targets
        self.targets_flat = self.targets.reshape(len(targets), -1)

        self.nn_sampler = faiss.IndexFlatL2(self.targets_flat.shape[-1])
        self.nn_sampler.add(self.targets_flat)

        n = self.nn_sampler.ntotal
        nt = '' if self.nn_sampler.is_trained else ' not'
        
        self.logger.info(f'NN sampler contains {n} examples and is{nt} trained')

        self.transition_probability = TransProb()

    def get_transition_probs(self, previous_states, current_states):
        '''
        Parameters
        ----------
        previous_states : list
            State indices.
        current_states : list
            State indices.

        Returns
        -------
        trans_p : np.ndarray
            transition probabilities. Indexing goes [current, previous]
        '''

        trans_p = np.zeros([self.k, self.k])

        for idx in range(self.k):
            q_state = previous_states[idx]

            p = self.transition_probability([self.targets[q_state]]*self.k,
                                            self.targets[previous_states])
            trans_p[idx, :] = p.cpu().detach().numpy()[..., 0]

        return trans_p

    def get_emission_probs_and_states(self, query_at_t, log=False):
        '''
        Parameters
        ----------
        query_at_t : t
            keypoints
        log : bool
            return log probabilities

        Returns
        -------
        probs : np.ndarray
            The probabilities of all possible states given the query.
        states : np.ndarray
            Indices of the possible states.
        '''
        distances, states = self.nn_sampler.search(
            query_at_t.reshape(-1)[None],  # [1, 17*2]
            self.k
        )  # [1, K]

        if not log:
            # Gibbs distribution
            energy = np.exp(-distances)
            partition = np.exp(-distances).sum(1)[:, None]

            probabilites = energy / partition
            # [1, K]
        else:
            # Gibbs distribution
            energy = -distances
            partition = -distances.sum(1)[:, None]

            probabilites = energy - partition
            # [1, K]

        # probs need to be [K]
        probs = probabilites[0]

        return probs, states[0]


    def __call__(self, query, name='V'):
        '''
        Arguments
        ---------
        query : np.ndarray
            A sequence of poses. Shape ``[T, K, 2]``
        name : str
            A name used to store the viterbi trace.

        Returns
        -------
        reference : torch.Tensor
            A reference sequence of pose encodings. Shape is the same as
            :attr:`query`.
        '''

        self.T = len(query)

        self.V, self.full_V = self.forward(query)
        store_V(self.V, name)
        store_V(self.full_V, name + '_full')

        return find_optimal(self.V)

    def forward(self, query):
        '''Does the forward tracking'''

        V = [{}]
        full_V = [{}]

        # Initialization: The probability for being in one of the possible
        # states is given by the emission probabilities.
        emit_p, states = self.get_emission_probs_and_states(query[0], True)
        for idx, state in enumerate(states):
            V[0][state] = {"prob": emit_p[idx], "prev": None}

        full_V[0]['emit'] = emit_p
        full_V[0]['states'] = states
        full_V[0]['trans'] = None

        prev_states = states
        
        # Run Viterbi when t > 0
        for t in trange(1, len(query), desc='Time'):
            V.append({})
            # At each timestep iterate over all possible states to find its
            # probability for being the current state and also find its
            # most probable predecessor.

            emit_p, states = self.get_emission_probs_and_states(query[t], True)
            trans_p = self.get_transition_probs(prev_states, states)
            trans_p = np.log(trans_p)

            full_V.append({'emit': emit_p, 'trans': trans_p, 'states': states})

            sum_prob = 0
            for idx, state in enumerate(states):

                # Now go through all previous states and find the one which
                # has the highest transition probability into the current
                # state combined with the probability of being in the
                # previous state in the first place.
                max_tr_prob = -float('inf')
        
                for jdx, prev_state in enumerate(prev_states):
                    tr_prob = V[t-1][prev_state]["prob"] + trans_p[jdx][idx]
        
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = prev_state
                        
                max_prob = max_tr_prob + emit_p[idx]
                V[t][state] = {'prob': max_prob, 'prev': prev_st_selected}

                sum_prob += max_prob

            # for state in states:
            #     V[t][state]['prob'] = V[t][state]['prob'] / sum_prob

            prev_states = states
        return V, full_V
        

def find_optimal(V):
    '''Does the backtracking'''
    # Now find the optimal Viterbi path by backtracking
    opt = []
    max_prob = -float('inf')
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = st
            previous = data['prev']

    opt.append(previous)
    opt.append(best_st)
    
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, 0, -1):
        previous = V[t][previous]["prev"]
        opt.insert(0, previous if previous is None else int(previous))

    return opt


def store_V(V, store_path):
    with open(store_path, 'wb') as pf:
        pickle.dump(V, pf)


def plot_outcome(D, opt, q_idxs, crop_key, full_name):
    import matplotlib.pyplot as plt
    from edflow.data.util import adjust_support

    f, AX = plt.subplots(2, len(opt),
                         figsize=[len(opt)/10*12.8, 7.2],
                         dpi=100,
                         constrained_layout=True)

    D.expand = True

    for i, [Ax, indices] in enumerate(zip(AX, [q_idxs, opt])):
        for ax, idx in zip(Ax, indices):
            D.base.loader_kwargs['target']['root'] = '/home/jhaux/remote/cg2'
            ex = D[idx]
            crop = D[idx][crop_key]
            im = adjust_support(crop, '0->1')
            ax.imshow(im)
            ax.axis('off')
            ax.set_title(f'{"Q" if i == 0 else "R"}: {idx}')

    f.savefig(f'{full_name}.pdf')


def plot_V(full_V, V, D, opt, q_idxs, crop_key, full_name, opt_orig):
    import matplotlib.pyplot as plt
    from edflow.data.util import adjust_support

    D.expand = True

    for t in range(1, len(full_V)):
        f = plt.figure(constrained_layout=True,
                       figsize=[12.8, 12.8],
                       dpi=100)

        gs = f.add_gridspec(3, 3,
                            height_ratios=[1, 1, 10],
                            width_ratios=[1, 1, 10],
                            wspace=0.1, hspace=0.1)

        trans = full_V[t]['trans']
        nt, ntm1 = trans.shape[:2]

        final_prob = [v['prob'] for v in V[t].values()]
        fin_prob_im = np.array(final_prob)[None]

        state_prob = [v['prob'] for v in V[t-1].values()]
        state_prob_im = np.array(state_prob)[:, None]

        states_tm1 = full_V[t-1]['states']
        prev_st = [list(states_tm1).index(v['prev']) for v in V[t].values()]

        chosen_state = opt[t-1]
        choice = list(states_tm1).index(chosen_state)
        choice_idx = prev_st.index(choice)

        im_q = adjust_support(D[q_idxs[t]][crop_key], '0->255')
        im_q_tm1 = adjust_support(D[q_idxs[t-1]][crop_key], '0->255')

        nn_t = [adjust_support(D[i][crop_key], '0->255') for i in full_V[t]['states']]
        nn_tm1 = [adjust_support(D[i][crop_key], '0->255') for i in full_V[t-1]['states']]
        im_nn_t = np.concatenate(nn_t, 1)
        im_nn_tm1 = np.concatenate(nn_tm1, 0)

        dx, dy = im_q.shape[:2]

        axes = []

        ax_t = f.add_subplot(gs[2, 2])
        ax_t.imshow(trans)
        ax_t.set_title('transition probability')
        ax_t.scatter(list(range(nt)), prev_st, color='r', marker='x')
        ax_t.scatter(choice_idx, choice, color='r', marker='X', s=150)
        axes += [ax_t]

        ax_qt = f.add_subplot(gs[0, 1])
        ax_qt.imshow(im_q)
        ax_qt.set_title(f'query @ {t}')
        axes += [ax_qt]

        ax_qtp = f.add_subplot(gs[1, 2])
        ax_qtp.imshow(fin_prob_im)
        ax_qtp.set_title(f'final prob')
        axes += [ax_qtp]

        ax_qtpm1 = f.add_subplot(gs[2, 1])
        ax_qtpm1.imshow(state_prob_im)
        ax_qtpm1.set_title(f'state prob')
        axes += [ax_qtpm1]

        ax_qtm1 = f.add_subplot(gs[1, 0])
        ax_qtm1.imshow(im_q_tm1)
        ax_qtm1.set_title(f'query @ {t-1}')
        axes += [ax_qtm1]

        ax_nnt = f.add_subplot(gs[0, 2])
        ax_nnt.imshow(im_nn_t)
        ax_nnt.set_title('nearest neighbours')
        axes += [ax_nnt]

        ax_nntm1 = f.add_subplot(gs[2, 0])
        ax_nntm1.imshow(im_nn_tm1)
        ax_nntm1.set_ylabel('nearest neighbours')
        axes += [ax_nntm1]

        for ax in axes:
            ax.axis('off')

        f.savefig(f'{full_name}_t{t}.pdf')
def _add_text(img, text):
    from PIL import ImageFont
    from PIL import ImageDraw 

    origin = x, y = (5, 5)
    fs = 30

    rect = (x - 2, y - 2, x + fs + 2, y + fs + 2)

    draw = ImageDraw.Draw(img)
    draw.rectangle(rect, fill=(255, 255, 255, 64))
    font = ImageFont.truetype("./fonts/fira/FiraSans-Medium.otf", fs)
    draw.text(origin, text, ( 0, 0, 0), font=font)
    del draw

    return img


def make_video(D, q_idxs, opt, crop_name, full_name):
    import tempfile
    from PIL import Image
    from edflow.data.util import adjust_support

    D.expand = True
    
    with tempfile.TemporaryDirectory() as tmpd:
        for i, [q, o] in enumerate(zip(q_idxs, opt)):
            imq = Image.fromarray(adjust_support(D[q][crop_name], '0->255'))
            imo = Image.fromarray(adjust_support(D[o][crop_name], '0->255'))

            imq = _add_text(imq, 'Q')
            imo = _add_text(imo, 'R')

            qpath = os.path.join(tmpd, f'q_{i:0>4d}.png')
            imq.save(qpath)
            imo.save(os.path.join(tmpd, f'o_{i:0>4d}.png'))

        pat_q = os.path.join(tmpd, 'q_%04d.png')
        pat_o = os.path.join(tmpd, 'o_%04d.png')
        name_q = os.path.join(tmpd, 'q.mp4')
        name_o = os.path.join(tmpd, 'o.mp4')
        out_name = f'{full_name}.mp4'

        vid_command = 'ffmpeg -i {im_pat} -vf fps=25 -vcodec libx264  -crf 18 {name}'

        qcommand = vid_command.format(im_pat=pat_q, name=name_q)
        os.system(qcommand)
        os.system(vid_command.format(im_pat=pat_o, name=name_o))

        stack_command = 'ffmpeg -y -i {top} -i {bot} -vcodec libx264  -crf 18 -filter_complex hstack {name}'
        os.system(stack_command.format(top=name_q, bot=name_o, name=out_name))


if __name__ == '__main__':
    from edflow.util import edprint
    from edflow.data.believers.meta_view import MetaViewDataset
    from edflow.data.believers.sequence import get_sequence_view
    import os

    import argparse
    A = argparse.ArgumentParser()

    A.add_argument('-s', action='store_true', help='Sample new reference')
    A.add_argument('-p', action='store_true', help='Use Prjoti_J dset')
    A.add_argument('name', type=str, help='Experiment name')
    A.add_argument('-n', type=int, default=20, help='number timesteps')
    A.add_argument('-k', type=int, default=100, help='number NNs')
    A.add_argument('-m', type=int, default=10, help='number querys')

    args = A.parse_args()

    sample = args.s
    prjoti = args.p
    name = args.name

    N = args.n
    K = args.k
    M = args.m

    if prjoti:
        D = Prjoti({'data_root': '/home/jhaux/Dr_J/Projects/VUNet4Bosch/Prjoti_J/'})
        kps = D.labels['kps_rel'][..., :2].astype('float32')
        kps = kps[:, [8, 9, 10, 11, 12, 13], :]
        crop_key = 'crop'

        healthy = D.labels['pid'] == 'J'  # ;)
        fids = D.labels['fid']
    else:
        root = "/export/scratch/jhaux/Data/human gait/train_view"
        if os.environ['HOME'] == '/home/jhaux':
            root = os.path.join('/home/jhaux/remote/cg2', root[1:])
        D = MetaViewDataset(root)
        kps = D.labels['kps_fixed_rel'][..., :2].astype('float32')
        crop_key = 'target'

        healthy = D.labels['healthy']
        fids = D.labels['fid']

    impaired = np.logical_not(healthy)

    hidden_idxs = np.arange(len(kps))[healthy]
    kps_hidden = kps[healthy]

    query_views = get_sequence_view(fids[impaired], N, strategy='reset')
    query_idxs = np.arange(len(kps))[impaired]
    kps_query = kps[impaired]

    prng = np.random.RandomState(42)
    for i in range(M):
        start = prng.randint(len(query_views))
        q_idxs = query_views[start]

        q = np.take(kps_query, q_idxs, axis=0)

        full_name = f'{name}_{i}_{N}x{K}'

        if sample:
            R = ReferenceSampler(kps_hidden, k=K)
            opt = R(q, f'{full_name}.p')
            full_V = R.full_V
            V = R.V
        else:
            with open(f'{full_name}.p', 'rb') as pf:
                V = pickle.load(pf)
            with open(f'{full_name}.p_full', 'rb') as pf:
                full_V = pickle.load(pf)
            opt = find_optimal(V)

        for t in range(len(full_V)):
            full_V[t]['states'] = [hidden_idxs[idx] for idx in full_V[t]['states']]
            for st in V[t].keys():
                V[t][st]['prev'] = hidden_idxs[V[t][st]['prev']]
            for st in list(V[t].keys()):
                V[t][hidden_idxs[st]] = V[t][st]
                del V[t][st]

        opt_orig = opt
        opt = np.take(hidden_idxs, opt, 0)
        q_idxs = np.take(query_idxs, q_idxs, 0)

        # plot_outcome(D, opt, q_idxs, crop_key, full_name)
        # plot_V(full_V, V, D, opt, q_idxs, crop_key, full_name, opt_orig)
        make_video(D, q_idxs, opt, crop_key, full_name)
