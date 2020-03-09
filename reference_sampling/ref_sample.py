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
        print('K', self.k)

        self.targets = targets
        self.targets_flat = self.targets.reshape(len(targets), -1)
        print(self.targets_flat.shape)

        self.nn_sampler = faiss.IndexFlatL2(self.targets_flat.shape[-1])
        self.nn_sampler.add(self.targets_flat)

        n = self.nn_sampler.ntotal
        nt = '' if self.nn_sampler.is_trained else ' not'
        
        self.logger.info(f'NN sampler contains {n} examples and is{nt} trained')

        self.transition_probability = TransProb()

    def emission_probabilities_and_states(self, query_sequence):
        # distances are l2 kp distances
        # states are integers

        self.logger.info('Computing emission probs...')

        distances, states = self.nn_sampler.search(
            query_sequence.reshape(len(query_sequence), -1),  # [T, 17*2]
            self.k
        )  # [T, K]

        print(distances.shape)
        print((-distances).min(), (-distances).max())

        # Gibbs distribution
        energy = np.exp(-distances)
        partition = np.exp(-distances).sum(1)[:, None]

        print(energy.min(), energy.max())
        print(partition.min(), partition.max())
        probabilites = energy / partition
        print(probabilites.min(), probabilites.max())
        # [T, K]

        # probs need to be [T, T*K]
        probs_wide = np.zeros([self.T, self.k * self.T])
        for t in range(self.T):
            probs_wide[t, t*self.k:(t+1)*self.k] = probabilites[t]

        # states need to be [T*K]
        states = states.flatten()

        self.logger.info('Done')

        return probs_wide, states

    def get_transition_probability(self, states):
        '''
        Arguments
        ---------
        states : k*T states
            All possible states
        ingnore_step : int
            transitions are only possible from t to t+1 and only between every
            pair of following sets of ``ignore_step`` states.
        '''

        self.logger.info('Computing transition probs...')

        trans_p = np.zeros([self.T*self.k, self.T*self.k])

        for idx1 in trange(self.k*(self.T-1), desc='pt 1'):
            interval = idx1 // self.k + 1
            # Transitions are only possible between timesteps
            pos_trans_states = slice(interval * self.k, (interval + 1) * self.k)
            q_state = states[idx1]
            ref_states = states[pos_trans_states]
            l = len(ref_states)
            p = self.transition_probability([self.targets[q_state]]*l,
                                            self.targets[ref_states])
            trans_p[idx1, pos_trans_states] = p.cpu().detach().numpy()[..., 0]

        self.logger.info('Done')

        return trans_p

    def get_transition_probs(self, previous_states, current_states):
        trans_p = np.zeros([self.k, self.k])

        for idx in range(self.k):
            q_state = previous_states[idx]

            p = self.transition_probability([self.targets[q_state]]*self.k,
                                            self.targets[previous_states])
            trans_p[idx, :] = p.cpu().detach().numpy()[..., 0]

        return trans_p

    def get_emission_probs_and_states(self, query_at_t):
        distances, states = self.nn_sampler.search(
            query_at_t.reshape(-1)[None],  # [1, 17*2]
            self.k
        )  # [1, K]

        # Gibbs distribution
        energy = np.exp(-distances)
        partition = np.exp(-distances).sum(1)[:, None]

        probabilites = energy / partition
        # [1, K]

        # probs need to be [K]
        probs = probabilites[0]

        return probs, states[0]


    def __call__(self, query, name='V'):
        '''
        Arguments
        ---------
        query : torch.Tensor
            A sequence of poses. Shape ``[T, Z]``

        Returns
        -------
        reference : torch.Tensor
            A reference sequence of pose encodings. Shape is the same as
            :attr:`query`.
        '''

        self.T = len(query)

        V = self.forward(query)
        store_V(V, name)
        return find_optimal(V)

    def forward(self, query):
        '''Does the forward tracking'''

        V = [{}]

        # Initialization: The probability for being in one of the possible
        # states is given by the emission probabilities.
        emit_p, states = self.get_emission_probs_and_states(query[0])
        emit_p = np.log(emit_p)
        for st in range(len(states)):
            st = str(st)
            V[0][st] = {"prob": emit_p[int(st)], "prev": None}

        prev_states = states
        
        # Run Viterbi when t > 0
        for t in trange(1, len(query), desc='Time'):
            V.append({})
            # At each timestep iterate over all possible states to find its
            # probability for being the current state and also find its
            # most probable predecessor.

            emit_p, states = self.get_emission_probs_and_states(query[t])
            emit_p = np.log(emit_p)
            trans_p = self.get_transition_probs(prev_states, states)
            trans_p = np.log(trans_p)
            prev_states = states

            sum_prob = 0
            for st in range(self.k):
                st = str(st)

                # Now go through all previous states and find the one which
                # has the highest transition probability into the current
                # state combined with the probability of being in the
                # previous state in the first place.
                max_tr_prob = V[t-1][st]["prob"] + trans_p[0][int(st)]
                prev_st_selected = str(0)
        
                for prev_st in range(1, self.k):
                    prev_st = str(prev_st)
                    tr_prob = V[t-1][prev_st]["prob"] + trans_p[int(prev_st)][int(st)]
        
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = prev_st
                        
                max_prob = max_tr_prob + emit_p[int(st)]
                V[t][st] = {'prob': max_prob, 'prev': str(prev_st_selected)}

                sum_prob += max_prob

            for st in range(self.k):
                V[t][str(st)]['prob'] = V[t][str(st)]['prob'] / sum_prob
        return V
        
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
            print(f'Changing best to {st}')

    opt.append(int(best_st))
    previous = best_st
    
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, 0, -1):
        print(f't, Prev: {t}, {previous}, {type(previous)}')
        print(V[t+1].keys())
        previous = str(previous)
        previous = V[t][previous]["prev"]
        opt.insert(0, previous if previous is None else int(previous))

    return opt

def store_V(V, store_path):
    with open(store_path, 'wb') as pf:
        pickle.dump(V, pf)


if __name__ == '__main__':
    from edflow.util import edprint
    from edflow.data.believers.meta_view import MetaViewDataset
    import os

    import argparse
    A = argparse.ArgumentParser()

    A.add_argument('-s', action='store_true', help='Sample new reference')
    A.add_argument('-p', action='store_true', help='Use Prjoti_J dset')
    A.add_argument('name', type=str, help='Experiment name')
    A.add_argument('-n', type=int, default=20, help='number timesteps')
    A.add_argument('-k', type=int, default=100, help='number NNs')

    args = A.parse_args()

    sample = args.s
    prjoti = args.p
    name = args.name

    N = args.n
    K = args.k

    full_name = f'{name}_{N}x{K}'

    if prjoti:
        D = Prjoti({'data_root': '/home/jhaux/Dr_J/Projects/VUNet4Bosch/Prjoti_J/'})
        kps = D.labels['kps_rel'][..., :2].astype('float32')
        kps = kps[:, [8, 9, 10, 11, 12, 13], :]
        crop_key = 'crop'
    else:
        root = "/export/scratch/jhaux/Data/human gait/train_view"
        if os.environ['HOME'] == '/home/jhaux':
            root = os.path.join('/home/jhaux/remote/cg2', root[1:])
        D = MetaViewDataset(root)
        kps = D.labels['kps_fixed_rel'][..., :2].astype('float32')
        crop_key = 'target'
    edprint(D.labels)

    print(kps.shape)

    hidden_start = 0
    hidden_end = int(0.84 * len(D))
    kp_hidden = kps[hidden_start:hidden_end]

    start = int(0.84 * len(D))
    end = start + N
    q_idxs = list(range(start, end))
    print(q_idxs)

    q = kps[start: end]

    if sample:
        R = ReferenceSampler(kp_hidden, k=K)
        opt = R(q, f'{full_name}.p')
    else:
        with open(f'{full_name}.p', 'rb') as pf:
            V = pickle.load(pf)
        edprint(V)
        opt = find_optimal(V)

    opt = [hidden_start + i for i in opt]

    import matplotlib.pyplot as plt
    from edflow.data.util import adjust_support

    f, AX = plt.subplots(2, len(opt),
                         figsize=[N/10*12.8, 7.2],
                         dpi=100,
                         constrained_layout=True)

    D.expand = True

    for i, [Ax, indices] in enumerate(zip(AX, [q_idxs, opt])):
        for ax, idx in zip(Ax, indices):
            print(D.base)
            D.base.loader_kwargs['target']['root'] = '/home/jhaux/remote/cg2'
            edprint(D.base.loader_kwargs)
            edprint(D.meta)
            ex = D[idx]
            edprint(ex)
            crop = D[idx][crop_key]
            im = adjust_support(crop, '0->1')
            ax.imshow(im)
            ax.axis('off')
            ax.set_title(f'{"Q" if i == 0 else "R"}: {idx}')

    f.savefig(f'{full_name}.pdf')
