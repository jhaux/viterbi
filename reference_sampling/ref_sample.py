import torch.nn as nn
import faiss
import numpy as np

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

    def __call__(self, query):
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

        # [T, T*K], [T*K]
        emit_p, states = self.emission_probabilities_and_states(query)
        # [T*K, T*K]
        trans_p = self.get_transition_probability(states)

        self.logger.info(f'e: {emit_p.min()}, {emit_p.max()}')
        self.logger.info(f't: {trans_p.min()}, {trans_p.max()}')

        V = [{}]

        # Initialization: The probability for being in one of the possible
        # states is given by the emission probabilities.
        for st in range(len(trans_p)):
            V[0][st] = {"prob": emit_p[0][st], "prev": None}
        
        # Run Viterbi when t > 0
        for t in range(1, len(query)):
            V.append({})
            # At each timestep iterate over all possible states to find its
            # probability for being the current state and also find its
            # most probable predecessor.
            for st in range(len(trans_p)):

                # Now go through all previous states and find the one which
                # has the highest transition probability into the current
                # state combined with the probability of being in the
                # previous state in the first place.
                max_tr_prob = V[t-1][st]["prob"] * trans_p[0][st]
                prev_st_selected = 0
        
                for prev_st in range(1, len(trans_p)):

                    # TODO: check trans_p
                    tr_prob = V[t-1][prev_st]["prob"] * trans_p[prev_st][st]
        
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = prev_st
                        
                max_prob = max_tr_prob * emit_p[t][st]
                V[t][st] = {'prob': max_prob, 'prev': prev_st_selected}
        
        # Now find the optimal Viterbi path by backtracking
        opt = []
        max_prob = 0.0
        previous = None
        # Get most probable state and its backtrack
        for st, data in V[-1].items():
            if data["prob"] > max_prob:
                max_prob = data["prob"]
                best_st_idx = st
                best_st = states[best_st_idx]

        opt.append(best_st)
        previous = best_st_idx
        
        # Follow the backtrack till the first observation
        for t in range(len(V) - 2, -1, -1):
            previous = V[t + 1][previous]["prev"]
            prev_state = states[previous]

            opt.insert(0, prev_state)

        return opt


if __name__ == '__main__':
    from abc_interpolation.datasets.human_gait.human_gait import HumanGaitFixedBox
    from abc_interpolation.datasets.human_gait.human_gait import HG_base
    from edflow.util import edprint

    HG = HG_base()
    edprint(HG.labels)

    kps = HG.labels['keypoints'][..., :2].astype('float32')
    print(kps)

    kp_hidden = kps[:int(0.84 * len(HG))]

    R = ReferenceSampler(kp_hidden, k=10)

    start = int(0.84 * len(HG))
    end = start + 10
    print(list(range(start, end)))
    q = kps[start: end]

    opt = R(q)

    print(opt)
