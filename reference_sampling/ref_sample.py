import torch.nn as nn
import faiss

from edflow.custom_logging import get_logger


class ReferenceSampler(nn.Module):
    '''Using Viterbi as a backend'''

    def __init__(self, targets, transition_probability):
        '''
        Arguments
        ---------
        targets : torch.Tensor
            All ``M`` possible reference pose encodings. Shape ``[M, Z]``
        transition_probability : Callable
            A function which takes a batch of pairs of pose encodings and
            returns the transition probability from the first to the second.
        '''

        super().__init__()
        self.logger = get_logger(self)

        self.targets = targets

        self.nn_sampler = faiss.IndexFlatL2(targets.shape[-1])
        self.nn_sampler.add(self.targets)
        self.k = 100

        n = self.nn_sampler.ntotal
        nt = '' if self.nn_sampler.is_trained else ' not'
        
        self.logger.info(f'NN sampler contains {n} examples and is{nt} trained')

        self.transition_probability = transition_probability

    def emission_probabilities_and_states(self, query_sequence):
        distances, states = self.nn_sampler.search(
            query_sequence,  # [T, Z]
            self.k
        )  # [T, K]

        # Gibbs distribution
        probabilites = np.exp(-distances) / np.exp(-distances).sum(1)[..., None]
        # [T, K]

        states = self.targets[states]

        return probabilities, states

    def get_transition_probability(self, states, ignore_step):
        '''
        Arguments
        ---------
        states : k*T states
            All possible states
        ingnore_step : int
            transitions are only possible from t to t+1 and only between every
            pair of following sets of ``ignore_step`` states.
        '''

        trans_p = np.zeros([self.T*self.k, self.T, self.k])

        def is_possible(idx1, idx2):
            if idx2 - idx1 < self.k and idx2 - idx1 > 0:
                return True
            return False

        for idx1 in range(self.k*self.T):
            for idx2 in range(self.k*self.T):
                if is_possible(idx1, idx2, self.k):
                    p = self.transition_probability(self.states[idx1],
                                                    self.states[idx2])
                    trans_p[idx1, idx2] = p

        return trans_p

    def forward(self, query_batch):
        '''
        Arguments
        ---------
        query_batch : torch.Tensor
            A sequence of pose encodings. Shape ``[B, T, Z]``

        Returns
        -------
        reference : torch.Tensor
            A reference sequence of pose encodings. Shape is the same as
            :attr:`query`.
        '''

        references = []

        # We have a batch of queries(?)
        for i, query in enumerate(query_batch):  # [T, Z]
            emit_p, states = self.emission_probabilities(query)  # [T, K], [T, K, Z]
            trans_p = self.get_transition_probability(states, self.k)

            V = [{}]

            # Initialization: The probability for being in one of the possible
            # states is given by the emission probabilities.
            for st in range(len(trans_p)):
                V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
            
            # Run Viterbi when t > 0
            for t in range(1, len(query)):
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
                        tr_prob = V[t-1][prev_state]["prob"] * trans_p[prev_st][st]
            
                        if tr_prob > max_tr_prob:
                            max_tr_prob = tr_prob
                            prev_st_selected = prev_st
                            
                    max_prob = max_tr_prob * emit_p[t][st]
                    V[t][st]["prob"] = max_prob
                    V[t][st]["prev"] = prev_st_selected
            
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
                prev_state = states(previous)

                opt.insert(prev_state)

            return opt
