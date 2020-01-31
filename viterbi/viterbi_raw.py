import copy


class Trellis:
    '''As taken from https://stackoverflow.com/a/9730066'''
    trell = []
    def __init__(self, hmm, observations):
        '''Viterbi algorithm

        Needs a hidden markov model (hmm) which know the emission probabilities
        :math:`p(x_{k}|z_{k})` and the transition probabilities
        :math:`p(z_{k}|z_{l})`, as well as the entire hidden state space.

        Arguments
        ---------
        hmm : object
            Hidden Markov model with the attributes
                - ``hmm.labels``: Names of all possible states.
                - ``hmm.emit(x, z)``: Emission probability of ``x`` given
                  ``z``. Arguments are given as names, e.g. not the actual
                  state but its index.
                - ``hmm.transition(z_i, z_j)``: Transition probability from
                  ``z_i`` to ``z_j``. Arguments are given as names, e.g. not
                  the actual state but its index.
        observations : list
            Names of the observed states. These must be interpretable by the
            :meth:`emit` method of the :attr:`hmm` object.
        '''
        # Tracker of observation, z_{1:k-1} pairs
        self.tracker = []

        # Initial assignments for hidden state at all timesteps are 0
        temp = {}
        for label in hmm.labels:
           temp[label] = [0,None]  # [probability, previous state] pairs
        for observation in observations:
            self.tracker.append([observation, copy.deepcopy(temp)])

        # Build the entire graph of state transitions
        self._fill_in(hmm)

    def _fill_in(self, hmm):
        '''Builds the entire graph of possible state transitions'''

        # Iterations over observed states
        for i in range(len(self.tracker)):

            # Iterate over hidden states
            for hidden_state in hmm.labels:

                observation = self.tracker[i][0]

                if i == 0:
                    p_xz = hmm.emit(hidden_state, observation, is_first=True)
                    self.tracker[i][1][hidden_state][0] = p_xz
                else:
                    max = None
                    guess = None
                    transition_prob = None

                    # Find max_{z_{k-1}} p(z_k|z_{k-1}) \mu
                    for other_state in hmm.labels:

                        p_zz = hmm.transition(hidden_state, other_state)
                        mu_zk_1 = self.tracker[i-1][1][other_state][0]

                        transition_prob = mu_zk_1 * p_zz

                        if max == None or transition_prob > max:
                            max = transition_prob
                            guess = other_state
                        print(max, guess)

                    p_xz = hmm.emit(hidden_state, observation)
                    max *= p_xz

                    self.tracker[i][1][hidden_state][0] = max
                    self.tracker[i][1][hidden_state][1] = guess

            for entry in self.tracker:
                print(entry)
            print('===')

    def return_max(self):
        hidden_states = []
        hidden = None
        for i in range(len(self.tracker)-1,-1,-1):
            if hidden == None:
                max = None
                guess = None
                for k in self.tracker[i][1]:
                    if max == None or self.tracker[i][1][k][0] > max:
                        max = self.tracker[i][1][k][0]
                        token = self.tracker[i][1][k][1]
                        guess = k
                hidden_states.append(guess)
            else:
                hidden_states.append(hidden)
                hidden = self.tracker[i][1][hidden][1]
        hidden_states.reverse()
        return hidden_states


def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}

    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = V[t-1][states[0]]["prob"]*trans_p[states[0]][st]
            prev_st_selected = states[0]
            for prev_st in states[1:]:
                tr_prob = V[t-1][prev_st]["prob"]*trans_p[prev_st][st]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st
                    
            max_prob = max_tr_prob * emit_p[st][obs[t]]
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected}
    
    opt = []
    max_prob = 0.0
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = st
    opt.append(best_st)
    previous = best_st
    
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    return opt, V
