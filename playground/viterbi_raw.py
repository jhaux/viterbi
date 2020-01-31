import numpy as np


class HF_MM:
    '''Some Hidden Markov model as shown in
    https://www.wikiwand.com/en/Viterbi_algorithm
    '''

    def __init__(self):
        '''
        '''

        self._t = { 'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
                   'Fever' : {'Healthy': 0.4, 'Fever': 0.6} }
        self._e = { 'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
                   'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6} }
        self._start_p = {'Healthy': 0.6, 'Fever': 0.4}

        self.labels = list(self._t.keys())

    def emit(self, i, j, is_first=False):
        if is_first:
            return self._e[i][j] * self._start_p[i]
        return self._e[i][j]

    def transition(self, i, j):
        return self._t[i][j]


class HF_MM_m:
    '''Some Hidden Markov model as shown in
    https://www.wikiwand.com/en/Viterbi_algorithm
    '''

    def __init__(self):
        '''
        '''

        self._t = np.array([[0.7, 0.3], [0.4, 0.6]])
        self._e = np.array([[0.5, 0.4, 0.1], [0.1, 0.4, 0.6]])
        self._start_p = np.array([0.6, 0.4])

        self.labels = np.arange(len(self._t))

    def emit(self, i, j, is_first=False):
        if is_first:
            return self._e[i][j] * self._start_p[i]
        return self._e[i][j]

    def transition(self, i, j):
        return self._t[i][j]


if __name__ == '__main__':
    from viterbi.viterbi_raw import Trellis, viterbi

    hmm = HF_MM()

    o = ['normal', 'cold', 'dizzy']

    T = Trellis(hmm, o)

    max_seq = T.return_max()
    print(max_seq)

    opt, V = viterbi(o, hmm.labels, hmm._start_p, hmm._t, hmm._e)
    for v in V:
        print(v)

    print('='*10)

    hmm = HF_MM_m()
    o = np.arange(3)

    T = Trellis(hmm, o)

    max_seq = T.return_max()

    print(max_seq)
