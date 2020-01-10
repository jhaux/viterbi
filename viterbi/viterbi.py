from typing import Callable
import numpy as np


def poseL2(pose1, pose2):
    '''Compute the mean keypoint distance of two poses'''

    diff = pose1 - pose2
    kp_mag = (diff ** 2).sum(-1)
    mean_dist = kp_mag.mean(-1)

    return mean_dist


class Viterbi(object):
    '''Implements a pose sequence Viterbi algorithm, which does not need to
    know the whole transition matric, but computes it on the fly for single
    examples. This is due to the fact, that poses are represented as keypoints
    in contiuos space, thus showing a huge amount of variance, resulting in an
    enormus state space, even when only looking at discrete pose samples.

    The transistion probabilities are always calculated based on the distances
    between two states, i.e. two pose representations.
    '''

    def __init__(self,
                 states: list,
                 transistion_fn: Callable=poseL2):
        '''
        Arguments
        ---------
        states : list
            A list of all possible hidden states, which can be visited, e.g. a
            set of poses from a certain class like `health/uhealthy`.
        transistion_fn : Callable
            A function, which given two pose representations computes their
            distance.
        '''

        self._states = np.array(states)

        self.distance = transistion_fn

        self._state_transitions = self.distance(self._states[None], self._states[:, None])


    def __call__(self, observations: list, blanket_size: int=5, direct_dist_weight=3.) -> list:
        '''
        Finds the sequences of hidden states, which are the most likely to fit
        the observations.

        Arguments
        ---------
        observations : list
            A sequence of pose vectors, which we want to match to the possible
            states.
        blanket_size : int
            How many elements of the hidden sequence should be considered when
            calculating the transitiona probabilities.
        direct_dist_weight : float
            Weight factor weighing the blanket distances against the
            observation - state distance.

        Returns
        -------
        hidden_states : list
            The sequence of hidden states which minimizes the distance of each
            state to its observation plus the distance to its ``blanket_size``
            neighbors.
        '''

        # First pass get all possible distance

        obs = np.array(observations)[:, None]  # [M, 1, 17, 2]

        states = self._states[None]   # [1, N, 17, 2]

        distances = poseL2(obs, states)  # [M, N]
        print(distances)

        state_distances = self._state_transitions  # [N, N]
        print(state_distances)

        # Second step: Get combined distance
        hidden_states = []
        for t in range(len(obs)):
            nn_dists = [direct_dist_weight * distances[t]]
            for i in range(1, min(t, blanket_size)+1):
                prev_idx = hidden_states[t - i][0]
                dist_prev_state_current_state = state_distances[prev_idx]
                nn_dists += [dist_prev_state_current_state]

            combined_dists = np.array(nn_dists).sum(0)
            print(combined_dists.shape)

            new_state_idx = np.argmin(combined_dists)
            new_state = [new_state_idx, combined_dists[new_state_idx]]

            hidden_states += [new_state]

        return hidden_states


if __name__ == '__main__':
    import os
    from sklearn import datasets
    import matplotlib.pyplot as plt


    n_samples = 500
    noise = 0.05
    X, y = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=42)

    X = X[:, None]

    states = X[y == 1]
    obs_ = X[y == 0]

    obs = obs_[np.random.choice(np.arange(len(obs_)), size=10, replace=False)]
    sort_idxs = np.argsort(obs[..., 0, 0])
    obs = obs[sort_idxs]

    co = obs_[..., 0, 0] * 2 + obs_[..., 0, 1]
    cs = states[..., 0, 0] * 2 + states[..., 0, 1]

    V = Viterbi(states)

    hidden_states = V(obs, 1)
    idxs = [s[0] for s in hidden_states]

    outs = np.array([states[i] for i in idxs])

    f, ax = plt.subplots(1, 1, figsize=[0.5*12.8, 0.5*7.2], dpi=300, constrained_layout=True)

    ax.scatter(obs_[..., 0, 0], obs_[..., 0, 1], alpha=0.1, label='all observations', c=co, ec='k')
    ax.scatter(obs[..., 0, 0], obs[..., 0, 1], c=np.arange(len(obs)), label='observation', ec='k')
    ax.scatter(states[..., 0, 0], states[..., 0, 1], alpha=0.1, label='all states', c=cs, ec='k')
    ax.scatter(outs[..., 0, 0], outs[..., 0, 1], c=np.arange(len(outs)), marker='s', label='matched states', ec='k')

    ax.legend()
    ax.set_title('NN sequence found using pose viterbi')

    sp = os.path.join(os.path.dirname(__file__), 'pose_viterbi_on_moon.png')
    f.savefig(sp)
