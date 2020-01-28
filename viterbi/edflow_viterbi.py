import numpy as np
from edflow.data.dataset import DatasetMixin, PRNGMixin
from edflow.data.dataset import SequenceDataset
from tqdm import tqdm

from viterbi import Viterbi


class ViterbiNeighbours(DatasetMixin):
    def __init__(
        self,
        base_dataset,
        nn_key="keypoints",
        same_keys=[],
        diff_keys=[],
        blanket_size=1,
    ):
        self.base_dataset = base_dataset
        self.labels = self.base_dataset.labels

        self.states = base_dataset.labels[nn_key]

        self.dataset_length = len(base_dataset)
        self.same_labels = [base_dataset.labels[k] for k in same_keys]
        self.diff_labels = [base_dataset.labels[k] for k in diff_keys]

        self.same_masks = self._calc_same_masks(self.same_labels)
        self.diff_masks = self._calc_diff_masks(self.diff_labels)

        self.blanket_size = blanket_size

    @staticmethod
    def _calc_same_masks(same_labels):
        same_masks = []
        for same_label in tqdm(same_labels, desc="Same"):
            possible_same_labels = np.unique(same_label)
            label_masks = {}
            for label in tqdm(possible_same_labels, desc="Label"):
                mask = np.array(same_label) == label
                label_masks[label] = mask  # [N]

            same_masks += [label_masks]
        return same_masks

    @staticmethod
    def _calc_diff_masks(diff_labels):
        diff_masks = []
        for diff_label in tqdm(diff_labels, desc="Diff"):  # [e.g. 1]
            possible_diff_labels = np.unique(diff_label)  # [e.g 3]
            label_masks = {}
            for label in tqdm(possible_diff_labels, desc="Label"):  # [e.g. 3]
                mask = np.array(diff_label) != label
                label_masks[label] = mask  # [N]

            diff_masks += [label_masks]  # [e.g. 1, 3, N]
        return diff_masks

    def _label_mask(self, idx):
        same_keys = [sl[idx] for sl in self.same_labels]
        diff_keys = [dl[idx] for dl in self.diff_labels]

        same_mask = np.ones([self.dataset_length], dtype=bool)
        for s, k in zip(self.same_masks, same_keys):
            same_mask = np.logical_and(same_mask, s[k])

        diff_mask = np.ones([self.dataset_length], dtype=bool)
        for s, k in zip(self.diff_masks, diff_keys):
            diff_mask = np.logical_and(diff_mask, s[k])

        label_mask = np.logical_and(same_mask, diff_mask)

        if not np.any(label_mask):
            raise ValueError(
                "The combination of same and diff labels does "
                "not allow for any neighbour samples: \n"
                "const: {}\nignore:{}".format(same_keys, diff_keys)
            )

        return label_mask

    def __len__(self):
        return self.dataset_length

    def get_example(self, idx):
        valid_indices = np.arange(self.dataset_length)[self._label_mask(idx)]
        # calculation of distance taking too long on all the states_
        valid_indices_choice = np.random.choice(valid_indices, size=1000)
        valid_states = self.states[valid_indices_choice][:, 0, ...]
        observation = self.states[idx]
        hidden_states = Viterbi(valid_states)(observation, self.blanket_size)
        example = self.base_dataset[idx]
        example["neighbours"] = self.base_dataset[hidden_states[:, 0].astype(int)]
        return example


def load_im_hack(path, size=[256, 256]):
    image = Image.open(path)
    image = image.resize(size)
    image = np.array(image)
    image = adjust_support(image, '-1->1', '0->255')

    return image

def make_abc_nn_seq_mag_dset(base_dset, diff=["healthy"], same=[]):
    """Sequence/Context NN sampling returning sequences/or single frames"""

    class ABC_Seq_Mag_Dset(DatasetMixin, PRNGMixin):
        """Samples a set of neighbours to a given frame and supplies these as
        references for magnifications.
        """

        def __init__(self, config):
            """
            Parameters
            ----------
            config : dict
                Containing the keys ``n_ref`` for the number of reference
                neighbours to be supplied, ``magnification_factors`` for the
                number of magnifications to be applied. There are various
                strategies ``nn_strategy``, how nearest neighbours are sampled as well:
                ``all`` considers all available neighbours and samples
                ``n_ref`` references uniformly from them. ``up_to`` considers
                all ``nn_consider`` closest neighbours and samples from them,
                while ``from`` considers the rest.
            """

            self.mags = config.setdefault("magnification_factors", [1, 2, 3])

            self.sequence_length = sl = config.get("sequence_length", 8)
            self.late_loading = config.get("late_loading")
            self.config = config

            self.im_shape = config.setdefault("spatial_size", [256, 256])
            if isinstance(self.im_shape, int):
                self.im_shape = [self.im_shape] * 2

            self.frames = base_dset(config)
            self.seqs = SequenceDataset(self.frames, sl, strategy="reset")

            self.seqs.labels["pid_s"] = self.seqs.labels["pid"][:, 0]
            self.seqs.labels["healthy_s"] = self.seqs.labels["healthy"][:, 0]
            if all(key in self.seqs.labels.keys() for key in same):
                for same_key in same:
                    self.seqs.labels[same_key + "_s"] = self.seqs.labels[same_key][:, 0]
                same_keys = [s + "_s" for s in same]
            else:
                same_keys = []

            diff_keys = [d + "_s" for d in diff]

            self.viterbi_neighbours = ViterbiNeighbours(
                self.seqs, nn_key="keypoints", same_keys=same_keys, diff_keys=diff_keys
            )

            self.labels = self.viterbi_neighbours.labels

        def get_appearance_image(self, pid):
            """Samples a random image of the person with the given id.

            Returns:
                numpy array: the sampled image
                int: its index
            """
            pid_labels = np.array(self.frames.labels["pid"])
            possible_indices = np.where(pid_labels == pid)[0]

            appearance_idx = self.prng.choice(possible_indices)

            if not self.late_loading:
                appearance_image = self.frames[appearance_idx]["target"]
            else:
                appearance_image = self.frames[appearance_idx]["target_path"]
                appearance_image = load_im_hack(appearance_image, self.im_shape)
            appearance_pid = self.frames.labels["pid"][appearance_idx]

            return appearance_image, appearance_idx, appearance_pid

        def get_example(self, idx):
            ex = self.viterbi_neighbours[idx]

            pid = ex["pid_s"]

            app_im, app_idx, app_pid = self.get_appearance_image(pid)

            ex["appearance"] = app_im
            ex["appearance_idx"] = app_idx
            ex["appearance_pid"] = app_pid

            if not self.late_loading:
                if self.rets == "sequence":
                    ex["keypoints_query"] = ex["keypoints"]
                    ex["image_query"] = ex["target"]

                    ex["keypoints_reference"] = [
                        e["keypoints"] for e in ex["neighbours"]
                    ]
                    ex["image_reference"] = [e["target"] for e in ex["neighbours"]]
                else:
                    ex["keypoints_query"] = ex["keypoints"][self.c]
                    ex["image_query"] = ex["target"][self.c]

                    ex["keypoints_reference"] = [
                        e["keypoints"][self.c] for e in ex["neighbours"]
                    ]
                    ex["image_reference"] = [
                        e["target"][self.c] for e in ex["neighbours"]
                    ]

            ex["magnification_factor"] = self.mags

            ex["frame_anchor_1"] = ex["image_reference"][0]
            ex["frame_anchor_2"] = ex["image_query"]

            return ex

        def __len__(self):
            return len(self.viterbi_neighbours)

    return ABC_Seq_Mag_Dset


if __name__ == "__main__":
    from abc_interpolation.data.human_gait import HumanGaitFixedBox

    dataset = HumanGaitFixedBox

    hg_viterbi = make_abc_nn_seq_mag_dset(dataset)
    data = hg_viterbi({"data_split": "train", "mode": "eval"})
    from edflow.util import pp2mkdtable

    pp2mkdtable(data.get_example(10))
