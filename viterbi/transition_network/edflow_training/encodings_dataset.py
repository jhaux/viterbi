import numpy as np
from edflow.custom_logging import get_logger
from edflow.data.dataset import DatasetMixin, PRNGMixin

from viterbi.transition_network.utilities.old_pipeline import EvalDataFolder

logger = get_logger(__name__)


def encodings_data(train=True):
    class Dataset(DatasetMixin, PRNGMixin):
        """
        The class is supposed to output the label, an pose encoding, and a certain variance we would expect given
        the encoding is from the same label.
        The variance is currently calculated by subtracting the nearest neighbours based on keypoints excluding the
        encodings from time-wise immediately after or before the frame.
        """

        def __init__(self, config: dict):
            self.config = config
            self.latent_dimension = self.config.get("latent_dimension", 128)
            self.data_folder = self.config.get("data_folder")
            self.step = 0

            pose_dataset = EvalDataFolder(root=self.data_folder + "/0/model_output.csv")
            if train:
                self.pose_encodings = pose_dataset.labels["ose"][:-10000]
            else:
                self.pose_encodings = pose_dataset.labels["ose"][-10000:]

            logger.info(f"Shape of the pose encodings: {self.pose_encodings.shape}")
            self.number_of_encodings = len(self.pose_encodings)

        def get_neg_idx(self, step, tau):
            next_neg_idx = 4 * tau
            if step % 5000 == 0 and next_neg_idx > 2 * tau:
                next_neg_idx -= tau
            self.step += 1
            return next_neg_idx

        def get_example(self, idx: int):
            """
            Filters the nearest neighbours for self and immediate neighbours, then gets the next nearest neighbour to
            calculate the variance. Returns that variance, the particular pose encoding, and the label we did this for.
            Parameters
            ----------
            idx Index

            Returns
            -------
            Dictionary containing the label, Variance, and beta i.e. the pose encoding.
            """
            tau = 4
            z_anchor = self.pose_encodings[idx]
            next_pos_idx = int(round(np.random.normal(tau, 0.4), 0))
            z_positive = self.pose_encodings[idx + next_pos_idx]

            next_neg_idx_1 = int(round(np.random.normal(0, 0.4), 0))
            next_neg_idx_2 = self.get_neg_idx(self.step, tau)
            z_negative_1 = self.pose_encodings[abs(idx + next_neg_idx_1)]
            z_negative_2 = self.pose_encodings[idx + next_neg_idx_2]
            output = dict()
            output["z_anchor"] = z_anchor
            output["z_positive"] = z_positive
            output["z_negative_1"] = z_negative_1
            output["z_negative_2"] = z_negative_2
            output["pos_dis"] = next_pos_idx
            output["neg_dis_1"] = next_neg_idx_1
            output["neg_dis_2"] = next_neg_idx_2

            return output

        def __len__(self):
            # Keeping in mind that the hard negative is samples from the next ~20 frames
            return self.number_of_encodings - 25

    return Dataset


def encodings_evaluation_data():
    class Dataset(DatasetMixin, PRNGMixin):
        """
        """

        def __init__(self, config: dict):
            self.config = config
            self.latent_dimension = self.config.get("latent_dimension", 128)
            self.data_folder = self.config.get("data_folder")
            self.pattern = self.config.get("pattern", "pose*")

            pose_dataset = EvalDataFolder(root=self.data_folder + "/0/model_output.csv")
            self.pose_encodings = pose_dataset.labels["ose"][-10000:]

            logger.info(f"Shape of the pose encodings: {self.pose_encodings.shape}")
            self.number_of_encodings = len(self.pose_encodings)

        def get_example(self, idx: int):
            """
            """
            z_anchor = self.pose_encodings[idx]
            z_concatenated = np.concatenate(
                (
                    np.tile(z_anchor, self.number_of_encodings).reshape(
                        self.number_of_encodings, 1, 1, 128
                    ),
                    self.pose_encodings,
                ),
                axis=-1,
            )
            output = dict()
            output["z_concatenated"] = z_concatenated

            return output

        def __len__(self):
            # Keeping in mind that the hard negative is samples from the next ~20 frames
            return self.number_of_encodings - 25

    return Dataset


training_encodings = encodings_data(train=True)
validation_encodings = encodings_data(train=False)
