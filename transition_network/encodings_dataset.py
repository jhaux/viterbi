"""
The dataset should return a sequence of latent pose embeddings created using abc_interpolation for one certain label -
for instance, `healthy`, `Johannes`, etc. Goal is to train a models which outputs a certain shift as a vector. This shift
is considered the ordinary behaviour for that particular label.
The residuals of the vectors are then of interest, as these will incorporate behaviour shift which are NOT ordinary.
"""

import numpy as np
from abc_interpolation.motion_calibration.utils import extract_data
from edflow.custom_logging import get_logger
from edflow.data.dataset import DatasetMixin, PRNGMixin
from edflow.util import linear_var

logger = get_logger(__name__)


def encodings_data():
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
            self.pattern = self.config.get("pattern", "pose*")
            extract_until = "all"

            self.step = 0

            pose_encodings = extract_data(
                self.data_folder,
                pattern=self.pattern.rstrip("*") + "*",
                until=extract_until,
            )
            self.pose_encodings = pose_encodings.squeeze(0)

            logger.info(f"Shape of the pose encodings: {self.pose_encodings.shape}")
            self.number_of_encodings = len(self.pose_encodings)

        def get_neg_idx(self, step):
            start_x, start_y = (0, 23)
            stop_x, stop_y = (90000, 7)
            neg_idx_mean = linear_var(step, start_x, stop_x, start_y, stop_y)
            self.step += 1
            return int(np.random.normal(neg_idx_mean, 1))

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
            z_anchor = self.pose_encodings[idx]
            next_pos_iter = int(np.random.normal(3, 1))
            z_positive = self.pose_encodings[idx + next_pos_iter]
            next_neg_iter = self.get_neg_idx(self.step)
            z_negative = self.pose_encodings[next_neg_iter]
            output = dict()
            output["z_anchor"] = z_anchor
            output["z_positive"] = z_positive
            output["z_negative"] = z_negative

            return output

        def __len__(self):
            # Keeping in mind that the hard negative is samples from the next ~20 frames
            return self.number_of_encodings - 25

    return Dataset


encodings = encodings_data()

if __name__ == "__main__":
    from timeit import timeit

    timeit(
        pose_encodings=extract_data(
            "/export/data/rmarwaha/projects/logs/2019-11-20T14-46-23_hg_disc/eval/2020-01-24T14-52-25_pose_enc",
            pattern="pose*",
            until="all",
        )
    )
