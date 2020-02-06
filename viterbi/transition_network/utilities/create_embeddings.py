"""
A script for generating transition probabilities between two given keypoints given an interpolation model and a
network which computes the probability of the transition.
"""
import numpy as np
import torch
import torch.nn as nn
from abc_interpolation.models.basic_model import BasicNet
from abc_pose.abcnet.heatmaps_pytorch import kp2heat

from viterbi.transition_network.edflow_training.model import TransNet


def compute_transition_probability(
    keypoints_1: np.ndarray,
    keypoints_2: np.ndarray,
    interpolation_model: nn.Module,
    transition_model: nn.Module,
):
    """
    Given an interpolation and a transition model, computes the transition probability.
    :param keypoints_1: 2-dimensional np.ndarray i.e. starting pose.
    :param keypoints_2: The pose to which the transition ends.
    :param interpolation_model: A trained model with the method `hm_head` which creates an embedding of the pose.
    :param transition_model: A trained model for computing the transition probability between two embeddings.
    :return: The transition probability as a torch.Tensor.
    """
    pose_embedding_1 = create_embedding(keypoints_1, interpolation_model)
    pose_embedding_2 = create_embedding(keypoints_2, interpolation_model)

    if len(pose_embedding_1.shape) == 1:
        pose_embedding_1.unsqueeze(0)
    if len(pose_embedding_2.shape) == 1:
        pose_embedding_2.unsqueeze(0)

    return transition_model(torch.cat((pose_embedding_1, pose_embedding_2), dim=1))


def create_embedding(keypoints: np.ndarray, model: nn.Module) -> torch.Tensor:
    """
    Return the pose embedding given the keypoints and the model.
    :param keypoints: 2-dimensional np.ndarray.
    :param model: A trained model with the method `hm_head` which creates an embedding of the pose.
    :return: The pose embedding as a torch.Tensor.
    """
    keypoints_tensor = torch.tensor(keypoints)
    if torch.cuda.is_available:
        keypoints_tensor = keypoints_tensor.cuda()
    heatmap = kp2heat(keypoints_tensor).float()
    if len(heatmap.shape) == 3:
        heatmap = heatmap.unsqueeze(0)
    return model.hm_head(heatmap).squeeze()


def load_interpolation_model(model_path: str) -> nn.Module:
    """
    Load the interpolation model with the given checkpoint.
    :param model_path: Path to the checkpoint file.
    :return: Model with the trained weights.
    """
    config = {"heatmap_input_channels": 6, "input_channels": 3, 'add_out_channels': 1}
    model = BasicNet(config)
    initialize_model(model, model_path)
    return model


def load_transition_model(model_path: str) -> nn.Module:
    """
    Load the transition probability model with the given checkpoint.
    :param model_path: Path to the checkpoint file.
    :return: Model with the trained weights.
    """
    config = {}
    model = TransNet(config)
    initialize_model(model, model_path)
    return model


def initialize_model(model: nn.Module, model_path: str) -> nn.Module:
    """
    Initializes the given `model` with checkpoints found in `model_path`.
    :param model: The naked model.
    :param model_path: Path to the checkpoint file.
    :return: Model with the trained weights.
    """
    model.cpu()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    if torch.cuda.is_available():
        model.cuda()


class TransProb:
    def __init__(self):
        interpolation_checkpoint_path = "/export/data/rmarwaha/projects/logs/2019-11-20T14-46-23_hg_disc/train/checkpoints/1498-100005_abc_net.ckpt"
        transition_checkpoint_path = '/export/home/rmarwaha/projects/viterbi/logs/2020-02-03T14-08-26_transistion_network_bce/train/checkpoints/model-71833.ckpt'
        # transition_checkpoint_path = "/export/home/rmarwaha/projects/logs/2020-01-31T10-45-03_transistion_network_triplet/train/checkpoints/model-71833.ckpt"

        self.interpolation_model = load_interpolation_model(interpolation_checkpoint_path)
        self.transition_model = load_transition_model(transition_checkpoint_path)

    def __call__(self, kp1, kp2):
        return compute_transition_probability(
                kp1, kp2,
                self.interpolation_model, self.transition_model
                )


if __name__ == '__main__':
    interpolation_checkpoint_path = "/export/data/rmarwaha/projects/logs/2019-11-20T14-46-23_hg_disc/train/checkpoints/1498-100005_abc_net.ckpt"

    transition_checkpoint_path = '/export/home/rmarwaha/projects/viterbi/logs/2020-02-03T14-08-26_transistion_network_bce/train/checkpoints/model-71833.ckpt'
    # transition_checkpoint_path = "/export/home/rmarwaha/projects/logs/2020-01-31T10-45-03_transistion_network_triplet/train/checkpoints/model-71833.ckpt"
    interpolation_model = load_interpolation_model(interpolation_checkpoint_path)
    transition_model = load_transition_model(transition_checkpoint_path)
    from abc_interpolation.datasets.human_gait.human_gait import HumanGait_abc
    hg_data = HumanGait_abc({})
    idx = int(np.random.rand(1) * len(hg_data))

    kp_1 = hg_data[idx]["keypoints_anchor_1"]
    kp_2 = hg_data[idx]["keypoints_anchor_2"]
    trans_prob = compute_transition_probability(kp_1, kp_2, interpolation_model, transition_model)
    print(trans_prob)

    TP = TransProb()
    print(TP(kp_1, kp_2))
