import torch
import yaml
from edflow.main import get_implementations_from_config
from viterbi.transition_network import TransNet


def load_model(checkpoint_path, config):
    pretrained_state_dict = torch.load(checkpoint_path)
    model = TransNet(config)
    model.load_state_dict(pretrained_state_dict)
    model.cuda()
    return model


def load_dataset(config):
    dataset = get_implementations_from_config({"dataset": "transition_network.encodings_dataset.encodings"}, ["dataset"])[
        "dataset"
    ](config)
    return dataset


def test_trans_net():
    with open("/export/home/rmarwaha/projects/viterbi/transition_network/training_config.yaml", "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    checkpoint_path = "/export/home/rmarwaha/projects/logs/2020-01-27T16-43-42_transistion_net_adapt_neg/train/checkpoints/29-72030_transition_net.ckpt"
    model = load_model(checkpoint_path, config)
    dataset = load_dataset(config)
    for i in range(0,1000, 20):







