import torch
from edflow.iterators.template_iterator import TemplateIterator
from edflow.util import retrieve

from .triplet_loss import TripletLoss
from .utils import prepare_logs, prepare_input


class TransitionTrainer(TemplateIterator):
    """Trainer class to train the network prediction transition probabilities in latent space."""

    def __init__(self, config, root, model, datasets, **kwargs):
        super().__init__(config, root, model, datasets, **kwargs)
        self.model = model

        triplet_margin = config.get("triplet_margin", 0.3)
        self.triplet_loss = TripletLoss(triplet_margin)
        self.bce_loss = torch.nn.BCELoss()
        self.bs = config["batch_size"]

        self.lr = retrieve(config, "optimizer/lr", default=1e-4)
        beta_1 = retrieve(config, "optimizer/beta_1", default=0.5)
        beta_2 = retrieve(config, "optimizer/beta_2", default=0.99)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, betas=(beta_1, beta_2))

    def save(self, path):
        self.model.cpu()
        torch.save(self.model.state_dict(), path)
        if torch.cuda.is_available():
            self.model.cuda()

    def restore(self, checkpoint_path):
        self.model.cpu()
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        if torch.cuda.is_available():
            self.model.cuda()

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        if torch.cuda.is_available():
            self.model.cuda()

    def step_op(self, model, **input_dict) -> dict:
        """
        Main step function required by the edflow framework. Loads the data and model, and performs the forward pass.
        :param model: Edflow model class
        :param input_dict: Dictionary returned by the edflow training or validation dataset.
        :return: Returns a dictionary containing the train_op, eval_op and log_op functions.
        """
        scalars = dict()
        prepare_input(input_dict)
        z_anchor = input_dict["z_anchor"]
        z_positive = input_dict["z_positive"]
        z_negative_1 = input_dict["z_negative_1"]
        z_negative_2 = input_dict["z_negative_2"]

        pos_dis = input_dict["pos_dis"]
        neg_dis_1 = input_dict["neg_dis_1"]
        neg_dis_2 = input_dict["neg_dis_2"]
        # Setting up the scalar logs dictionary
        scalars["negative_distance_min"] = neg_dis_1[0]
        scalars["negative_distance_max"] = neg_dis_2[0]
        scalars["positive_distance"] = pos_dis[0]
        # Initialize Optimizer
        positive_ex = torch.cat((z_anchor, z_positive), dim=3).squeeze()
        neg_ex_1 = torch.cat((z_anchor, z_negative_1), dim=3).squeeze()
        neg_ex_2 = torch.cat((z_anchor, z_negative_2), dim=3).squeeze()

        positive_pass = model(positive_ex)
        negative_pass_1 = model(neg_ex_1)
        negative_pass_2 = model(neg_ex_2)
        if self.config.get("loss") == "triplet":
            loss = self.triplet_loss(
                anchor=torch.zeros_like(positive_ex), pos=positive_pass, neg=negative_pass_1
            ) + self.triplet_loss(
                anchor=torch.zeros_like(positive_ex), pos=positive_pass, neg=negative_pass_2
            )
        elif self.config.get("loss") == "bce":
            loss = self.bce_loss(positive_pass, torch.ones_like(positive_pass)) + 0.5 * (
                self.bce_loss(negative_pass_1, torch.ones_like(negative_pass_1))
                + self.bce_loss(negative_pass_2, torch.zeros_like(negative_pass_2))
            )
        else:
            raise ValueError("Unknown loss function.")
        scalars["loss"] = loss

        def train_op():
            """ Backward gradient prop only if training """
            scalars["lr"] = self.lr
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        def log_op():
            """ Return scalars needed to be logged"""
            prepare_logs(scalars)
            return {"scalars": scalars}

        def eval_op():
            """ Return nothing as of now"""
            return {}

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}
