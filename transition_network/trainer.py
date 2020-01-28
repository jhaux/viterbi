import numpy as np
import torch
from edflow.custom_logging import ProjectManager
from edflow.eval.pipeline import EvalHook
from edflow.hooks.pytorch_hooks import PyCheckpointHook
from edflow.hooks.pytorch_hooks import PyLoggingHook
from edflow.hooks.pytorch_hooks import ToFromTorchHook
from edflow.hooks.util_hooks import IntervalHook
from edflow.iterators.model_iterator import PyHookedModelIterator
from edflow.util import walk

from .triplet_loss import TripletLoss


class TransitionTrainer(PyHookedModelIterator):
    """Can train and eval :class:`ABC_Net`."""

    def __init__(self, config, root, model, **kwargs):
        super().__init__(config, root, model, **kwargs)

        self.config = config
        self.root = root
        self.model = model

        # Mode. What do you want to do?
        self.mode = config.get("mode", "train")

        self.tb_log_freq = config.get("tb_log_freq", 100)
        self.log_freq = config.get("log_freq", 1000)

        triplet_margin = config.get("triplet_margin", 0.4)
        self.triplet_loss = TripletLoss(triplet_margin)
        self.bs = config["batch_size"]

        # Training stuff
        if self.mode == "train":
            self.lr = lr = config.get("lr", 1e-6)
            beta_1 = config.get("beta_1", 0.5)
            beta_2 = config.get("beta_2", 0.9)

            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, betas=(beta_1, beta_2)
            )

            # Initialize losses
            self.loss_weights = config.get(
                "loss_weights",
                {"kl": 0.01, "rec": 5, "reg": 0.1, "fp": 0.01, "app_enc": 0.0},
            )

        DPrepH = DataPrepHook()

        # Logging scalars. Check if they are also in log_dict.
        scalars = [
            "step_ops/losses/triplet_loss",
        ]

        def get_inputs(
            model,
            z_anchor,
            z_positive,
            z_negative,
            *args,
            **kwargs,
        ):
            input_dict = {
                "inputs": {
                    "z_anchor": z_anchor,
                    "z_positive": z_positive,
                    "z_negative": z_negative,
                }
            }
            return input_dict

        loghook = PyLoggingHook(
            log_ops=[get_inputs],
            scalar_keys=scalars,
            root_path=ProjectManager.train,
            interval=self.tb_log_freq,
        )

        ckpt_hook = PyCheckpointHook(
            root_path=ProjectManager.checkpoints,
            model=model,
            modelname="transition_net",
            interval=self.log_freq,
        )

        interval = IntervalHook(
            [loghook, ckpt_hook],
            1,
            modify_each=1,
            modifier=lambda x: x,
            max_interval=1000,
            get_step=self.get_global_step,
        )

        if self.mode == "train":
            self.hooks += [DPrepH, ckpt_hook, interval]  # ,tb_interval]
        elif self.mode == "eval" or self.mode == "test":
            from edflow.main import get_implementations_from_config

            eval_dataset = get_implementations_from_config(self.config, ["dataset"])[
                "dataset"
            ](config)

            labels_key = "step_ops/labels"
            eval_hook = EvalHook(
                dataset=eval_dataset,
                labels_key=labels_key,
                callbacks={},
                config=config,
                step_getter=self.get_global_step,
            )

            self.hooks += [DPrepH, eval_hook]

    def initialize(self, checkpoint_path: str = None, **kwargs):
        if checkpoint_path is not None:
            pretrained_state_dict = torch.load(checkpoint_path)
            self.model.load_state_dict(pretrained_state_dict)
            self.logger.info(f"Found checkpoint at {checkpoint_path}.")
        self.model.cuda()

    def train_op(
        self,
        model,
        z_anchor,
        z_positive,
        z_negative,
        **kwargs,
    ):
        # Setting up the scalar logs dictionary
        loss_dict = {
            "triplet_loss": [],
        }
        log_dict = {"lr": self.lr}

        # Initialize Optimizer
        self.optimizer.zero_grad()
        positive_ex = torch.cat((z_anchor, z_positive), dim=1).squeeze(2).squeeze(2)
        neg_ex = torch.cat((z_anchor, z_negative), dim=1).squeeze(2).squeeze(2)
        positive_pass = model(positive_ex)
        negative_pass = model(neg_ex)

        triplet_loss = self.triplet_loss(anchor=torch.zeros_like(positive_ex),
                                         pos=positive_pass,
                                         neg=negative_pass)
        if self.get_global_step() % 1000 == 0:
            print("LOSS:------", triplet_loss)
            print("POS:-------", positive_pass)
            print("NEG:-------", negative_pass)
        loss_dict["triplet_loss"] += [triplet_loss]

        triplet_loss.backward()
        self.optimizer.step()

        for k, v in loss_dict.items():
            if len(v) == 1:
                loss_dict[k] = v[0]
            else:
                t_loss = 0
                for t in v:
                    t_loss += t
                loss_dict[k] = t_loss

        return {"losses": loss_dict, "other": log_dict}

    def eval_op(
        self,
        model,
        z_anchor,
        z_positive,
        z_negative,
        **kwargs,
    ):
        """
        :param model:
        :param z_anchor:
        :param z_positive:
        :param z_negative:
        :param kwargs:
        :return:
        """
        return {"results": []}

    def step_ops(self):
        if self.config.get("test_mode", False):
            return self.eval_op
        else:
            return self.train_op


class DataPrepHook(ToFromTorchHook):
    """Do not only convert to and from torch tensor, but also make sure
    image dimensions are correct.
    """

    def before_step(self, step, fetches, feeds, batch):
        """ Before converting: transpose to ``[B, C, H, W]``."""

        def to_image(obj):
            if isinstance(obj, np.ndarray) and len(obj.shape) == 4:
                return obj.transpose(0, 3, 1, 2)
            else:
                return obj

        walk(feeds, to_image, inplace=True)

        super().before_step(step, fetches, feeds, batch)

    def after_step(self, step, results):
        """After converting: transpose to ``[B, H, W, C]``."""

        super().after_step(step, results)

        def to_image(k, obj):
            if (
                "weights" not in k
                and isinstance(obj, np.ndarray)
                and len(obj.shape) == 4
            ):
                return obj.transpose(0, 2, 3, 1)
            else:
                return obj

        walk(results, to_image, inplace=True, pass_key=True)
