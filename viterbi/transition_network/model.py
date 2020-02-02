import torch
import torch.nn as nn


class FCNet(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(FCNet, self).__init__()
        self.input_layer = nn.Linear(z_dim, hidden_dim)
        self.activation_layer = nn.ReLU()
        self.hidden_1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_2 = nn.Linear(hidden_dim, 1)
        self.output_layer = nn.Sigmoid()

    def forward(self, zs_concatenated):
        x = self.input_layer(zs_concatenated)
        x = self.activation_layer(x)
        x = self.hidden_1(x)
        x = self.activation_layer(x)
        x = self.hidden_1(x)
        x = self.activation_layer(x)
        x = self.hidden_2(x)
        x = self.output_layer(x)
        return x


class TransNet(torch.nn.Module):
    """
    Creates our models class for edflow which will compute the variance per label for motion calibration. This might also
    compute the expected variation between two different labels. Not sure though.
    """

    def __init__(self, config, **kwargs):
        """
        Initialising our models.
        """
        super().__init__()

        self.config = config
        self.latent_dimension = config.get("latent_dimension", 128)

        self.trans_net = FCNet(self.latent_dimension * 2, self.latent_dimension * 4)

    def forward(self, zs_concatenated):
        """
        Filter out the right module for the particular label and return the output.
        Parameters
        ----------
        zs_concatenated The pose encoding

        Returns
        -------
        The expected variance for that class.
        """
        return self.trans_net(zs_concatenated)
