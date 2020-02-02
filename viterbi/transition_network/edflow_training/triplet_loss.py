import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """
    Compute normal triplet loss or soft margin triplet loss given triplets
    """
    def __init__(self, margin: float = 0.5):
        """
        Initializes the triplet loss accustomed to this project.
        :param margin: Margin for the triplet loss.
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        """
        For our case, we are interchanging the positive and negative as the positive successor should intuitively
        have the higher probability. As we're working with distance in contrast to points or vectors specifically, a
        zero array will be mostly passed as an anchor.
        :param anchor The anchor which it is compared to.
        :param pos The positive example.
        :param neg The negative example
        """
        loss = self.Loss(anchor, neg, pos)

        return loss
