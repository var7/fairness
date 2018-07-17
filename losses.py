import torch
import torch.nn as nn
import torch.nn.functional as F

class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    Modified nn.functional.triplet_margin_loss
    """
    def __init__(self, triplet_selector, margin=1.0, p=2):
        super(OnlineTripletLoss, self).__init__()
        # self.margin = margin
        self.triplet_selector = triplet_selector
        # self.p = p
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=p)

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        anchor = embeddings[triplets[:, 0]]
        positive = embeddings[triplets[:, 1]]
        negative = embeddings[triplets[:, 2]]
        # print(anchor, positive, negative)
        return self.loss_fn(anchor, positive, negative)
        # return nn.TripletMarginLoss(anchor, positive, negative, margin=self.margin, p=self.p)


    # def __init__(self, margin, triplet_selector):
    #     super(OnlineTripletLoss, self).__init__()
    #     self.margin = margin
    #     self.triplet_selector = triplet_selector
    #
    # def forward(self, embeddings, target):
    #
    #     triplets = self.triplet_selector.get_triplets(embeddings, target)
    #
    #     if embeddings.is_cuda:
    #         triplets = triplets.cuda()
    #
    #     ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
    #     an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
    #     losses = F.relu(ap_distances - an_distances + self.margin)
    #
    #     return losses.mean(), len(triplets)
