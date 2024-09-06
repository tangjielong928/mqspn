import torch
from scipy.optimize import linear_sum_assignment
# from lapsolver import solve_dense
# from .lap import auction_lap
from torch import nn

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_span: float = 1, cost_region: float = 1, solver = "hungarian"):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.cost_region = cost_region
        self.solver = solver

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        if self.solver == "order":
            sizes = targets["sizes"]
            indices = [(list(range(size)),list(range(size))) for size in sizes]
        else:
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(dim=-1) # [batch_size * num_queries, 5]

            entity_left = outputs["pred_left"].flatten(0, 1)
            entity_right = outputs["pred_right"].flatten(0, 1) # [batch_size * num_queries]
            entity_region = outputs["pred_region"].flatten(0, 1)


            gt_ids = targets["labels"]
            gt_left = targets["gt_left"]
            gt_right = targets["gt_right"]
            gt_region = targets["gt_region"]
            
            # import pdb;pdb.set_trace()
            cost_class = -out_prob[:, gt_ids]
            cost_span = -(entity_left[:, gt_left] + entity_right[:, gt_right])
            cost_region = -entity_region[:, gt_region]

            # Final cost matrix
            C = self.cost_span * cost_span + self.cost_class * cost_class + self.cost_region * cost_region

            C = C.view(bs, num_queries, -1)

            sizes = targets["sizes"]
            indices = None
            
            if self.solver == "hungarian":
                C = C.cpu()
                indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            if self.solver == "auction":
                indices = [auction_lap(c[i])[:2] for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def auction_lap(X, eps=None, compute_score=True):
    device = X.device
    if X.size(0) == 0 or X.size(1) == 0:
        return torch.tensor([], device=device), torch.tensor([], device=device), 0

    X = -X
    flag = 0
    if X.size(0) > X.size(1):
        flag = 1
        X = X.transpose(0, 1)
    eps = 1 / X.shape[0] if eps is None else eps

    cost = torch.zeros((1, X.shape[1]), device=device)
    curr_ass = torch.zeros(X.shape[0], device=device).long() - 1
    bids = torch.zeros(X.shape, device=device)

    counter = 0
    while (curr_ass == -1).any():
        counter += 1

        # bidding

        unassigned = (curr_ass == -1).nonzero().squeeze(-1)
        value = X[unassigned] - cost
        top_value, top_idx = value.topk(2, dim=1)

        first_idx = top_idx[:, 0]
        first_value, second_value = top_value[:, 0], top_value[:, 1]

        bid_increments = first_value - second_value + eps

        bids_ = bids[unassigned]
        bids_.zero_()
        bids_.scatter_(
            dim=1,
            index=first_idx.contiguous().view(-1, 1),
            src=bid_increments.view(-1, 1)
        )

        # assignment

        have_bidder = (bids_ > 0).int().sum(dim=0).nonzero()

        high_bids, high_bidders = bids_[:, have_bidder].max(dim=0)
        high_bidders = unassigned[high_bidders.squeeze()]

        cost[:, have_bidder] += high_bids

        curr_ass[(curr_ass.view(-1, 1) == have_bidder.view(1, -1)).sum(dim=1)] = -1
        curr_ass[high_bidders] = have_bidder.squeeze()

    score = None
    if compute_score:
        score = int(X.gather(dim=1, index=curr_ass.view(-1, 1)).sum())

    if flag == 1:
        return curr_ass, torch.arange(X.size(0), device=device), -score
    return torch.arange(X.size(0), device=device), curr_ass, -score
    # return curr_ass, score, counter

class SinkhornDistance(torch.nn.Module):
    r"""
        Given two empirical measures each with :math:`P_1` locations
        :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
        outputs an approximation of the regularized OT cost for point clouds.
        Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of
        elements in the output, 'sum': the output will be summed. Default: 'none'
        Shape:
            - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
            - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps=1e-3, max_iter=100, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, mu, nu, C):
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)

        # Sinkhorn iterations
        for i in range(self.max_iter):
            v = self.eps * \
                (torch.log(
                    nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            u = self.eps * \
                (torch.log(
                    mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(
            self.M(C, U, V)).detach()
        # Sinkhorn distance
        cost = torch.sum(
            pi * C, dim=(-2, -1))
        return cost, pi

    def M(self, C, u, v):
        '''
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / epsilon$"
        '''
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps